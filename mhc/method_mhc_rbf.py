#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010 Christian Widmer
# Copyright (C) 2010 Max-Planck-Society

"""
Created on 18.05.2010
@author: Christian Widmer
@summary: Implementation of MKL MTL with WDK_RBF
"""


import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight
from shogun.Kernel import MultitaskKernelNormalizer, KernelNormalizerToMultitaskKernelNormalizer
from base_method import MultiMethod, PreparedMultitaskData

from helper import Options

import numpy






class SequencesHandlerRbf(object):
    
    
    def __init__(self, degree, sigma, active_set, wdk_rbf_on):
        '''
        loads data into handler
        '''
    
        self.active_set = active_set
        
        fn = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHCsequenzen/pseudo.txt"
        
        tmp_key = ""
        tmp_idx = 0
        
        self.seqs = []
        self.keys = []
        self.name_to_id = {}
        
        # parse file
        for line in file(fn):
            
            if line.startswith(">"):
                tmp_key = line.strip()[1:]
            else:
           
                if active_set.count(tmp_key) > 0:
                
                    assert self.keys.count(tmp_key) == 0, "key %s is already contained in self.keys" % (tmp_key)
                    
                    self.seqs.append(line.strip())
                    self.keys.append(tmp_key)
                    self.name_to_id[tmp_key] = tmp_idx
            
                    tmp_idx += 1
            
                    assert len(self.seqs) == tmp_idx, "incorrect number of sequences %i != %i" % (len(self.seqs), tmp_idx)
                    assert len(self.keys) == tmp_idx, "incorrect number of keys %i != %i" % (len(self.keys), tmp_idx)
        
            
        
        # setup kernel
        param = Options()
        
        if wdk_rbf_on:
            param.kernel = "WeightedDegreeRBFKernel"
        else:
            param.kernel = "WeightedDegreeStringKernel"
        param.wdk_degree = degree
        param.transform = sigma
        
        self.kernel = shogun_factory.create_kernel(self.seqs, param)
        
        #######################
        # compute kernel
        #######################
        
        num_tasks = len(self.seqs)
        
        self.similarity = numpy.zeros((num_tasks, num_tasks))
        
        for i in xrange(num_tasks):
            for j in xrange(num_tasks):
                self.similarity[i,j] = self.kernel.kernel(i, j)
                
        # normalize kernel
        my_min = numpy.min(self.similarity)
        my_max = numpy.max(self.similarity)
        my_diff = my_max - my_min
    
        # scale to interval [0,1]    
        #self.similarity = (self.similarity - my_min) / my_diff
        self.similarity = (self.similarity) / my_max
    
        print self.similarity
                
                
    def get_similarity(self, task_name_lhs, task_name_rhs):
        '''
        computes position specific similarities between task pseudo-sequences
        
        @param task_name_lhs: name of task on left hand side
        @param task_name_rhs: name of task on right hand side
        
        @return: task similarity according to WDK_RBF
        '''
        
        idx_lhs = self.name_to_id[task_name_lhs]
        idx_rhs = self.name_to_id[task_name_rhs]
        
        #return self.kernel.kernel(idx_lhs, idx_rhs)
        
        return self.similarity[idx_lhs, idx_rhs]

    

class Method(MultiMethod):



    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        """
        
          
        # merge data sets
        data = PreparedMultitaskData(train_data, shuffle=False)
        
        
        # create shogun data objects
        base_wdk = shogun_factory.create_kernel(data.examples, param)
        lab = shogun_factory.create_labels(data.labels)

        # set normalizer
        normalizer = MultitaskKernelNormalizer(data.task_vector_nums)

        ########################################################
        print "creating a kernel for each node:"
        ########################################################

        
        # init seq handler 
        task_kernel = SequencesHandlerRbf(1, param.base_similarity, data.get_task_names(), param.flags["wdk_rbf_on"])
        similarities = numpy.zeros((data.get_num_tasks(), data.get_num_tasks()))
        
        # convert distance to similarity
        for task_name_lhs in data.get_task_names():
            for task_name_rhs in data.get_task_names():
                
                
                 
                
                # convert similarity with simple transformation
                similarity = task_kernel.get_similarity(task_name_lhs, task_name_rhs)
                
                print similarity
                
                print "similarity (%s,%s)=%f" % (task_name_lhs, task_name_rhs, similarity)
                
                normalizer.set_task_similarity(data.name_to_id(task_name_lhs), data.name_to_id(task_name_rhs), similarity)
                
                # save for later
                similarities[data.name_to_id(task_name_lhs),data.name_to_id(task_name_rhs)] = similarity
                
                
        # set normalizer                
        base_wdk.set_normalizer(normalizer)
        base_wdk.init_normalizer()
        

        # set up svm
        svm = SVMLight(param.cost, base_wdk, lab)
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)
        
        
        # normalize cost
        norm_c_pos = param.cost / float(len([l for l in data.labels if l==1]))
        norm_c_neg = param.cost / float(len([l for l in data.labels if l==-1]))
        
        svm.set_C(norm_c_neg, norm_c_pos)
        
        
        # start training
        svm.train()


        # save additional information
        self.additional_information["svm objective"] = svm.get_objective()
        self.additional_information["num sv"] = svm.get_num_support_vectors()
        #self.additional_information["distances"] = distances
        self.additional_information["similarities"] = similarities


        # wrap up predictors
        svms = {}
        
        # use a reference to the same svm several times
        for task_name in data.get_task_names():
            
            task_num = data.name_to_id(task_name)
            
            # save svm and task_num
            svms[task_name] = (task_num, param, svm)

        return svms
    
    

    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: dict<str, tuple<SVM, int> >
        @param examples: list of examples
        @type examples: list<str> 
        @param task_id: task identifier
        @type task_id: str
        """

        (task_num, param, svm) = predictor

        # shogun data
        feat = shogun_factory.create_features(examples, param)

        # fetch kernel normalizer & update task vector
        normalizer = svm.get_kernel().get_normalizer()
        
        # cast using dedicated SWIG-helper function
        normalizer = KernelNormalizerToMultitaskKernelNormalizer(normalizer)
        
        # set task vector
        normalizer.set_task_vector_rhs([task_num]*len(examples))
        
        # predict
        out = svm.classify(feat).get_labels()
        
        
        return out




def training_for_sigma(sigma):

    print "starting debugging:"


    from expenv import MultiSplitSet
        
    # select dataset
    multi_split_set = MultiSplitSet.get(393)

    SPLIT_POINTER = 1
    
    #create mock param object by freezable struct
    param = Options()
    param.kernel =  "WeightedDegreeStringKernel" #"WeightedDegreeRBFKernel" # #
    param.wdk_degree = 2
    param.cost = 1.0
    param.transform = 1.0 
    param.id = 666
    param.base_similarity = sigma
    param.degree = 2
    param.flags = {}
    
    param.flags["wdk_rbf_on"] = False   
    param.freeze()
    

    data_train = multi_split_set.get_train_data(SPLIT_POINTER)
    data_eval = multi_split_set.get_eval_data(SPLIT_POINTER)


    # train
    mymethod = Method(param)
    mymethod.train(data_train)

    print "training done"

    assessment = mymethod.evaluate(data_eval)
    
    print assessment
    
    assessment.destroySelf()



    return assessment.auROC


def training_simple():

    print "starting debugging:"


    from expenv import MultiSplitSet
        
    # select dataset
    multi_split_set = MultiSplitSet.get(399)

    SPLIT_POINTER = 1
    
    #create mock param object by freezable struct
    param = Options()
    param.kernel =  "WeightedDegreeStringKernel" #"WeightedDegreeRBFKernel" # #
    param.wdk_degree = 2
    param.cost = 1.0
    param.transform = 1.0 
    param.id = 666
    param.base_similarity = 1.0
    param.degree = 2
    param.flags = {}
    
    param.flags["wdk_rbf_on"] = False    
    param.freeze()
    

    data_train = multi_split_set.get_train_data(SPLIT_POINTER)
    data_eval = multi_split_set.get_eval_data(SPLIT_POINTER)


    from method_mhc_simple import Method as Simple

    # train
    mymethod_simple = Simple(param)
    mymethod_simple.train(data_train)
    assessment_simple = mymethod_simple.evaluate(data_eval)
    assessment_simple.destroySelf()


    return assessment_simple.auROC


def main():

    sigmas = [float(c) for c in numpy.linspace(0.0001, 40, 10)]
    
    #performances_simple = [training_simple()]*len(sigmas)
    performances_rbf = [training_for_sigma(sigma) for sigma in sigmas]
    

    print sigmas
    #print performances_simple
    print performances_rbf

    import pylab
    
    pylab.plot(sigmas, performances_rbf)
    #pylab.plot(sigmas, performances_simple)
    pylab.show()

    
if __name__ == "__main__":
    main()
    