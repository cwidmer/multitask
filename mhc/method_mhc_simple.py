#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 05.03.2010
@author: Christian Widmer
@summary: Implementation of the augmented SVM multitask method
This methods uses a modified kernel such that tasks, 
which are close to each other are more similar by default.
"""

import unittest

import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight
from shogun.Kernel import MultitaskKernelNormalizer, KernelNormalizerToMultitaskKernelNormalizer
from base_method import MultiMethod, PreparedMultitaskData
import numpy


 
debug = False



class Method(MultiMethod):



    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        """


        assert(param.base_similarity >= 1)
        
        # merge data sets
        data = PreparedMultitaskData(train_data, shuffle=False)
        
        
        # create shogun data objects
        base_wdk = shogun_factory.create_kernel(data.examples, param)
        lab = shogun_factory.create_labels(data.labels)

        # set normalizer
        normalizer = MultitaskKernelNormalizer(data.task_vector_nums)
        
        # load data
        #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_PseudoSeq_BlosumEnc_pearson.txt")
        f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/All_PseudoSeq_Hamming.txt")
        #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_PseudoSeq_BlosumEnc_euklid.txt")
        #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_RAxML.txt")
        
        num_lines = int(f.readline().strip())
        task_distances = numpy.zeros((num_lines, num_lines))
        name_to_id = {}
        for (i, line) in enumerate(f):
            tokens = line.strip().split("\t")
            name = str(tokens[0])
            name_to_id[name] = i
            entry = numpy.array([v for (j,v) in enumerate(tokens) if j!=0])
            assert len(entry)==num_lines, "len_entry %i, num_lines %i" % (len(entry), num_lines)
            task_distances[i,:] = entry
            
        
        # cut relevant submatrix
        active_ids = [name_to_id[name] for name in data.get_task_names()] 
        tmp_distances = task_distances[active_ids, :]
        tmp_distances = tmp_distances[:, active_ids]
        print "distances ", tmp_distances.shape

        
        # normalize distances
        task_distances = task_distances / numpy.max(tmp_distances)
        
        
        similarities = numpy.zeros((data.get_num_tasks(), data.get_num_tasks()))
                                
        
        # convert distance to similarity
        for task_name_lhs in data.get_task_names():
            for task_name_rhs in data.get_task_names():
                
                
                # convert similarity with simple transformation
                similarity = param.base_similarity - task_distances[name_to_id[task_name_lhs], name_to_id[task_name_rhs]]
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




def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = 1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(399)

    
    #create mock param object by freezable struct
    param = Options()
    param.kernel =  "WeightedDegreeRBFKernel" #"WeightedDegreeStringKernel"# #
    param.wdk_degree = 1
    param.cost = 1.0
    param.transform = 1.0
    param.sigma = 1.0
    param.id = 666
    param.base_similarity = 1
    param.degree = 2
    
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



    
if __name__ == "__main__":
    main()
    
