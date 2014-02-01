#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 06.01.2010
@author: Christian Widmer
@summary: Implementation of the augmented SVM multitask method
This methods uses a modified kernel such that tasks, 
which are close to each other are more similar by default.
"""


import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight, MKLClassification, ST_CPLEX, ST_NEWTON, ST_DIRECT, ST_GLPK
from shogun.Kernel import MultitaskKernelTreeNormalizer, KernelNormalizerToMultitaskKernelTreeNormalizer
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
        
                
        # merge data sets
        data = PreparedMultitaskData(train_data, shuffle=False)
        
        # create shogun data objects
        base_wdk = shogun_factory.create_kernel(data.examples, param)
        lab = shogun_factory.create_labels(data.labels)

        # fetch taxonomy from parameter object
        taxonomy = shogun_factory.create_taxonomy(param.taxonomy.data)


        # set normalizer
        normalizer = MultitaskKernelTreeNormalizer(data.task_vector_names, data.task_vector_names, taxonomy)
        
        
        ########################################################
        gammas = self.taxonomy_to_gammas(data, taxonomy)
        print "gammas before MKL:"
        print gammas
        ########################################################
        
        
        base_wdk.set_normalizer(normalizer)
        base_wdk.init_normalizer()

        svm = None
        
        num_subk = base_wdk.get_num_subkernels()
        
        print "num subkernels:", num_subk
        
        #print "subkernel weights:", base_wdk.get_subkernel_weights()
        
        self.additional_information["weights_before"] = [normalizer.get_beta(i) for i in range(num_subk)]        
        
        print "using MKL:", (param.transform >= 1.0)
        
        if param.transform >= 1.0:
        
        
            num_threads = 4
            
            svm = MKLClassification()
            
            svm.set_mkl_norm(param.transform)
            #svm.set_solver_type(ST_CPLEX) #GLPK) #DIRECT) #NEWTON)#ST_CPLEX) 
        
            
            svm.set_kernel(base_wdk)
            svm.set_labels(lab)
            
            svm.parallel.set_num_threads(num_threads)
            svm.set_linadd_enabled(False)
            svm.set_batch_computation_enabled(False)


            if param.flags["normalize_cost"]:        
                # normalize cost
                norm_c_pos = param.cost / float(len([l for l in data.labels if l==1]))
                norm_c_neg = param.cost / float(len([l for l in data.labels if l==-1]))
                
                svm.set_C(norm_c_neg, norm_c_pos)
            else:
                svm.set_C(param.cost, param.cost)
            
            
            svm.train()
        
            #print "subkernel weights (after):", base_wdk.get_subkernel_weights()    
            
        else:
            
            # create SVM (disable unsupported optimizations)
            svm = SVMLight(param.cost, base_wdk, lab)
            svm.set_linadd_enabled(False)
            svm.set_batch_computation_enabled(False)
            
            svm.train()
        
        
        print "svm objective:", svm.get_objective()     
        


        self.additional_information["weights"] = [normalizer.get_beta(i) for i in range(num_subk)]
        self.additional_information["gammas"] = self.taxonomy_to_gammas(data, taxonomy) 
       
        print "debug weights:"
        print self.additional_information
        print ""
        
        
        # wrap up predictors
        svms = {}
            
        # use a reference to the same svm several times
        for task_id in train_data.keys():
            svms[task_id] = svm


        return svms
    
    

    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: dict<str, tuple<SVMLight, Kernel, int> >
        @param examples: list of examples
        @type examples: list<str> 
        @param task_id: task id
        @type task_id: str
        """

        svm = predictor

        # shogun data
        feat = shogun_factory.create_features(examples)

        # get objects
        #kernel = svm.get_kernel()

        # fetch kernel normalizer
        normalizer = svm.get_kernel().get_normalizer()
        
        # cast using dedicated SWIG-helper function
        normalizer = KernelNormalizerToMultitaskKernelTreeNormalizer(normalizer)
        
        # set task vector
        normalizer.set_task_vector_rhs([str(task_id)]*len(examples))

        # init kernel
        #kernel.init_normalizer()
        #kernel.init(kernel.get_lhs(), feat) 

        # predict
        out = svm.classify(feat).get_labels()
        
        
        return out



    def taxonomy_to_gammas(self, data, taxonomy):
        '''
        helper function to convert shogun taxonomy to gamma matrix
        
        @param taxonomy:
        '''
        

        gammas = numpy.zeros((data.get_num_tasks(), data.get_num_tasks()))
        
        for t1_name in data.get_task_names():
            for t2_name in data.get_task_names():
                
                similarity = taxonomy.compute_node_similarity(taxonomy.get_id(t1_name), taxonomy.get_id(t2_name))        
                gammas[data.name_to_id(t1_name), data.name_to_id(t2_name)] = similarity
        
        
        return gammas
        
        


def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = 1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(379)

    dataset_name = multi_split_set.description

    print "dataset_name", dataset_name
        
    
    #create mock param object by freezable struct
    param = Options()
    param.kernel = "WeightedDegreeStringKernel"
    param.wdk_degree = 1
    param.cost = 1.0
    param.transform = 1.0
    param.taxonomy = multi_split_set.taxonomy
    param.id = 666
    
    param.freeze()
    

    data_train = multi_split_set.get_train_data(SPLIT_POINTER)
    data_eval = multi_split_set.get_eval_data(SPLIT_POINTER)

    # train hierarchical xval
    mymethod = Method(param)
    mymethod.train(data_train)
    
    assessment = mymethod.evaluate(data_eval)
    
    print assessment
    
    assessment.destroySelf();
    
    
    
if __name__ == "__main__":
    main()

