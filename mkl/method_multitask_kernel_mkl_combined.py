#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 27.01.2010
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
import helper




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
        kernel_matrix = base_wdk.get_kernel_matrix()
        lab = shogun_factory.create_labels(data.labels)
        

        # fetch taxonomy from parameter object
        taxonomy = param.taxonomy.data

        # create name to leaf map
        nodes = taxonomy.get_all_nodes()


        ########################################################
        print "creating a kernel for each node:"
        ########################################################


        # assemble combined kernel
        from shogun.Kernel import CombinedKernel, CustomKernel
        
        combined_kernel = CombinedKernel()
        
        # indicator to which task each example belongs
        task_vector = data.task_vector_names
        
        for node in nodes:
            
            print "creating kernel for ", node.name
            
            # fetch sub-tree
            leaf_names = [leaf.name for leaf in node.get_leaves()]
            
            print "masking all entries other than:", leaf_names
            
            # init matrix
            kernel_matrix_node = numpy.zeros(kernel_matrix.shape)
            
            # fill matrix for node
            for (i, task_lhs) in enumerate(task_vector):
                for (j, task_rhs) in enumerate(task_vector):
                    
                    # only copy values, if both tasks are present in subtree
                    if task_lhs in leaf_names and task_rhs in leaf_names:
                        kernel_matrix_node[i,j] = kernel_matrix[i,j]
                    
            # create custom kernel
            kernel_node = CustomKernel()
            kernel_node.set_full_kernel_matrix_from_full(kernel_matrix_node)
            
            
            # append custom kernel to CombinedKernel
            combined_kernel.append_kernel(kernel_node)                
            
            print "------"
        

        print "subkernel weights:", combined_kernel.get_subkernel_weights()

        svm = None
                
        
        print "using MKL:", (param.transform >= 1.0)
        
        if param.transform >= 1.0:
        
        
            num_threads = 4

            
            svm = MKLClassification()
            
            svm.set_mkl_norm(param.transform)
            svm.set_solver_type(ST_GLPK) #DIRECT) #NEWTON)#ST_CPLEX)
        
            svm.set_C(param.cost, param.cost)
            
            svm.set_kernel(combined_kernel)
            svm.set_labels(lab)
            
            svm.parallel.set_num_threads(num_threads)
            #svm.set_linadd_enabled(False)
            #svm.set_batch_computation_enabled(False)
            
            svm.train()
        
            print "subkernel weights (after):", combined_kernel.get_subkernel_weights()    
            
        else:
            
            # create SVM (disable unsupported optimizations)
            svm = SVMLight(param.cost, combined_kernel, lab)
            svm.set_linadd_enabled(False)
            svm.set_batch_computation_enabled(False)
            
            svm.train()


        ########################################################
        print "svm objective:"
        print svm.get_objective()
        ########################################################
        
        
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

        assert False, "_predict NOT IMPLEMENTED"

        svm = predictor

        # shogun data
        feat = shogun_factory.create_features(examples)

        # get objects
        # kernel = svm.get_kernel()

        # fetch kernel normalizer
        normalizer = svm.get_kernel().get_normalizer()
        
        # cast using dedicated SWIG-helper function
        normalizer = KernelNormalizerToMultitaskKernelTreeNormalizer(normalizer)
        
        # set task vector
        normalizer.set_task_vector_rhs([str(task_id)]*len(examples))

        # init kernel
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

    #dataset_name = multi_split_set.description

    
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
    #data_eval = multi_split_set.get_eval_data(SPLIT_POINTER)


    # train hierarchical xval
    mymethod = Method(param)
    mymethod.train(data_train)

    
    
if __name__ == "__main__":
    main()

    