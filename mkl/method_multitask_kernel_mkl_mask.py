#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010 Christian Widmer
# Copyright (C) 2010 Max-Planck-Society

"""
Created on 30.01.2010
@author: Christian Widmer
@summary: Implementation of MKL MTL with CombinedKernel
"""


import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight, MKLClassification, ST_CPLEX, ST_NEWTON, ST_DIRECT, ST_GLPK
from shogun.Kernel import MultitaskKernelMaskNormalizer, KernelNormalizerToMultitaskKernelMaskNormalizer, CombinedKernel
from shogun.Features import CombinedFeatures
from base_method import MultiMethod, PreparedMultitaskData

import shogun

import numpy



class Method(MultiMethod):


    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        """
        

        #numpy.random.seed(1337)
        numpy.random.seed(666)

        # merge data sets
        data = PreparedMultitaskData(train_data, shuffle=True)

                
        # create shogun label
        lab = shogun_factory.create_labels(data.labels)


        # assemble combined kernel
        combined_kernel = CombinedKernel()
        combined_kernel.io.set_loglevel(shogun.Kernel.MSG_DEBUG)    
        # set kernel cache
        if param.flags.has_key("cache_size"):
            combined_kernel.set_cache_size(param.flags["cache_size"])
        

        # create features
        base_features = shogun_factory.create_features(data.examples)
        
        combined_features = CombinedFeatures()
        


        ########################################################
        print "creating a masked kernel for each node:"
        ########################################################
        

        # fetch taxonomy from parameter object
        taxonomy = param.taxonomy.data

        # create name to leaf map
        nodes = taxonomy.get_all_nodes()

        
        for node in nodes:
            
            print "creating kernel for ", node.name
            
            # fetch sub-tree
            active_task_ids = [data.name_to_id(leaf.name) for leaf in node.get_leaves()]
            
            print "masking all entries other than:", active_task_ids
            
        
            # create mask-based normalizer
            normalizer = MultitaskKernelMaskNormalizer(data.task_vector_nums, data.task_vector_nums, active_task_ids)
            
            # normalize trace
            if param.flags.has_key("normalize_trace") and param.flags["normalize_trace"]:
                norm_factor = len(node.get_leaves()) / len(active_task_ids)
                normalizer.set_normalization_constant(norm_factor)
            
            # create kernel
            kernel = shogun_factory.create_empty_kernel(param)
            kernel.set_normalizer(normalizer)
            
            
            # append current kernel to CombinedKernel
            combined_kernel.append_kernel(kernel)
        
            # append features
            combined_features.append_feature_obj(base_features)

            print "------"
        

        combined_kernel.init(combined_features, combined_features)                
        #combined_kernel.precompute_subkernels()
                
        print "subkernel weights:", combined_kernel.get_subkernel_weights()

        svm = None
                        
        print "using MKL:", (param.flags["mkl_q"] >= 1.0)

        
        if param.flags["mkl_q"] >= 1.0:
            
            # set up MKL    
            svm = MKLClassification()

            # set the "q" in q-norm MKL
            svm.set_mkl_norm(param.flags["mkl_q"])
            
            # set interleaved optimization
            if param.flags.has_key("interleaved"):
                svm.set_interleaved_optimization_enabled(param.flags["interleaved"])
            
            # set solver type
            if param.flags.has_key("solver_type") and param.flags["solver_type"]:
                if param.flags["solver_type"] == "ST_CPLEX":
                    svm.set_solver_type(ST_CPLEX)
                if param.flags["solver_type"] == "ST_DIRECT":
                    svm.set_solver_type(ST_DIRECT)
                if param.flags["solver_type"] == "ST_NEWTON":
                    svm.set_solver_type(ST_NEWTON)
                if param.flags["solver_type"] == "ST_GLPK":
                    svm.set_solver_type(ST_GLPK)
            
            svm.set_kernel(combined_kernel)
            svm.set_labels(lab)
            
        else:
            # create vanilla SVM 
            svm = SVMLight(param.cost, combined_kernel, lab)


        # optimization settings
        num_threads = 4
        svm.parallel.set_num_threads(num_threads)
        
        if param.flags.has_key("epsilon"):
            svm.set_epsilon(param.flags["epsilon"])
        
        
        # enable output        
        svm.io.enable_progress()
        svm.io.set_loglevel(shogun.Classifier.MSG_DEBUG)
        
        
        # disable unsupported optimizations (due to special normalizer)
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)
        
        
        # set cost
        if param.flags["normalize_cost"]:
            
            norm_c_pos = param.cost / float(len([l for l in data.labels if l==1]))
            norm_c_neg = param.cost / float(len([l for l in data.labels if l==-1]))
            svm.set_C(norm_c_neg, norm_c_pos)
            
        else:
            
            svm.set_C(param.cost, param.cost)
        
        
        # start training
        svm.train()


        ########################################################
        print "svm objective:"
        print svm.get_objective()
        ########################################################
        
        # store additional info
        self.additional_information["svm objective"] = svm.get_objective()
        self.additional_information["weights"] = combined_kernel.get_subkernel_weights()
        
        
        # wrap up predictors
        svms = {}
            
        # use a reference to the same svm several times
        for task_name in train_data.keys():
            svms[task_name] = (data.name_to_id(task_name), len(nodes), combined_kernel, svm)

        
        return svms
    
    

    def _predict(self, predictor, examples, task_name):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor (task_id, num_nodes, combined_kernel, predictor)
        @type predictor: tuple<int, int, CombinedKernel, SVM>
        @param examples: list of examples
        @type examples: list<object>
        @param task_name: task name
        @type task_name: str
        """

        (task_id, num_nodes, combined_kernel, svm) = predictor

        # shogun data
        base_feat = shogun_factory.create_features(examples)
                
        # construct combined kernel
        feat = CombinedFeatures()
        
        for i in xrange(num_nodes):
            feat.append_feature_obj(base_feat)

            # fetch kernel normalizer
            normalizer = combined_kernel.get_kernel(i).get_normalizer()
            
            # cast using dedicated SWIG-helper function
            normalizer = KernelNormalizerToMultitaskKernelMaskNormalizer(normalizer)
            
            # set task vector
            normalizer.set_task_vector_rhs([task_id]*len(examples))


        combined_kernel = svm.get_kernel()
        combined_kernel.init(combined_kernel.get_lhs(), feat)
        
        # predict
        out = svm.classify().get_labels()
        
        return out


        

def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = -1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(382)

    #dataset_name = multi_split_set.description

    
    #create mock param object by freezable struct
    param = Options()
    #param.kernel = "PolyKernel"
    param.kernel = "WeightedDegreeStringKernel"
    param.wdk_degree = 1
    param.cost = 100
    param.transform = 0 #2.0
    param.taxonomy = multi_split_set.taxonomy
    param.id = 666
    
    param.freeze()
    

    data_train = multi_split_set.get_train_data(SPLIT_POINTER)
    data_eval = multi_split_set.get_eval_data(SPLIT_POINTER)


    # train
    mymethod = Method(param)
    mymethod.train(data_train)


    assessment = mymethod.evaluate(data_eval)
    
    print assessment
    
    assessment.destroySelf()
    
    
if __name__ == "__main__":
    main()

    
