#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010 Christian Widmer
# Copyright (C) 2010 Max-Planck-Society

"""
Created on 23.02.2010
@author: Christian Widmer
@summary: Implementation of MKL MTL with CombinedKernel
"""


import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight, MKLClassification, ST_CPLEX, ST_NEWTON, ST_DIRECT, ST_GLPK
from shogun.Kernel import MultitaskKernelMaskPairNormalizer, KernelNormalizerToMultitaskKernelMaskPairNormalizer, CombinedKernel
from shogun.Kernel import Pairii, PairiiVec
from shogun.Features import CombinedFeatures
from base_method import MultiMethod, PreparedMultitaskData

import shogun

class Method(MultiMethod):



    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        """
        
        # dict to save additional information for later analysis
        self.additional_information = {}
        
          
        # merge data sets
        data = PreparedMultitaskData(train_data, shuffle=True)

                
        # create shogun label
        lab = shogun_factory.create_labels(data.labels)
        


        ########################################################
        print "creating a kernel for each node:"
        ########################################################


        # assemble combined kernel
        
        combined_kernel = CombinedKernel()
        
        combined_kernel.io.set_loglevel(shogun.Kernel.MSG_INFO)
        
        
        base_features = shogun_factory.create_features(data.examples, param)
        
        combined_features = CombinedFeatures()
        
        
        
        
        ##################################################
        # intra-domain blocks (dirac kernel)
        
        
        intra_block_vec = PairiiVec()
        
        for task_id in data.get_task_ids():
            intra_block_vec.push_back(Pairii(task_id, task_id))
        
        
        
        # create mask-based normalizer
        normalizer = MultitaskKernelMaskPairNormalizer(data.task_vector_nums, intra_block_vec)        
        kernel = shogun_factory.create_empty_kernel(param)
        kernel.set_normalizer(normalizer)
        
        # append current kernel to CombinedKernel
        combined_kernel.append_kernel(kernel)
    
        # append features
        combined_features.append_feature_obj(base_features)

        print "------"
        
        ##################################################
        # all blocks (full kernel matrix)
        
        
        all_block_vec = PairiiVec()
        
        for task_id_1 in data.get_task_ids():
            for task_id_2 in data.get_task_ids():
                all_block_vec.push_back(Pairii(task_id_1, task_id_2))
                
        
        # create mask-based normalizer
        normalizer_all = MultitaskKernelMaskPairNormalizer(data.task_vector_nums, all_block_vec)        
        kernel_all = shogun_factory.create_empty_kernel(param)
        kernel_all.set_normalizer(normalizer_all)
                
        # append current kernel to CombinedKernel
        combined_kernel.append_kernel(kernel_all)
    
        # append features
        combined_features.append_feature_obj(base_features)

        
        ##################################################
        # hack
        
        
        #        hack_block_vec = PairiiVec()
        #        
        #        for task_id_1 in data.get_task_ids():
        #            for task_id_2 in data.get_task_ids():
        #                hack_block_vec.push_back(Pairii(task_id_1, task_id_2))
        #        
        #        hack_block_vec.push_back(Pairii(data.name_to_id("B_2705"), data.name_to_id("B_4001")))
        #        other_group = ["B_0702", "B_1501", "B_5801"]
        #        for task_id_1 in other_group:
        #            for task_id_2 in other_group:
        #                hack_block_vec.push_back(Pairii(data.name_to_id(task_id_1), data.name_to_id(task_id_2)))
        #        
        #        
        #        
        #        # create mask-based normalizer
        #        normalizer_hack = MultitaskKernelMaskPairNormalizer(data.task_vector_nums, hack_block_vec)        
        #        kernel_hack = shogun_factory.create_empty_kernel(param)
        #        kernel_hack.set_normalizer(normalizer_hack)
        #                
        #        # append current kernel to CombinedKernel
        #        combined_kernel.append_kernel(kernel_hack)
        #    
        #        # append features
        #        combined_features.append_feature_obj(base_features)
        
        
        
            
        ##################################################
        # init combined kernel
        
        combined_kernel.init(combined_features, combined_features)    
        
            
        #combined_kernel.precompute_subkernels()
        self.additional_information["mkl weights before"] = combined_kernel.get_subkernel_weights()
        
        print "subkernel weights:", combined_kernel.get_subkernel_weights()

        svm = None
                
        
        print "using MKL:", (param.flags["mkl_q"] >= 1.0)
        
        if param.flags["mkl_q"] >= 1.0:
            
            svm = MKLClassification()
            
            svm.set_mkl_norm(param.flags["mkl_q"])
            svm.set_kernel(combined_kernel)
            svm.set_labels(lab)
        
        else:
            
            # create SVM (disable unsupported optimizations)
            combined_kernel.set_cache_size(500)
            svm = SVMLight(param.cost, combined_kernel, lab)


        num_threads = 8
        svm.io.enable_progress()
        svm.io.set_loglevel(shogun.Classifier.MSG_INFO)
        
        svm.parallel.set_num_threads(num_threads)
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)
    
        svm.set_epsilon(0.03)
        
        # set cost
        if param.flags["normalize_cost"]:
            
            norm_c_pos = param.cost / float(len([l for l in data.labels if l==1]))
            norm_c_neg = param.cost / float(len([l for l in data.labels if l==-1]))
            svm.set_C(norm_c_neg, norm_c_pos)
            
        else:

            svm.set_C(param.cost, param.cost)
        
        svm.train()
    
    
        print "subkernel weights (after):", combined_kernel.get_subkernel_weights()

        ########################################################
        print "svm objective:"
        print svm.get_objective()
        
        
        self.additional_information["svm_objective"] = svm.get_objective()
        self.additional_information["svm num sv"] = svm.get_num_support_vectors()
        self.additional_information["mkl weights post-training"] = combined_kernel.get_subkernel_weights()
         
        ########################################################
        
        
        # wrap up predictors
        svms = {}
            
        # use a reference to the same svm several times
        for task_name in train_data.keys():
            svms[task_name] = (data.name_to_id(task_name), combined_kernel, svm, param)

        
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

        (task_id, combined_kernel, svm, param) = predictor

        # shogun data
        base_feat = shogun_factory.create_features(examples, param)
                
        # construct combined kernel
        feat = CombinedFeatures()
        
        for i in xrange(combined_kernel.get_num_subkernels()):
            feat.append_feature_obj(base_feat)

            # fetch kernel normalizer
            normalizer = combined_kernel.get_kernel(i).get_normalizer()
            
            # cast using dedicated SWIG-helper function
            normalizer = KernelNormalizerToMultitaskKernelMaskPairNormalizer(normalizer)
            
            # set task vector
            normalizer.set_task_vector_rhs([task_id]*len(examples))


        combined_kernel = svm.get_kernel()
        combined_kernel.init(combined_kernel.get_lhs(), feat)
        
        # predict
        out = svm.classify().get_labels()

        # predict
        #out = svm.classify(feat).get_labels()
        
        
        return out


        

def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = -1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    #multi_split_set = MultiSplitSet.get(387)
    multi_split_set = MultiSplitSet.get(386)

    #dataset_name = multi_split_set.description

    
    # create mock param object by freezable struct
    param = Options()
    param.kernel = "WeightedDegreeStringKernel"#"PolyKernel" 
    param.wdk_degree = 1
    param.cost = 100
    param.transform = 2 #2.0
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

    