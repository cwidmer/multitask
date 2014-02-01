#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010 Christian Widmer
# Copyright (C) 2010 Max-Planck-Society

"""
Created on 16.03.2010
@author: Christian Widmer
@summary: Implementation of MKL MTL with CombinedKernel
"""


import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight, MKLClassification
from shogun.Kernel import MultitaskKernelMaskPairNormalizer, KernelNormalizerToMultitaskKernelMaskPairNormalizer, CombinedKernel
from shogun.Kernel import MultitaskKernelNormalizer, KernelNormalizerToMultitaskKernelNormalizer, Pairii, PairiiVec
from shogun.Features import CombinedFeatures
from base_method import MultiMethod, PreparedMultitaskData

import shogun




class SequencesHandler(object):
    
    
    def __init__(self):
        '''
        loads data into handler
        '''
        
        fn = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHCsequenzen/pseudo.txt"
        
        tmp_key = ""
        
        self.seqs = {}
        self.seq_length = 0
        
        # parse file
        for line in file(fn):
            
            if line.startswith(">"):
                tmp_key = line.strip()[1:]
            else:
                self.seqs[tmp_key] = line.strip()
                self.seq_length = len(self.seqs[tmp_key])
        
        #print self.seqs.keys()
                
                
                
    def get_similarity(self, task_name_lhs, task_name_rhs, pos):
        '''
        computes position specific similarities between task pseudo-sequences
        
        @param task_name_lhs: name of task on left hand side
        @param task_name_rhs: name of task on right hand side
        @param pos: position to take into account
        
        @return: kroneker delta
        '''
        
        seq_lhs = self.seqs[task_name_lhs]
        seq_rhs = self.seqs[task_name_rhs]
        
        if seq_lhs[pos] == seq_rhs[pos]:
            return 1
        else:
            return 0 


    def get_seq(self, task_name):
        '''
        getter for seqs
        
        @param task_name: name of task to get
        '''
        
        return self.seqs[task_name]
    
    

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
        data = PreparedMultitaskData(train_data, shuffle=True)
        
        # create shogun label
        lab = shogun_factory.create_labels(data.labels)
        


        ########################################################
        print "creating a kernel for each node:"
        ########################################################


        # assemble combined kernel
        
        combined_kernel = CombinedKernel()
        
        combined_kernel.io.set_loglevel(shogun.Kernel.MSG_INFO)
        
        
        base_features = shogun_factory.create_features(data.examples)
        
        combined_features = CombinedFeatures()
        
        
        
        
        ##################################################
        # intra-domain blocks
        
        
        #        intra_block_vec = PairiiVec()
        #        
        #        for task_id in data.get_task_ids():
        #            intra_block_vec.push_back(Pairii(task_id, task_id))
        #        
        #        
        #        
        #        # create mask-based normalizer
        #        normalizer = MultitaskKernelMaskPairNormalizer(data.task_vector_nums, intra_block_vec)        
        #        kernel = shogun_factory.create_empty_kernel(param)
        #        kernel.set_normalizer(normalizer)
        #        
        #        # append current kernel to CombinedKernel
        #        combined_kernel.append_kernel(kernel)
        #    
        #        # append features
        #        combined_features.append_feature_obj(base_features)
        #
        #        print "------"
        #        
        #        ##################################################
        #        # all blocks
        #        
        #        
        #        all_block_vec = PairiiVec()
        #        
        #        for task_id_1 in data.get_task_ids():
        #            for task_id_2 in data.get_task_ids():
        #                all_block_vec.push_back(Pairii(task_id_1, task_id_2))
        #                
        #        
        #        # create mask-based normalizer
        #        normalizer_all = MultitaskKernelMaskPairNormalizer(data.task_vector_nums, all_block_vec)        
        #        kernel_all = shogun_factory.create_empty_kernel(param)
        #        kernel_all.set_normalizer(normalizer_all)
        #                
        #        # append current kernel to CombinedKernel
        #        combined_kernel.append_kernel(kernel_all)
        #    
        #        # append features
        #        combined_features.append_feature_obj(base_features)

        
        ##################################################
        # add one kernel per similarity position
        
        
        # init seq handler 
        pseudoseqs = SequencesHandler()
        pseudoseq_length = pseudoseqs.seq_length


        for pos in range(pseudoseq_length):
            
            print "appending kernel for pos %i" % (pos)
        
            print "nums", data.task_vector_nums

    
            pos_block_vec = PairiiVec()
    
            # set similarity
            for task_name_lhs in data.get_task_names():
                for task_name_rhs in data.get_task_names():
                    
                    similarity = pseudoseqs.get_similarity(task_name_lhs, task_name_rhs, pos)
                    #print "computing similarity for tasks (%s, %s) = %i" % (task_name_lhs, task_name_rhs, similarity)
                    
                    if similarity == 1:                    
                        tmp_pair = Pairii(data.name_to_id(task_name_lhs), data.name_to_id(task_name_rhs))
                        pos_block_vec.push_back(tmp_pair)

            print "creating normalizer"
            normalizer_pos = MultitaskKernelMaskPairNormalizer(data.task_vector_nums, pos_block_vec)   

            print "creating empty kernel"
            kernel_pos = shogun_factory.create_empty_kernel(param)
            
            print "setting normalizer"
            kernel_pos.set_normalizer(normalizer_pos)
                
            print "appending kernel"
            # append current kernel to CombinedKernel
            combined_kernel.append_kernel(kernel_pos)
    
            print "appending features"
            # append features
            combined_features.append_feature_obj(base_features)
        
        
        print "done constructing combined kernel"
        
        ##################################################
        # init combined kernel
        
        combined_kernel.init(combined_features, combined_features)    
        
            

                
        print "subkernel weights:", combined_kernel.get_subkernel_weights()

        svm = None
                
        
        print "using MKL:", (param.transform >= 1.0)
        
        if param.transform >= 1.0:
            
            svm = MKLClassification()
            
            svm.set_mkl_norm(param.transform)
            #svm.set_solver_type(ST_CPLEX) #ST_GLPK) #DIRECT) #NEWTON)#ST_CPLEX) #auto
        
            svm.set_C(param.cost, param.cost)
            
            svm.set_kernel(combined_kernel)
            svm.set_labels(lab)
            
                
        else:
            
            # create SVM (disable unsupported optimizations)
            combined_kernel.set_cache_size(500)
            
            svm = SVMLight(param.cost, combined_kernel, lab)


        # set up SVM
        num_threads = 8
        svm.io.enable_progress()
        #svm.io.set_loglevel(shogun.Classifier.MSG_INFO)
        svm.io.set_loglevel(shogun.Classifier.MSG_DEBUG)
        
        svm.parallel.set_num_threads(num_threads)
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)
        
        print "WARNING: custom epsilon set"
        svm.set_epsilon(0.05)    
        
        # normalize cost
        norm_c_pos = param.cost / float(len([l for l in data.labels if l==1]))
        norm_c_neg = param.cost / float(len([l for l in data.labels if l==-1]))
        
        svm.set_C(norm_c_neg, norm_c_pos)
        
        
        # start training
        svm.train()
    
        
        # save additional info
        self.additional_information["svm_objective"] = svm.get_objective()
        self.additional_information["svm num sv"] = svm.get_num_support_vectors()
        self.additional_information["mkl weights post-training"] = combined_kernel.get_subkernel_weights()
        
        print self.additional_information 
        
        
        
        # wrap up predictors
        svms = {}
            
        # use a reference to the same svm several times
        for task_name in train_data.keys():
            svms[task_name] = (data.name_to_id(task_name), combined_kernel, svm)

        
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

        (task_id, combined_kernel, svm) = predictor

        # shogun data
        base_feat = shogun_factory.create_features(examples)
                
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
        
        return out


def compute_similarity(task_name_lhs, task_name_rhs, pos):
    
    return 1        



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
    