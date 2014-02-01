#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010 Christian Widmer
# Copyright (C) 2010 Max-Planck-Society

"""
Created on 30.03.2010
@author: Christian Widmer
@summary: Implementation of the augmented SVM multitask method
This methods uses a modified kernel such that tasks, 
which are close to each other are more similar by default.
"""

import unittest

import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight, MKLClassification
from shogun.Kernel import CombinedKernel, MultitaskKernelNormalizer, KernelNormalizerToMultitaskKernelNormalizer
from shogun.Kernel import Pairii, PairiiVec, MultitaskKernelMaskPairNormalizer, KernelNormalizerToMultitaskKernelMaskPairNormalizer
from shogun.Features import CombinedFeatures
from base_method import MultiMethod, PreparedMultitaskData
import numpy
import shogun

 
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
        base_wdk = shogun_factory.create_empty_kernel(param)
        lab = shogun_factory.create_labels(data.labels)

        combined_kernel = CombinedKernel()
        combined_kernel.io.set_loglevel(shogun.Kernel.MSG_INFO)
        base_features = shogun_factory.create_features(data.examples)
        combined_features = CombinedFeatures()
        
        


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
        #base_wdk.init_normalizer()
        

        combined_features.append_feature_obj(base_features)
        combined_kernel.append_kernel(base_wdk)
        
        
        ##################################################
        # intra-domain blocks
        

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

        # set mixing factor (used if MKL is OFF)
        assert(param.base_similarity <= 1)
        assert(param.base_similarity >= 0)
        combined_kernel.set_subkernel_weights([param.base_similarity, 1-param.base_similarity])

        combined_kernel.init(combined_features, combined_features)

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


        # save additional information
        self.additional_information["svm objective"] = svm.get_objective()
        self.additional_information["num sv"] = svm.get_num_support_vectors()
        self.additional_information["similarities"] = similarities
        self.additional_information["post_weights"] = combined_kernel.get_subkernel_weights()

        # wrap up predictors
        svms = {}
        
        # use a reference to the same svm several times
        for task_name in data.get_task_names():
            
            task_num = data.name_to_id(task_name)
            
            # save svm and task_num
            svms[task_name] = (task_num, combined_kernel, svm)

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

        (task_num, combined_kernel, svm) = predictor

        # shogun data
        base_feat = shogun_factory.create_features(examples)
        feat = CombinedFeatures()
        feat.append_feature_obj(base_feat)
        feat.append_feature_obj(base_feat)
        
        # update normalizers
        normalizer = combined_kernel.get_kernel(0).get_normalizer()
        normalizer = KernelNormalizerToMultitaskKernelNormalizer(normalizer)
        normalizer.set_task_vector_rhs([task_num]*len(examples))

        normalizer_dirac = combined_kernel.get_kernel(1).get_normalizer()
        normalizer_dirac = KernelNormalizerToMultitaskKernelMaskPairNormalizer(normalizer_dirac)
        normalizer_dirac.set_task_vector_rhs([task_num]*len(examples))
                
        # predict
        out = svm.classify(feat).get_labels()
        
        
        return out



def reweight_kernel_matrix(km, task_vector, gammas):
    """
    method that computes explicit reweighting of kernel matrix
    """

    # basic sanity checks
    assert(km.shape[0]==len(task_vector))
    assert(km.shape[1]==len(task_vector))
    assert(len(set(task_vector))==len(gammas))

    N = len(task_vector)

    # weight km entries according to gammas
    for i in xrange(N):

        task_i = task_vector[i]

        for j in xrange(N):
            task_j = task_vector[j]
            weight = gammas[task_i][task_j]
            
            km[i][j] = km[i][j] * weight

    return km


class TestAugmentedTraining(unittest.TestCase):

    def setUp(self):

        import expenv

        run = expenv.Run.get(13490)
        self.instances = run.get_train_data()
        self.param = run.method.param
        
    def testtrainsimple(self):

        method_internal = Method(self.param)
        method_internal.external_task_weights = True
        preds_internal = method_internal.train(self.instances)


    def notesttraining(self):
        # make sure weighting the KM internally and externally leads to the same alphas

        method_internal = Method(self.param)
        method_internal.external_task_weights = False
        preds_internal = method_internal.train(self.instances)

        method_external = Method(self.param)
        method_external.external_task_weights = True
        preds_external = method_external.train(self.instances)

        self.assertEqual(len(preds_internal), len(preds_external))

        for i in xrange(len(preds_internal)):

            pred_internal = preds_internal[i]
            pred_external = preds_external[i]

            alphas_external = pred_external.get_alphas()
            alphas_internal = pred_internal.get_alphas()

            # do we get the same number of support vectors
            self.assertEqual(len(alphas_internal), len(alphas_external))

            # are the objectives similar
            self.assertAlmostEqual(pred_internal.get_objective(), pred_external.get_objective())

            # are the alphas similar
            for j in len(alphas_internal):
                self.assertAlmostEqual(alphas_internal[i], alphas_external[j])


    def notestindependet(self):
        # for certain parameters of the augmented kernel, the result should be
        # identical to the one of the plain svm approach

        #TODO implement this test

        print "implement"


    def notestaggregate(self):
        # for certain parameters of the augmented kernel, the result should be
        # identical to the aggregate (union) svm approach

        #TODO implement this test
        print "implement"




def create_plot_inner(param, data_train, data_eval):
    """
    this will create a performance plot for manually set values
    """


    # train hierarchical xval
    mymethod = Method(param)
    mymethod.train(data_train)
    
    
    for a in mymethod.evaluate(data_eval).assessments:
        print a
        a.destroySelf()
    

    #return (mymethod.debug_perf, assessment["auROC"], mymethod.debug_best_idx)



def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = 1

    from expenv import MultiSplitSet
    from helper import Options 
    from task_similarities import fetch_gammas
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(317)
    #multi_split_set = MultiSplitSet.get(374)
    #multi_split_set = MultiSplitSet.get(2)

    dataset_name = multi_split_set.description

    transform = 1.0
    base = 1.0
    similarity_matrix = fetch_gammas(transform, base, dataset_name) 
        

    #create mock taxonomy object by freezable struct
    taxonomy = Options()
    taxonomy.data = similarity_matrix
    taxonomy.description = dataset_name
    taxonomy.freeze()
    
    
    #create mock param object by freezable struct
    param = Options()
    param.kernel = "WeightedDegreeStringKernel"
    param.wdk_degree = 1
    param.cost = 1.0
    param.transform = 1.0
    param.taxonomy = taxonomy
    param.id = 666
    
    param.freeze()
    


    data_train = multi_split_set.get_train_data(SPLIT_POINTER)
    data_eval = multi_split_set.get_eval_data(SPLIT_POINTER)

    create_plot_inner(param, data_train, data_eval)

    
    
if __name__ == "__main__":
    main()
    
