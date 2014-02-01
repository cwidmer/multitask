#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010-2011 Christian Widmer
# Copyright (C) 2010-2011 Max-Planck-Society

"""
Created on 06.01.2010
@author: Christian Widmer
@summary: Implementation of the augmented SVM multitask method
This methods uses a modified kernel such that tasks, 
which are close to each other are more similar by default.
"""

import unittest
import pprint

import shogun_factory_new as shogun_factory

from shogun.Kernel import MultitaskKernelNormalizer, KernelNormalizerToMultitaskKernelNormalizer
from base_method import MultiMethod, PreparedMultitaskData
import numpy
import helper
import task_similarities

 
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


        # create normalizer
        normalizer = MultitaskKernelNormalizer(data.task_vector_nums)

        # load hard-coded task-similarity
        task_similarity = helper.load("/fml/ag-raetsch/home/cwidmer/svn/projects/alt_splice_code/src/task_sim_tis.bz2")


        # set similarity
        similarities = numpy.zeros((data.get_num_tasks(), data.get_num_tasks()))
        
        for (i, task_name_lhs) in enumerate(data.get_task_names()):
            
            #max_value_row = max(task_similarity.get_row(task_name_lhs))
            max_value_row = 1.0
            
            for (j, task_name_rhs) in enumerate(data.get_task_names()):
                
                similarity = task_similarity.get_value(task_name_lhs, task_name_rhs) / max_value_row
                normalizer.set_task_similarity(i, j, similarity)
                similarities[i,j] = similarity
                
        
        pprint.pprint similarities
        
        # set normalizer
        #print "WARNING MTK disabled!!!!!!!!!!!!!!!!!!!!!"                
        base_wdk.set_normalizer(normalizer)
        base_wdk.init_normalizer()
        
        
        # set up svm
        param.flags["svm_type"] = "svmlight" #fix svm type
        
        svm = shogun_factory.create_svm(param, base_wdk, lab)
        
        # make sure these parameters are set correctly
        #print "WARNING MTK WONT WORK WITH THESE SETTINGS!!!!!!!!!!!!!!!!!!!!!"
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)
        

        assert svm.get_linadd_enabled() == False, "linadd should be disabled"
        assert svm.get_batch_computation_enabled == False, "batch compute should be disabled"
        
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
            svms[task_name] = (task_num, svm)

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

        (task_num, svm) = predictor

        # shogun data
        feat = shogun_factory.create_features(examples, self.param)

        # fetch kernel normalizer & update task vector
        normalizer = svm.get_kernel().get_normalizer()
        
        # cast using dedicated SWIG-helper function
        normalizer = KernelNormalizerToMultitaskKernelNormalizer(normalizer)
        
        # set task vector
        normalizer.set_task_vector_rhs([task_num]*len(examples))
        
        # predict
        out = svm.classify(feat).get_labels()
        
        
        return out



def create_normalizer_from_taxonomy(taxonomy):
    """
    creates kernel normalizer with similarities set
    from hop-distance according to taxnomoy
    """


    #TODO fix --> num tasks can be computed from leaves etc...

    # fetch taxonomy
    # taxonomy = param.taxonomy.data

    print "WARNING; HARDCODED DISTANCE MATRIX IN HERE"

    hardcoded_distances = helper.load("/fml/ag-raetsch/home/cwidmer/svn/projects/alt_splice_code/src/task_sim_tis.bz2")

    # set normalizer
    normalizer = MultitaskKernelNormalizer(data.task_vector_nums)
    
    
    # compute distances
    distances = numpy.zeros((data.get_num_tasks(), data.get_num_tasks()))
    similarities = numpy.zeros((data.get_num_tasks(), data.get_num_tasks()))
    
    for (i,task_name_lhs) in enumerate(data.get_task_names()):
        for (j, task_name_rhs) in enumerate(data.get_task_names()):
            distances[i,j] = task_similarities.compute_hop_distance(taxonomy, task_name_lhs, task_name_rhs)

            
    # normalize distances
    distances = distances / numpy.max(distances)
    
    
    # set similarity
    for (i, task_name_lhs) in enumerate(data.get_task_names()):
        for (j, task_name_rhs) in enumerate(data.get_task_names()):
            
            similarity = param.base_similarity - distances[i,j]
            normalizer.set_task_similarity(i, j, similarity)            
            # save for later
            similarities[i,j] = similarity


    return normalizer





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
    
