#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009-2011 Christian Widmer
# Copyright (C) 2009-2011 Max-Planck-Society

"""
Created on 02.06.2009
@author: Christian Widmer
@summary: Implementation of the augmented SVM multitask method

This methods uses a modified kernel such that tasks, 
which are close to each other are more similar by default.

This implementation uses openopt as solver.
"""


import numpy

import unittest
import shogun_factory_new as shogun_factory
from base_method import MultiMethod
from openopt import QP
from helper import Options

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
 
        
                
        # fix dimensions
        M = len(train_data)

        N = 0
        for key in train_data.keys():
            N += len(train_data[key])
        
        # init containers
        examples = []
        labels = []


        # vector to indicate to which task each example belongs
        task_vector = []
        task_num = 0
        tmp_examples = 0

        label_matrix = numpy.zeros((M,N))


        # extract training data
        for (task_id, instance_set) in train_data.items():
  
            print "train task id:", task_id
            #assert(instance_set[0].dataset.organism==task_id)
            
            examples.extend([inst.example for inst in instance_set])
            
            tmp_labels = [inst.label for inst in instance_set]
            labels.extend(tmp_labels)
            
            begin_idx = tmp_examples
            end_idx = tmp_examples + len(tmp_labels) 
            
            # fill matrix row
            label_matrix[task_num, begin_idx:end_idx] = tmp_labels

            task_vector.extend([task_num]*len(instance_set))

            task_num += 1
            tmp_examples += len(tmp_labels)


        # fetch gammas from parameter object
        # TODO: compute gammas outside of this
        gammas = numpy.ones((M,M)) + numpy.eye(M)
        #gammas = numpy.eye(M)
        

        # create kernel
        kernel = shogun_factory.create_kernel(examples, param)


        y = numpy.array(labels)

        print "computing kernel matrix"

        km = kernel.get_kernel_matrix()
        km = reweight_kernel_matrix(km, gammas, task_vector)

        # "add" labels to Q-matrix
        km = numpy.transpose(y.flatten() * (km*y.flatten()).transpose())

        print "done computing kernel matrix, calling solver"


        f = -numpy.ones(N)
        b = numpy.zeros((M,1))

        # set up QP
        p = QP(km, f, Aeq=label_matrix, beq=b, lb=numpy.zeros(N), ub=param.cost*numpy.ones(N))
        p.debug=1
        
        # run solver
        r = p.solve('cvxopt_qp', iprint = 0)

        print "done with training"

        alphas = r.xf
        objective = r.ff


        print "alphas:", alphas

        predictors = {}

        for (k, task_id) in enumerate(train_data.keys()):
            # pack all relevant information in predictor
            predictors[task_id] = (alphas, param, task_vector, k, gammas, examples, labels)

        return predictors


    
    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: array
        @param examples: list of examples
        @type examples: list
        @param task_id: task identifier
        @type task_id: str
        """


        (alphas, param, task_vector_lhs, task_num, gammas, train_examples, train_labels) = predictor


        print "length alphas:", len(alphas)
        
        # shogun data
        feat_train = shogun_factory.create_features(train_examples, param)
        feat_test = shogun_factory.create_features(examples, param)
        
        # create kernel
        kernel = shogun_factory.create_empty_kernel(param)
        kernel.init(feat_train, feat_test)
        

        # all examples belong to same task (called individually per task)
        task_vector_rhs = [task_num]*len(examples)

        # re-weight kernel matrix
        km = kernel.get_kernel_matrix()
        km = reweight_kernel_matrix(km, gammas, task_vector_lhs, task_vector_rhs)

        # compute output
        out = numpy.zeros(len(examples))

        for test_idx in xrange(len(examples)):
            for train_idx in xrange(len(train_examples)):
                
                out[test_idx] += alphas[train_idx] * train_labels[train_idx] * km[train_idx, test_idx]
                

        return out



def reweight_kernel_matrix(km, gammas, task_vector_lhs, task_vector_rhs=None):
    """
    method that computes explicit reweighting of kernel matrix
    """

    if task_vector_rhs==None:
        task_vector_rhs = task_vector_lhs

    # basic sanity checks
    assert(km.shape[0]==len(task_vector_lhs))
    assert(km.shape[1]==len(task_vector_rhs))
    assert(len(set(task_vector_lhs))==len(gammas))

    N_lhs = len(task_vector_lhs)
    N_rhs = len(task_vector_rhs)

    # weight km entries according to gammas
    for i in xrange(N_lhs):

        task_i = task_vector_lhs[i]

        for j in xrange(N_rhs):
            task_j = task_vector_rhs[j]
            weight = gammas[task_i][task_j]
            
            km[i][j] = km[i][j] * weight

    return km



def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = 1

    from expenv import MultiSplitSet
     
        
    # select dataset
    multi_split_set = MultiSplitSet.get(384)
    
    # flags
    flags = {}
    flags["normalize_cost"] = False
    #flags["epsilon"] = 0.005
    flags["kernel_cache"] = 200
    flags["use_bias"] = False 

    # arts params
    #flags["svm_type"] = "liblineardual"

    flags["degree"] = 24

    flags["local"] = False
    flags["mem"] = "6G"
    flags["maxNumThreads"] = 1
    
    
    #create mock param object by freezable struct
    param = Options()
    #param.kernel = "GaussianKernel"
    param.kernel = "PolyKernel"
    param.sigma = 3.0
    param.cost = 10.0
    param.transform = 1.0
    param.id = 666
    param.flags = flags
    param.taxonomy = multi_split_set.taxonomy.data
    
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

    
class TestAugmentedTraining(unittest.TestCase):

    def setUp(self):

        import expenv

        run = expenv.Run.get(13490)
        self.instances = run.get_train_data()
        self.test_data = run.get_eval_data()
        
        self.param = run.method.param
        flags = {}
        flags["kernel_cache"] = 200 

        #create mock param object by freezable struct
        param = Options()
        param.kernel = "GaussianKernel"
        param.sigma = 3.0
        param.cost = 10.0
        param.flags = flags
        
        self.param = param
        
        
    def testtrainsimple(self):

        method_internal = Method(self.param)
        preds_internal = method_internal.train(self.instances)        
        assessment = method_internal.evaluate(self.test_data)
        assessment.clean_up()


if __name__ == '__main__':
    #unittest.main()
    main()

