#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 02.06.2009
@author: Christian Widmer
@summary: Implementation of the augmented SVM multitask method5
This methods uses a modified kernel such that tasks, 
which are close to each other are more similar by default.
"""

import unittest
import numpy
import helper
from shogun.Shogun import LibSVM, SVMLight, StringCharFeatures, Labels 
from shogun.Shogun import DNA, WeightedDegreeStringKernel
from base_method import MultiMethod
from expenv_runner import fetch_gammas
from openopt import QP

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
 
        
                
        # fix parameters
        M = len(train_data)

        
        # init containers
        examples = []
        labels = []


        # vector to indicate to which task each example belongs
        task_vector = []
        task_num = 0

        # extract training data
        for (task_id, instance_set) in train_data.items():
  
            print "train task id:", task_id
            assert(instance_set[0].dataset.organism==task_id)
            
            examples.extend([inst.example for inst in instance_set])
            labels.extend([inst.label for inst in instance_set])

            task_vector.extend([task_num]*len(instance_set))

            task_num += 1


        # shogun data
        feat = StringCharFeatures(DNA)
        feat.set_string_features(examples)
        lab = Labels(numpy.double(labels))


        # fetch gammas from parameter object
        gammas = param.taxonomy.data
        from expenv_runner import TaskMap
        tm = TaskMap(param.taxonomy.data)
        # fetch gammas from parameter object
        gammas = tm.taskmap2matrix(train_data.keys())
        print gammas

        assert(gammas.shape[0] == len(train_data))

        # create kernel
        k = WeightedDegreeStringKernel(feat, feat, param.wdk_degree, 0)


        y = numpy.array(labels)

        km = k.get_kernel_matrix()
        km = reweight_kernel_matrix(km, gammas, task_vector)

        km = numpy.transpose(y.flatten() * (km*y.flatten()).transpose())

        N = len(labels)
        f = -numpy.ones(N)

        # set up QP
        p = QP(km, f, Aeq=y, beq=0, lb=numpy.zeros(N), ub=param.cost*numpy.ones(N))

        # run solver
        r = p.solve('cvxopt_qp', iprint = 0)


        alphas = r.xf
        objective = r.ff

        predictors = {}


        for (k, task_id) in enumerate(train_data.keys()):
            # pack all relevant information in predictor
            predictors[task_id] = (alphas, param.wdk_degree, task_vector, k, gammas, examples, labels)

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


        (alphas, wdk_degree, task_vector_lhs, task_num, gammas, train_examples, train_labels) = predictor


        print "length alphas:", len(alphas), ", length train_examples:", len(train_examples), ", length train_labels:", len(train_labels)

        # shogun data
        feat_train = StringCharFeatures(DNA)
        feat_train.set_string_features(list(train_examples))

        feat_test = StringCharFeatures(DNA)
        feat_test.set_string_features(list(examples))

        k = WeightedDegreeStringKernel(feat_train, feat_test, wdk_degree, 0)

        task_vector_rhs = [task_num]*len(examples)

        km = k.get_kernel_matrix()
        km = reweight_kernel_matrix(km, gammas, task_vector_lhs, task_vector_rhs)
       
        alphas = numpy.array(alphas)

        out = numpy.dot(alphas, km)
        ####################

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


class TestAugmentedTraining(unittest.TestCase):

    def setUp(self):

        import expenv

        run = expenv.Run.get(13490)
        self.instances = run.get_train_data()
        self.param = run.method.param
        
    def testtrainsimple(self):

        method_internal = Method(self.param)
        preds_internal = method_internal.train(self.instances)


if __name__ == '__main__':
    unittest.main()

