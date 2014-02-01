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
@summary: Implementation of the plain svm method
using openopt as solver backend
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

        predictors = {}

        # extract training data
        for (task_id, instance_set) in train_data.items():
  
            print "train task id:", task_id
            assert(instance_set[0].dataset.organism==task_id)
            
            examples = [inst.example for inst in instance_set]
            labels = [inst.label for inst in instance_set]

        
            # shogun data
            feat = StringCharFeatures(DNA)
            feat.set_string_features(examples)
            lab = Labels(numpy.double(labels))

            # create kernel
            k = WeightedDegreeStringKernel(feat, feat, param.wdk_degree, 0)

            y = numpy.array(labels)

            km = k.get_kernel_matrix()
            km = numpy.transpose(y.flatten() * (km*y.flatten()).transpose())

            N = len(labels)
            f = -numpy.ones(N)

            # set up QP
            p = QP(km, f, Aeq=y, beq=0, lb=numpy.zeros(N), ub=param.cost*numpy.ones(N))

            # run solver
            r = p.solve('cvxopt_qp', iprint = 0)

            alphas = r.xf
            objective = r.ff

            print "objective:", objective


            predictors[task_id] = (alphas, param.wdk_degree, examples, labels)

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


        (alphas, wdk_degree, train_examples, train_labels) = predictor


        print "length alphas:", len(alphas), ", length train_examples:", len(train_examples), ", length train_labels:", len(train_labels)

        # shogun data
        feat_train = StringCharFeatures(DNA)
        feat_train.set_string_features(list(train_examples))

        feat_test = StringCharFeatures(DNA)
        feat_test.set_string_features(list(examples))

        k = WeightedDegreeStringKernel(feat_train, feat_test, wdk_degree, 0)

        km = k.get_kernel_matrix()
       
        alphas = numpy.array(alphas)


        print "warning: labels missing" #TODO FIX
        out = numpy.dot(alphas, km)
        ####################

        return out


