#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 20.03.2009
@author: Christian Widmer
@summary: Plain multitask method, that treats all tasks equally related 

"""



import copy

import scipy.io
import scipy.optimize
import numpy

from method_multisource import MultiMethod, MultiAssessment
import helper

debug = False





class PlainMultitask(MultiMethod):
    """
    Plain Multitask Method based on the logistic loss
    with all regularization constants C_ij set to 1
    """
    

    def _train(self, instance_sets, param):
    
   
        #fix parameters
        M = len(instance_sets)
        Cs = numpy.ones((M,M))    
   
        
        #init containers
        example_sets = []
        label_sets = []
        

        #perform data conversion -> strToVec
        for instance_set in instance_sets:
            
            example_set = helper.gen_features([inst.example for inst in instance_set])
            print "example set:", example_set.shape
            label_set = numpy.array([inst.label for inst in instance_set])
            print "label set:", label_set.shape
            print "num positive labels", sum(label_set)+len(label_set)
            
            example_sets.append(example_set)
            label_sets.append(label_set)


        #determine starting point
        M = len(example_sets)
        d = len(example_sets[0][0])
        
        print "M:", M, "d:", d
        
        dim_x0 = d*M
        numpy.random.seed(123967)
        x0 = numpy.random.randn(dim_x0)
        

        """
        Example:
        Consider the problem
        0.5 * (x1^2 + 2x2^2 + 3x3^2) + 15x1 + 8x2 + 80x3 -> min        (1)
        subjected to
        x1 + 2x2 + 3x3 <= 150            (2)
        8x1 +  15x2 +  80x3 <= 800    (3)
        x2 - x3 = 25                              (4)
        x1 <= 15                                  (5)
        """

        #TODO map variables
        
        from numpy import diag, matrix, inf
        from openopt import QP
        p = QP(diag([1,2,3]), [15,8,80], A = matrix('1 2 3; 8 15 80'), b = [150, 800], Aeq = [0, 1, -1], beq = 25, ub = [15,inf,inf])
        # or p = QP(H=diag([1,2,3]), f=[15,8,80], A = matrix('1 2 3; 8 15 80'), b = [150, 800], Aeq = [0, 1, -1], beq = 25, ub = [15,inf,inf])
        r = p.solve('cvxopt_qp', iprint = 0)
        #r = p.solve('nlp:ralg', xtol=1e-7, alp=3.9, plot=1)#, r = p.solve('nlp:algencan')
        #f_opt, x_opt = r.ff, r.xf
        f_opt = r.xf
        # x_opt = array([-14.99999995,  -2.59999996, -27.59999991])
        # f_opt = -1191.90000013



        return numpy.reshape(xopt, (M,d))
    
    

    def _predict(self, predictor, examples):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: array
        @param examples: list of examples
        @type examples: list 
        """

        #takes care of conversion to string features
        examples_vec = helper.gen_features(examples)
        
        out = numpy.dot(examples_vec, predictor)
        
        return out

