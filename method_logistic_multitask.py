#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 13.03.2009
@author: Christian Widmer
@summary: Multitask method using logistic regression

"""



import copy

import scipy.io
import scipy.optimize
import numpy

from base_method import MultiAssessment
import helper

debug = False

def fun(x, example_sets, label_sets, C, par):
    """
    adapter to multitask_obj that computes objective value
    and gradient in one run
    """

    (F, gradient) = multitask_obj(x, example_sets, label_sets, C, par)

    return F


def fun_prime(x, example_sets, label_sets, C, par):
    """
    adapter to multitask_obj_obj that computes objective value
    and gradient in one run
    """

    (F, gradient) = multitask_obj(x, example_sets, label_sets, C, par)

    return gradient



def multitask_obj(x, example_sets, label_sets, C, par):
    """
    compute objective function
    """

    #fetch dimensions
    M = len(example_sets)
    d = len(example_sets[0][0])
    #d = example_sets[0].shape[1]

    if debug: print "M:", M, "d:", d
    
    #reshape
    W = numpy.transpose(numpy.reshape(x, (M, d)))
    if debug: print "W:", W.shape
    

    #init obj and gradient
    F = 0
    grad = numpy.zeros((d,M))
    if debug: print "grad:", grad.shape

    #compute regularizer
    for i in xrange(M):

        F = F + numpy.dot(W[:,i],W[:,i])/2

        grad[:,i] = W[:,i]

        for j in xrange(M):

            grad[:,i] = grad[:,i] + C[i,j]*(W[:,i]-W[:,j])
            grad[:,j] = grad[:,j] + C[i,j]*(W[:,j]-W[:,i])

            difference_vec = W[:,i]-W[:,j]

            #square
            F = F + C[i,j] * numpy.dot(difference_vec,difference_vec)/2 ;

        #normalize by gammas
        #TODO make this optional
        #F = F / numpy.sum(C[i,:])



    #loss term
    for i in xrange(M):

        if debug: print "------"
        if debug: print "loss loop", i
        tmp_w = W[:,i]

        if debug: print "es:", example_sets[i].shape
        if debug: print "tmp_w:", tmp_w.shape

        output = numpy.dot(example_sets[i], tmp_w)
        
        #print "output:", output.shape

        labels = label_sets[i]
        #print "labels:", labels.shape

        #pointwise multiplication
        mg = output * labels
        #print "mg:", mg.shape
        

        #add loss to objective
        F = F + sum((1/par.sharpness)*numpy.log(1+numpy.exp(par.sharpness*(-mg+par.shift))))
        #F = F + sum(numpy.log(1+numpy.exp(-mg)))

        #add loss to gradient
        dmg = (1/(1+numpy.exp(par.sharpness*(-mg+par.shift))) * numpy.exp(par.sharpness*(-mg+par.shift)))
        #print "dmg:", dmg.shape
        
        dmg2 = numpy.multiply(dmg, labels)
        #print "dmg2:", dmg2.shape
        
        tmp_grad = numpy.dot(example_sets[i].transpose(), dmg2)
        #print "tmp_grad:", tmp_grad.shape
        
        grad[:,i] = grad[:,i] - tmp_grad
        #print "grad:", grad.shape
        
        
    #construct final gradient        
    grad = numpy.reshape(grad, d*M, 1)
    
    #print "final grad:", grad.shape
    
    #print "======================="
    return (F, grad)




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
        
        #TODO compare to matlab implementation
        #print fun(x0, example_sets, label_sets, Cs, param)
        
        
        #call optimization procedure      
        print "starting optimization"
        xopt = scipy.optimize.fmin_ncg(fun, x0, fun_prime, args=[example_sets, label_sets, Cs, param])
    
    
        #TODO optimize convergence parameters
        #scipy.optimize.fmin_ncg(fun, x0, fun_prime)
        #scipy.optimize.fmin_bfgs(fun, x0, fun_prime)
        #ret = scipy.optimize.fmin_ncg(fun, x0, fun_prime, full_output=1, avextol = 1.0e-2)
        #ret = scipy.optimize.fmin_ncg(fun, x0, fun_prime, full_output=1, avextol = 0.01)
        
        #unpack tuple
        #(xopt, fopt, fcalls, gcalls, hcalls, warnflag) = ret
        
        #return numpy.transpose(numpy.reshape(xopt, (d, M)))    
        #return numpy.reshape(xopt, (M,d), order="FORTRAN")
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


class tmpPar(object):
    """
    
    """
    
    sharpness = 1.0
    shift = 1.0
    cost = 1.0
    
    def __init__(self, sharpness=1.0, shift=1.0, cost=1.0):
        """
        
        """
        
        self.sharpness = sharpness
        self.shift = shift
        self.cost = cost
    
    
