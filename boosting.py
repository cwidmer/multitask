#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Christian Widmer
# Copyright (C) 2011 Max-Planck-Society

"""
Created on 11.03.2011
@author: Christian Widmer
@summary: Implementation of boosting for MTL using CVXMOD
"""


import numpy
import cvxmod
from cvxmod.sets import probsimp
from cvxmod.atoms import square, norm2



def solve_svm(out, labels, nu, solver):
    '''
    solve boosting formulation used by gelher and nowozin
    
    @param out: matrix (N,F) of predictions (for each f_i) for all examples
    @param labels: vector (N,1) label for each example 
    @param nu: regularization constant
    @param solver: which solver to use. options: 'mosek', 'glpk'
    '''
    
    
    # get dimension
    N = out.size[0]
    F = out.size[1]
    
    assert N==len(labels), str(N) + " " + str(len(labels))
    
    norm_fact = 1.0 / (nu * float(N))
    print "normalization factor %f" % (norm_fact)
    
    
    # avoid point-wise product
    label_matrix = cvxmod.zeros((N,N))
    
    for i in xrange(N):
        label_matrix[i,i] = labels[i] 
    
    
    #### parameters
    
    f = cvxmod.param("f", N, F)
    y = cvxmod.param("y", N, N, symm=True)
    norm = cvxmod.param("norm", 1) 
    
    
    #### varibales
    
    # rho
    rho = cvxmod.optvar("rho", 1)
    
    # dim = (N x 1)
    chi = cvxmod.optvar("chi", N)
    
    # dim = (F x 1)
    beta = cvxmod.optvar("beta", F)
    
    
    #objective = -rho + cvxmod.sum(chi) * norm_fact + square(norm2(beta)) 
    objective = -rho + cvxmod.sum(chi) * norm_fact
    
    print objective
    
    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))
    
    
    # create contraints for probability simplex
    #p.constr.append(beta |cvxmod.In| probsimp(F))
    p.constr.append(cvxmod.sum(beta)==1.0)
    p.constr.append(beta >= 0.0)
    p.constr.append(chi >= 0.0)
    
    # attempt to perform non-sparse boosting
    #p.constr.append(square(norm2(beta)) <= 1.0)
    
    
    #    y       f     beta          y    f*beta      y*f*beta
    # (N x N) (N x F) (F x 1) --> (N x N) (N x 1) --> (N x 1)
    p.constr.append(y * (f * beta) + chi >= rho)
    
    
    # set values for parameters
    f.value = out
    y.value = label_matrix
    norm.value = norm_fact 


    print "solving problem"
    print "============================================="
    print p
    print "============================================="
    

    # start solver
    p.solve(lpsolver=solver)
    
    
    # print variables    
    cvxmod.printval(chi)
    cvxmod.printval(beta)
    cvxmod.printval(rho)
    

    return numpy.array(cvxmod.value(beta))



def solve_nu_svm(out, labels, nu, solver, reg):
    '''
    solve boosting formulation used by gelher and nowozin
    
    @param out: matrix (N,F) of predictions (for each f_i) for all examples
    @param labels: vector (N,1) label for each example 
    @param nu: regularization constant
    @param solver: which solver to use. options: 'mosek', 'glpk'
    '''
    
    
    # get dimension
    N = out.size[0]
    F = out.size[1]
    
    assert N==len(labels), str(N) + " " + str(len(labels))
    
    norm_fact = 1.0 / (nu * float(N))
    print "normalization factor %f" % (norm_fact)
    
    
    # avoid point-wise product
    label_matrix = cvxmod.zeros((N,N))
    
    for i in xrange(N):
        label_matrix[i,i] = labels[i] 
    
    
    #### parameters
    
    f = cvxmod.param("f", N, F)
    y = cvxmod.param("y", N, N, symm=True)
    norm = cvxmod.param("norm", 1) 
    
    
    #### varibales
    
    # rho
    rho = cvxmod.optvar("rho", 1)
    
    # dim = (N x 1)
    chi = cvxmod.optvar("chi", N)
    
    # dim = (F x 1)
    beta = cvxmod.optvar("beta", F)
    
    # Q
    Q = cvxmod.eye(F)
    
    # regularize vs ones
    if reg:
        objective =  0.5*cvxmod.atoms.quadform(beta, Q) - (1.0/float(F))*cvxmod.sum(beta) -rho*nu + norm_fact*cvxmod.sum(chi)
        #objective =  0.5*cvxmod.atoms.quadform(beta, Q) - (1.0/float(F))*cvxmod.sum(beta) -rho*nu + norm_fact*cvxmod.sum(chi)
    else:
        objective = 0.5*cvxmod.atoms.quadform(beta, Q) -rho*nu + norm_fact*cvxmod.sum(chi)
    
    
    print objective
    
    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))
    
    
    # create contraints for probability simplex
    #p.constr.append(beta |cvxmod.In| probsimp(F))
    #p.constr.append(cvxmod.sum(beta)==1.0)
    p.constr.append(beta >= 0.0)
    p.constr.append(chi >= 0.0)
    
    # attempt to perform non-sparse boosting
    #p.constr.append(square(norm2(beta)) <= 1.0)
    
    
    #    y       f     beta          y    f*beta      y*f*beta
    # (N x N) (N x F) (F x 1) --> (N x N) (N x 1) --> (N x 1)
    p.constr.append(y * (f * beta) + chi >= rho)
    
    
    # set values for parameters
    f.value = out
    y.value = label_matrix
    norm.value = norm_fact 


    print "solving problem"
    print "============================================="
    print p
    print "============================================="
  

    # start solver
    p.solve(lpsolver=solver)
    
    
    # print variables    
    cvxmod.printval(chi)
    cvxmod.printval(beta)
    cvxmod.printval(rho)
    

    return numpy.array(cvxmod.value(beta))



def main(nu=0.9):
    '''
    solve boosting formulation used by gelher and nowozin
    
    @param out: matrix (N,F) of predictions (for each f_i) for all examples
    @param labels: vector (N,1) label for each example 
    @param nu: regularization constant
    @param solver: which solver to use. options: 'mosek', 'glpk'
    '''    
    
    # one classifier is always wrong
    dat = numpy.array([[1,1], [1, -1], [1, -1]])
    # avoid dimension mess-up
    out = cvxmod.matrix(dat)

    print out.size

    labels = [1,1,-1]

    
    #solve_boosting(out, labels, nu, solver="glpk")
    return solve_nu_svm(out, labels, nu, solver="mosek")
    
    
if __name__ == "__main__":
    main()

