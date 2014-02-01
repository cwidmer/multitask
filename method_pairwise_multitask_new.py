#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009-2010 Christian Widmer, Jose Leiva
# Copyright (C) 2009-2010 Max-Planck-Society

"""
Created on 23.03.2009
@author: Christian Widmer, Jose Leiva
@summary: Pairwise SVM-based multitask method 

"""


import numpy

import shogun_factory_new as shogun_factory
from natural_sort import natsorted
from shogun.Classifier import LibSVM
from base_method import MultiMethod


debug = False



class Method(MultiMethod):
    """
    Pairwise Multitask Method based on the SVM
    """
    

    def _train(self, instance_sets, param):

         
        B = 1.0

    
        # keep track of classifiers (one for each task)
        task_names = natsorted(instance_sets.keys())
        svms = dict.fromkeys(task_names)
        

        # while not converged:
        for i in xrange(4):

            print "DEBUG: iteration", i

            for j in task_names:


                # extract examples 
                examples = [inst.example for inst in instance_sets[j]]
                labels = [inst.label for inst in instance_sets[j]]
                tmp_lab = numpy.double(labels)
                feat = shogun_factory.create_features(examples, param)

                # create SVM
                svm = shogun_factory.create_initialized_svm(param, examples, labels)
                

                # compute linear term from other SVMs (not for first iteration)
                if i > 0:

                    # print "computing linear term"
        
                    # compute linear term
                    p = numpy.zeros(len(examples))

                    # get svms from other tasks
                    old_svms = [svms[idx] for idx in task_names if idx != j]
        
                    for (k, old_svm) in enumerate(old_svms):
                        
                        # compute cross-kernel                
                        kv = old_svm.get_kernel()
                        left = old_svm.get_kernel().get_lhs()                    
                        kv.init(left, feat)
        
                        for idx in xrange(len(examples)):
        
                            tmp = 0
        
                            for l in xrange(old_svm.get_num_support_vectors()):
        
                                sv_id = int(old_svm.get_support_vectors()[l])
                                alpha = old_svm.get_alpha(l)
        
                                tmp = tmp + alpha * kv.kernel(sv_id, idx)
        
                            # add to linear term
                            #TODO set gamma from taxonomy
                            gamma = 1.0
                            p[idx] = p[idx] + (-B * gamma * (tmp_lab[idx] * tmp) - 1.0)
                    
      
                    # train regularized SVM
                    svm.set_linear_term(p)

                # train svm
                svm.train()

                # debugging output
                obj_primal = svm.compute_svm_primal_objective()
                obj_dual = svm.compute_svm_dual_objective()

                print "DEBUG:", j, "obj_primal:", obj_primal, "obj_dual:", obj_dual, "num_sv:", svm.get_num_support_vectors()
                
                # save predictor
                svms[j] = svm
               

        # wrap up data needed for predictor (identical for all tasks)
        prediction_data = (param, svms)

        return dict.fromkeys(task_names, prediction_data)
 

    def _predict(self, prediction_data, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: array
        @param examples: list of examples
        @type examples: list 
        """

        # un-wrap prediction data
        (param, svms) = prediction_data

        feat = shogun_factory.create_features(examples, param)

        total_out = numpy.zeros(len(examples))
       
        for (i, predictor) in svms.items():

            if param.flags.has_key("debug") and param.flags["debug"] == True:
                print "using predictor #" + str(i)

            #TODO set gamma from taxonomy
            gamma = 1.0

            #init kernel with evaluation data
            left = predictor.get_kernel().get_lhs()
            predictor.get_kernel().init(left, feat)
    
            #predict
            svm_out = predictor.classify().get_labels()
    
            total_out += gamma * svm_out


        return total_out



def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = 1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(384)

    
    # flags
    flags = {}
    flags["normalize_cost"] = False
    flags["kernel_cache"] = 1000
    flags["use_bias"] = False
    #flags["debug"] = False

    #create mock param object by freezable struct
    param = Options()
    param.kernel = "PolyKernel"
    param.cost = 100.0
    param.id = 1
    param.flags = flags
    
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
    
    
if __name__ == "__main__":
    main()

