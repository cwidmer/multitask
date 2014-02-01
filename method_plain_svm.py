#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009-2011 Christian Widmer
# Copyright (C) 2009-2011 Max-Planck-Society

"""
Created on 13.03.2009
@author: Christian Widmer
@summary: Implementation of the plain SVM multitask method 

"""


import shogun_factory_new as shogun_factory


from base_method import MultiMethod


class Method(MultiMethod):
    """
    defines the Plain SVM method
    """
 

    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        
        @return: trained SVMs
        @rtype: dict<task_id: svm>
        """

        
        # initialize containers
        svms = {}


        # compute predictors for each task
        for (task_id, instance_set) in train_data.items():
            
            print "processing task %s" %  (task_id)
            
            # compile training set
            example_set = [inst.example for inst in instance_set]
            label_set = [inst.label for inst in instance_set]

            print "unique tasks:", set([len(s) for s in example_set])

            # create svm
            svm = shogun_factory.create_initialized_svm(param, example_set, label_set)
            
            # train SVM
            svm.train()
       
            # store svm for current task
            svms[task_id] = svm
        
        
        return svms
    
    

    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: SVM object
        @param examples: list of examples
        @type examples: list 
        @param task_id: task id (e.g. organism name)
        @type task_id: str
        """
        
        
        #shogun data
        feat = shogun_factory.create_features(examples, self.param)
        
        #predict
        svm_out = predictor.classify(feat).get_labels()

        return svm_out



def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = 1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(417)

    
    # flags
    flags = {}
    flags["normalize_cost"] = False
    #flags["epsilon"] = 0.01
    flags["kernel_cache"] = 200
    flags["use_bias"] = False

    # arts params
    flags["svm_type"] = "liblineardual"

    flags["degree"] = 24
    flags["degree_spectrum"] = 4
    flags["shifts"] = 0 #32
    flags["center_offset"] = 70
 

    #create mock param object by freezable struct
    param = Options()
    param.kernel = "Promoter"
    param.cost = 1.0
    param.id = 666
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
