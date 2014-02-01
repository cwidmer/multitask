#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009-2010 Christian Widmer
# Copyright (C) 2009-2010 Max-Planck-Society

"""
Created on 29.03.2009
@author: Christian Widmer
@summary: Implementation of the union SVM multitask method

Essentially, a single SVM is trained on the union of datasets. 
  
"""


import shogun_factory_new as shogun_factory
from base_method import MultiMethod, PreparedMultitaskData



class Method(MultiMethod):
    """
    defines the Union SVM method
    """
    

    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training, mapped by task_id
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        
        @return: trained predictors, mapped by task_id
        @rtype: dict<str, SVM>
        """
                
        # init container       
        svms = {}

        # concatenate data
        data = PreparedMultitaskData(train_data, shuffle=False)
        
        # create svm
        svm = shogun_factory.create_initialized_svm(param, data.examples, data.labels)
            
        print "starting training procedure"    
        
        # train SVM
        svm.train()
        
        print "training done"
        
        # use a reference to the same svm several times
        for task_id in train_data.keys():
            svms[task_id] = svm
        
        return svms
    
    

    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: SVMLight
        @param examples: list of examples
        @type examples: list<str> 
        @param task_id: task id (e.g. organism name)
        @type task_id: str
        
        @return: prediction output for each data point
        @rtype: list<float>
        """

        svm = predictor

        # shogun data
        feat = shogun_factory.create_features(examples, self.param)

        # predict
        svm_out = svm.classify(feat).get_labels()


        print "trying to delete feature object"
        del feat

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
