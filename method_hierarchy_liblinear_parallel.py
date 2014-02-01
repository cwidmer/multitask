#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010 Christian Widmer
# Copyright (C) 2010 Max-Planck-Society

"""
Created on 11.11.2011
@author: Christian Widmer
@summary: Hierarchical SVM-based multitask method using liblinear

"""

from collections import defaultdict
import shogun_factory_new as shogun_factory
import pythongrid as pg
from base_method import MultiMethod

import method_hierarchy_liblinear_parallel


class Method(MultiMethod):
    """
    Hierarchical Multitask Method based on the SVM
    """

    

    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        """

        data = defaultdict(dict)

        for task_id in train_data.keys():
            print "task_id:", task_id
            
            data[task_id]["XT"] = [inst.example for inst in train_data[task_id]]
            data[task_id]["LT"] = [inst.label for inst in train_data[task_id]]
            
        
        root = param.taxonomy


        #####################################################
        #     top-down processing of taxonomy
        #####################################################

        results_dict = process_node((root, data, param))
         
        
        #####################################################
        #    Wrap things up    
        #####################################################
 
        # wrap up predictors for later use
        predictors = {}

        for leaf in root.get_leaves():
            
            predictors[leaf.name] = (results_dict[leaf.name], param)
            
            assert(results_dict[leaf.name] != None)
 
 
        # make sure we have the same keys (potentiall in a different order)  
        sym_diff_keys = set(train_data.keys()).symmetric_difference(set(predictors.keys()))
        assert len(sym_diff_keys)==0, "symmetric difference between keys non-empty: " + str(sym_diff_keys)  


        return predictors
   

    
    
    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: array
        @param examples: list of examples
        @type examples: list
        @param task_id: task identifier
        @type task_id: int
        
        @return: svm output
        @rtype: list<float>
        """


        #####################################################
        #    classification
        #####################################################


        (svm, param) = predictor        


        #shogun data
        feat = shogun_factory.create_features(examples, param)

        out = svm.classify(feat).get_labels()        

        return out



def process_children(node, train_data, param):
    """
    takes current node and distributes tasks
    using pythongrid. Results are merged afterwards
    """
    
    assert(node.children != None)

    
    if len(node.children) > 1:
        # do map-reduce
        
        # map
        input_args = [(child_node, train_data, param) for child_node in node.children]
        result_dicts = pg.map(method_hierarchy_liblinear_parallel.process_node, input_args, local=param.flags["local"], mem=param.flags["mem"], maxNumThreads=param.flags["maxNumThreads"])

        assert(len(result_dicts) == len(node.children))

        # merge dictionaries (reduce)
        ret = result_dicts[0]
        
        for tmp_dict in result_dicts[1:]:
            ret.update(tmp_dict)

    else:
        # local processing
        ret = process_node(node.children[0], train_data, param)

            
    return ret



def process_node(args_tuple):
    """
    learns classifier for current node
    """

    # expand arguments
    (node, train_data, param) = args_tuple

    print "data at current level", node.get_data_keys()
  
      
    # initialize containers
    examples = []
    labels = []       
      
    # concatenate data at level
    for key in node.get_data_keys():
    
        instance_set = train_data[key]
        examples.extend(instance_set["XT"])
        labels.extend(instance_set["LT"])
    
    
    #####################################################
    #    train predictors    
    #####################################################
    
    
    # set up presvm
    if node.is_root():
    
        # no parent at root node
        parent_svm = None 
        weight = 0
    
    else:
        
        # regularize against parent predictor
        parent_svm = node.parent.predictor
        weight = param.transform
    
    
    print "current edge_weight:", weight, " ,name:", node.name
    
    # create SVM object
    svm = shogun_factory.create_initialized_domain_adaptation_svm(param, examples, labels, parent_svm, weight)
    
    svm.set_train_factor(param.flags["train_factor"])
    # invoke training procedure
    svm.train()

    # attach current predictor for ancestor nodes
    node.predictor = svm

    #TODO copy only relevant keys

    # recurse
    if node.is_leaf():
        results_dict = {}
    else: 
        results_dict = process_children(node, train_data, param)
    
    # update dict with current predictor    
    results_dict[node.name] = svm
    
    
    return results_dict
    

def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = 1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(432)

    
    # flags
    flags = {}
    flags["normalize_cost"] = False
    #flags["epsilon"] = 0.005
    flags["kernel_cache"] = 200
    flags["use_bias"] = False 

    # arts params
    flags["svm_type"] = "liblineardual"

    flags["degree"] = 24
    flags["degree_spectrum"] = 4
    flags["shifts"] = 0 #32
    flags["center_offset"] = 70
    flags["train_factor"] = 1

    flags["local"] = False
    flags["mem"] = "6G"
    flags["maxNumThreads"] = 1
    
    #create mock param object by freezable struct
    param = Options()
    param.kernel = "Promoter"
    param.cost = 1.0
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
    
    
if __name__ == "__main__":
    main()

