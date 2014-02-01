#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009-2010 Christian Widmer
# Copyright (C) 2009-2010 Max-Planck-Society

"""
Created on 23.03.2009
@author: Christian Widmer
@summary: Hierarchical SVM-based multitask method using liblinear

"""

import shogun_factory_new as shogun_factory
from base_method import MultiMethod


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



        for task_id in train_data.keys():
            print "task_id:", task_id
            
        
        root = param.taxonomy.data

        grey_nodes = [root]
        

        #####################################################
        #     top-down processing of taxonomy
        #####################################################
 

        while len(grey_nodes)>0:
           
            node = grey_nodes.pop(0) # pop first item
            
            # enqueue children
            if node.children != None:
                grey_nodes.extend(node.children)
    

            # get data below current node
            data = [train_data[key] for key in node.get_data_keys()]
            
    
            #print "data at current level"
            #for instance_set in data:        
            #    print instance_set[0].dataset
            
            
            # initialize containers
            examples = []
            labels = []       
    

            # concatenate data at level
            for instance_set in data:
      
                #print "train split_set:", instance_set[0].dataset.organism
                
                for inst in instance_set:
                    examples.append(inst.example)
                    labels.append(inst.label)
    

            print "done concatenating data"

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

            #print "============================================="
            #print "WARNING: MAX TRAIN TIME SET TO 1!!!!!!!!!!!!!!!!"
            #print "============================================="

            #svm.set_max_train_time(0)
            #svm.set_max_iterations(1)
 
            print "invoking shogun training procedure"
            
            # invoke training procedure
            svm.train()
            
            # attach svm to current node
            node.predictor = svm
            
            # save some information
            # TODO refactor
            #self.additional_information[node.name + " svm obj"] = svm.get_objective()
            #self.additional_information[node.name + " svm num sv"] = svm.get_num_support_vectors()
            #self.additional_information[node.name + " runtime"] = svm.get_runtime()
            


        #####################################################
        #    Wrap things up    
        #####################################################
 
        # wrap up predictors for later use
        predictors = {}

        for leaf in root.get_leaves():
            
            predictors[leaf.name] = leaf.predictor
            
            assert(leaf.predictor!=None)
 

        # make sure we have the same keys (potentially in a different order)  
        sym_diff_keys = set(train_data.keys()).symmetric_difference(set(predictors.keys()))
        assert len(sym_diff_keys)==0, "symmetric difference between keys non-empty: " + str(sym_diff_keys)  


        # clear tree to avoid having __del__ attributes in cyclic-reference graph
        # otherwise, this will not be cleaned up by garbage collector
        root.clear_predictors()


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


        svm = predictor        

        #shogun data
        feat = shogun_factory.create_features(examples, self.param)

        out = svm.classify(feat).get_labels()
        
        
        # flush feats (not very elegant, but avoids having huge test sets floating around in mem)
        one_feat = shogun_factory.create_features([examples[0]], self.param)
        svm.classify(one_feat)
        

        return out






def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = 1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(434)

    
    # flags
    flags = {}
    flags["normalize_cost"] = False
    flags["epsilon"] = 1.0 
    #0.005
    flags["kernel_cache"] = 200
    flags["use_bias"] = False 

    # arts params
    flags["svm_type"] = "liblineardual"

    flags["degree"] = 24
    flags["degree_spectrum"] = 4
    flags["shifts"] = 0 #32
    flags["center_offset"] = 70
    flags["train_factor"] = 1

    #create mock param object by freezable struct
    param = Options()
    param.kernel = "Promoter"
    param.cost = 1.0
    param.transform = 1.0
    param.id = 666
    param.flags = flags
    param.taxonomy = multi_split_set.taxonomy
    
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

