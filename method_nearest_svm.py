#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 10.12.2009
@author: Christian Widmer
@summary: Hierarchical SVM-based multitask method 

"""


from shogun.Classifier import SVMLight, DomainAdaptationSVM
import shogun_factory_new

from base_method import MultiMethod

debug = False



class Method(MultiMethod):
    """
    Hierarchical Multitask Method based on the SVM
    """
    

    def get_data(self, node, train_data):
 
        #####################################################
        #     init data structures
        #####################################################

        # get data below current node
        data = [train_data[key] for key in node.get_data_keys()]
        

        print "data at current level"
        for instance_set in data:        
            print instance_set[0].dataset
        
        
        # initialize containers
        examples = []
        labels = []       


        # concatenate data
        for instance_set in data:
  
            print "train split_set:", instance_set[0].dataset.organism
            
            for inst in instance_set:
                examples.append(inst.example)
                labels.append(inst.label)

        return (examples, labels)


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
        

        # top-down processing of taxonomy
        
        for node in root.get_leaves():


            #####################################################
            #    train predictor 
            #####################################################
 
            parent_node = node.get_nearest_neighbor()

            cost = param.cost
   
            (examples, labels) = self.get_data(parent_node, train_data)

            # create shogun data objects
            k_parent = shogun_factory_new.create_kernel(examples, param)
            lab_parent = shogun_factory_new.create_labels(labels)

            parent_svm = SVMLight(cost, k_parent, lab_parent)
            
            parent_svm.train()
    


            #####################################################
            #    train predictors    
            #####################################################
            

            (examples, labels) = self.get_data(node, train_data)

            # create shogun data objects
            k = shogun_factory_new.create_kernel(examples, param)
            lab = shogun_factory_new.create_labels(labels)

               
            # regularize vs parent predictor
            
            weight = param.transform
            print "current edge_weight:", weight, " ,name:", node.name
            
            svm = DomainAdaptationSVM(cost, k, lab, parent_svm, weight)
            svm.train()

                                    
            # attach svm to node
            node.predictor = svm
 


        #####################################################
        #    Wrap things up    
        #####################################################
 
        # wrap up predictors for later use
        predictors = {}

        for leaf in root.get_leaves():
            
            predictors[leaf.name] = leaf.predictor
            
            assert(leaf.predictor!=None)
            
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

        #shogun data
        feat = shogun_factory_new.create_features(examples)

        out = predictor.classify(feat).get_labels()        

        return out

