#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 23.03.2009
@author: Christian Widmer
@summary: Hierarchical SVM-based multitask method 

"""


import shogun
from shogun.Classifier import DomainAdaptationSVM
import shogun_factory_new

from base_method import MultiMethod

debug = False



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
        

        # top-down processing of taxonomy
        
        while len(grey_nodes)>0:
           
            node = grey_nodes.pop(0) # pop first item
            
            # enqueue children
            if node.children != None:
                grey_nodes.extend(node.children)
    

    
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
    

            # create shogun data objects
            k = shogun_factory_new.create_kernel(examples, param)
            lab = shogun_factory_new.create_labels(labels)

            
            cost = param.cost
            #cost = node.cost
            
            print "using cost:", cost




            #####################################################
            #    train predictors    
            #####################################################
            
                
            # init predictor variable
            svm = None
            

            # set up SVM
            if node.is_root():
                
                print "training svm at top level"
                svm = SVMLight(cost, k, lab)

            else:
                
                # regularize vs parent predictor
                
                #weight = node.edge_weight
                weight = param.transform
                
                print "current edge_weight:", weight, " ,name:", node.name
                
                parent_svm = node.parent.predictor
                
                svm = DomainAdaptationSVM(cost, k, lab, parent_svm, weight)
                #svm.set_train_factor(param.base_similarity)
             

            if param.flags["normalize_cost"]:
                
                norm_c_pos = param.cost / float(len([l for l in lab.get_labels() if l==1]))
                norm_c_neg = param.cost / float(len([l for l in lab.get_labels() if l==-1]))
                svm.set_C(norm_c_neg, norm_c_pos)
             

            # set epsilon
            if param.flags.has_key("epsilon"):
                svm.set_epsilon(param.flags["epsilon"])

               
            # enable output
            svm.io.enable_progress()
            svm.io.set_loglevel(shogun.Classifier.MSG_INFO)
            

            svm.set_train_factor(param.flags["train_factor"])
            svm.train()
            
            # attach svm to node
            node.predictor = svm
            
            # save some information
            self.additional_information[node.name + " svm obj"] = svm.get_objective()
            self.additional_information[node.name + " svm num sv"] = svm.get_num_support_vectors()
            self.additional_information[node.name + " runtime"] = svm.get_runtime()
            


        #####################################################
        #    Wrap things up    
        #####################################################
 
        # wrap up predictors for later use
        predictors = {}

        for leaf in root.get_leaves():
            
            predictors[leaf.name] = leaf.predictor
            
            assert(leaf.predictor!=None)
          
        # make sure we have the same keys (potentiall in a different order)  
        sym_diff_keys = set(train_data.keys()).symmetric_difference(set(predictors.keys()))
        assert len(sym_diff_keys)==0, "symmetric difference between keys non-empty: " + str(sym_diff_keys)  



        # save graph plot
        mypath = "/fml/ag-raetsch/share/projects/multitask/graphs/"
        filename = mypath + "graph_" + str(param.id)
        root.plot(filename)#, plot_cost=True, plot_B=True)


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


        # shogun data
        feat = shogun_factory_new.create_features(examples, self.param)

        out = predictor.classify(feat).get_labels()        

        return out

