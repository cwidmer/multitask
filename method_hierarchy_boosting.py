#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 27.04.2009
@author: Christian Widmer
@summary: Hierarchical SVM-based multitask method trained using boosting

"""


import shogun
from shogun.Classifier import SVMLight
import shogun_factory_new as shogun_factory

from base_method import MultiMethod
from helper import SequencesHandler, split_data
from boosting import solve_boosting, solve_nu_svm
import cvxmod
import numpy

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

        # split data for training weak_learners and boosting
        (train_weak, train_boosting) = split_data(train_data, 4)
                  

        for task_id in train_data.keys():
            print "task_id:", task_id
            
        
        root = param.taxonomy.data
        
        # train on first part of dataset (evaluate on other)
        (classifiers, classifier_at_node) = self._inner_train(train_weak, param)

        # train on entire dataset
        (final_classifiers, final_classifier_at_node) = self._inner_train(train_data, param)

        ###

        print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        print "done training weak learners"

        #####################################################
        #    perform boosting and wrap things up    
        #####################################################

        # wrap up predictors for later use
        predictors = {}

        for task_name in train_boosting.keys():
            
            
            instances = train_boosting[task_name]
            
            # get ids of predecessor nodes            
            node_names = [node.name for node in root.get_node(task_name).get_path_root()]
            node_names.append(task_name)
            
            print "node: %s --> %s" % (task_name, str(node_names))
            
            N = len(instances)
            
            
            if param.flags["use_all_nodes"]:
                # use classifiers only from parent nodes
                F = len(classifiers)
                tmp_classifiers = classifiers
                tmp_final_classifiers = final_classifiers
                
            else:
                # use classifiers from all leaves
                F = len(node_names)
                tmp_classifiers = []
                tmp_final_classifiers = []
            
            
            examples = [inst.example for inst in instances]
            labels = [inst.label for inst in instances]
            
            # dim = (F x N)
            out = cvxmod.zeros((N,F))
            
            for i in xrange(F):
                
                if param.flags["use_all_nodes"]:
                    svm = classifiers[i]
                else:
                    svm = classifier_at_node[node_names[i]]
                    tmp_classifiers.append(svm)

                    final_svm = final_classifier_at_node[node_names[i]]
                    tmp_final_classifiers.append(final_svm)
                    
                tmp_out = self._predict_weak(svm, examples, task_name)

                if param.flags["signum"]:
                    out[:,i] = numpy.sign(tmp_out)
                else:
                    out[:,i] = tmp_out
            
            
            if param.flags["boosting"] == "ones":
                weights = numpy.ones(F)/float(F)
            if param.flags["boosting"] == "L1":
                weights = solve_boosting(out, labels, param.transform, solver="glpk")
            if param.flags["boosting"] == "L2":            
                weights = solve_nu_svm(out, labels, param.transform, solver="glpk", reg=False)
            if param.flags["boosting"] == "L2_reg":            
                weights = solve_nu_svm(out, labels, param.transform, solver="glpk", reg=True)
            
            
            predictors[task_name] = (tmp_final_classifiers, weights)
            
        
        #####################################################
        #    Some sanity checks
        ##################################################### 
        
        # make sure we have the same keys (potentiall in a different order)  
        sym_diff_keys = set(train_weak.keys()).symmetric_difference(set(predictors.keys()))
        assert len(sym_diff_keys)==0, "symmetric difference between keys non-empty: " + str(sym_diff_keys)  


        # save graph plot
        mypath = "/fml/ag-raetsch/share/projects/multitask/graphs/"
        filename = mypath + "graph_" + str(param.id)
        root.plot(filename)#, plot_cost=True, plot_B=True)


        return predictors



    def _inner_train(self, train_data, param):
        """
        perform inner training by processing the tree
        """

        data_keys = []
        # top-down processing of taxonomy


        classifiers = []
        classifier_at_node = {}

        root = param.taxonomy.data

        grey_nodes = [root]
        
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
            
            data_keys.append(node.get_data_keys())
    
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
            k = shogun_factory.create_kernel(examples, param)
            lab = shogun_factory.create_labels(labels)


            #####################################################
            #    train weak learners    
            #####################################################
            
            cost = param.cost
            
            # set up svm
            svm = SVMLight(cost, k, lab)
                        
            if param.flags["normalize_cost"]:
                # set class-specific Cs
                norm_c_pos = param.cost / float(len([l for l in labels if l==1]))
                norm_c_neg = param.cost / float(len([l for l in labels if l==-1]))
                svm.set_C(norm_c_neg, norm_c_pos)
            
            
            print "using cost: negative class=%f, positive class=%f" % (norm_c_neg, norm_c_pos) 
            
            # enable output
            svm.io.enable_progress()
            svm.io.set_loglevel(shogun.Classifier.MSG_INFO)
            
            # train
            svm.train()
            
            # append svm object
            classifiers.append(svm)
            classifier_at_node[node.name] = svm                            
            
            # save some information
            self.additional_information[node.name + " svm obj"] = svm.get_objective()
            self.additional_information[node.name + " svm num sv"] = svm.get_num_support_vectors()
            self.additional_information[node.name + " runtime"] = svm.get_runtime()


        return (classifiers, classifier_at_node)


    def _predict_weak(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: SVMLight
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
        feat = shogun_factory.create_features(examples)

        out = predictor.classify(feat).get_labels()        

        return out



    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: SVMLight
        @param examples: list of examples
        @type examples: list
        @param task_id: task identifier
        @type task_id: int
        
        @return: svm output
        @rtype: list<float>
        """

        (classifiers, weights) = predictor
        

        assert(len(classifiers) == len(weights.tolist()))
        
        out = numpy.zeros(len(examples))

        for i in xrange(len(classifiers)):
            out += weights[i] * numpy.array(self._predict_weak(classifiers[i], examples, task_id))        


        return out



def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = -1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    #multi_split_set = MultiSplitSet.get(387)
    #multi_split_set = MultiSplitSet.get(407)
    multi_split_set = MultiSplitSet.get(399)

    #dataset_name = multi_split_set.description

    
    # create mock param object by freezable struct
    param = Options()
    param.kernel = "WeightedDegreeStringKernel"#"PolyKernel" 
    param.wdk_degree = 4
    param.cost = 1.0
    param.transform = 0.2
    param.taxonomy = multi_split_set.taxonomy
    param.id = 666
    
    flags= {}
    #flags["boosting"] = "ones"
    flags["boosting"] = "L1"
    #flags["boosting"] = "L2"
    #flags["boosting"] = "L2_reg"
    flags["use_all_nodes"] = True
    flags["signum"] = False
    flags["normalize_cost"] = True
    param.flags = flags
    
    param.freeze()
    

    data_train = multi_split_set.get_train_data(SPLIT_POINTER)
    data_eval = multi_split_set.get_eval_data(SPLIT_POINTER)


    # train
    mymethod = Method(param)
    mymethod.train(data_train)


    assessment = mymethod.evaluate(data_eval)
    
    print assessment
    
    assessment.destroySelf()
    
    
if __name__ == "__main__":
    main()

