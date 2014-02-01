#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009-2011 Christian Widmer
# Copyright (C) 2009-2011 Max-Planck-Society

"""
Created on 12.07.2009
@author: Christian Widmer
@summary: Hierarchical SVM-based multitask method with local cross-validation 

"""


import shogun_factory_new as shogun_factory

from base_method import MultiMethod
import helper

import shogun
import numpy
import numpy.random

import pdb

debug = False

#import pylab


#RANGE = [pow(10, i) for i in (numpy.double(range(0, 61))/10 - 4)] #[0.1, 1.0, 10]
#RANGE = numpy.exp(numpy.linspace(numpy.log(0.1), numpy.log(100), 10))
RANGE = [float(c) for c in numpy.linspace(0.01, 2, 10)]
#RANGE = [0.1, 1.0, 10]
#TARGET_TASK = "toy_1"
TARGET_TASK = "hsa" #"thaliana" #drosophila" #"thaliana"
TARGET_PARAM = "B" #"both" #"C"   
TARGET_MEASURE = "auPRC"
SPLIT_POINTER = -1 #-1 #use whole training and testing data

FOLD = 3

#numpy.random.seed(666) # important to keep this fix


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

        root = param.taxonomy.data
        
        print ">>>" + str(param.taxonomy.data) + "<<<"
        print "initial root weight:", root.edge_weight
        print "tasks", train_data.keys()
        print "tax keys", root.get_data_keys()


        numpy.random.seed(1)
        # prepare data splits for inner validation
        
        # set up validation strategy
        # this has to be done here, because the training set CANNOT contain
        # any examples that will be used to evaluate further down the tree
        # 
        # also by doing it this way, we have equally many examples from each
        # task in each split
        
        inner_train_data = {}
        inner_eval_data = {}
        
        for task_id in root.get_data_keys():
            
            idx = range(len(train_data[task_id]))
            
            idx_pos = [idx for idx in range(len(train_data[task_id])) if train_data[task_id][idx].label == 1]
            idx_neg = [idx for idx in range(len(train_data[task_id])) if train_data[task_id][idx].label == -1]
            
            numpy.random.shuffle(idx_pos)
            numpy.random.shuffle(idx_neg)

            # distribute pos/negs evenly across splits            
            splits_pos = helper.split_list(idx_pos, FOLD)
            splits_neg = helper.split_list(idx_neg, FOLD)
        
            eval_split_id = 0
            train_idx_pos = list(helper.flatten([splits_pos[j] for j in xrange(FOLD) if j!=eval_split_id]))
            train_idx_neg = list(helper.flatten([splits_neg[j] for j in xrange(FOLD) if j!=eval_split_id]))
            
            train_idx = train_idx_pos
            train_idx.extend(train_idx_neg)
            numpy.random.shuffle(train_idx)
            
            
            eval_idx_pos = splits_pos[eval_split_id]
            eval_idx_neg = splits_neg[eval_split_id]
            
            eval_idx = eval_idx_pos
            eval_idx.extend(eval_idx_neg)
            
            numpy.random.shuffle(eval_idx)
            
            
            
            #            numpy.random.shuffle(idx)
            #    
            #            splits = helper.split_list(idx, FOLD)
            #        
            #            eval_split_id = 0
            #            train_idx = list(helper.flatten([splits[j] for j in xrange(FOLD) if j!=eval_split_id]))
            #            eval_idx = splits[eval_split_id]
            
            # make sure idx lists are disjoint
            assert( len(set(train_idx).intersection(set(eval_idx))) == 0 )
           
            print "len train data", len(train_data[task_id]), task_id
 
            # select data sets
            inner_train_data[task_id] = [train_data[task_id][idx] for idx in train_idx]
            inner_eval_data[task_id] = [train_data[task_id][idx] for idx in eval_idx]

        

        ###########################################################
        #    Learn Taxonomy Parameters
        ###########################################################
        
        grey_nodes = [root]
        
        #initialize inner cost
        inner_cost = param.cost
        
        
        while len(grey_nodes)>0:
           
            # fetch next node to process
            node = grey_nodes.pop(0) #pop first item
            
            # enqueue children
            if not node.is_leaf():
                grey_nodes.extend(node.children)
    
    
    
            ###################################
            #train current node
            ###################################
            
            
            # concatenate instances from all task for nodes below
            instance_set_train = list(helper.flatten([inner_train_data[key] for key in node.get_data_keys()]))
            instance_set_eval = list(helper.flatten([inner_eval_data[key] for key in node.get_data_keys()]))
            
            # shuffle to avoid having instances from one task in consecutive order
            numpy.random.shuffle(instance_set_train)
            numpy.random.shuffle(instance_set_eval)

            # extract examples and labels
            train_examples = [inst.example for inst in instance_set_train]
            train_labels = [inst.label for inst in instance_set_train]
            
            eval_examples = [inst.example for inst in instance_set_eval]
            eval_labels = [inst.label for inst in instance_set_eval]
            
            
            #import copy
            #debug_examples = copy.copy(train_examples)
            #debug_examples.extend(eval_examples)
            
            #debug_labels = copy.copy(train_labels)
            #debug_labels.extend(eval_labels)
                            
            # only local xval for leaves
            #if node.is_root():
            #    inner_param = 0.0
            #    predictor = self._train_inner_classifier(node, train_examples, train_labels, param, inner_param, param.cost)
            
            #else:
            #TODO: also perform inner validation on non-leaves 
            if node.is_leaf():# not node.is_root():

                print "performing inner xval at node", node.name               
 
                # perform local model selection
                result_dict = self._perform_inner_xval(node, train_examples, train_labels, eval_examples, eval_labels, param)
            
                # use dict for returning args to avoid order glitches
                inner_edge_weight = result_dict["best_edge_weight"]
                inner_cost = result_dict["best_inner_cost"]
                predictor = result_dict["best_predictor"]
                
                
            else:
                # for non-leaves train without model selection
                inner_edge_weight = param.transform
                inner_cost = param.cost    
                
                predictor = self._train_inner_classifier(node, train_examples, train_labels, param, inner_edge_weight, inner_cost)
                #predictor = self._train_inner_classifier(node, debug_examples, debug_labels, param, inner_edge_weight, inner_cost)
                
            
            
            node.predictor = predictor
            node.edge_weight = inner_edge_weight
            node.cost = inner_cost



        ###########################################################
        # Retrain on whole training set with optimal parameters
        ###########################################################

        grey_nodes = [root]
        
        
        while len(grey_nodes)>0:
           
            node = grey_nodes.pop(0) #pop first item
            
            # enqueue children
            if not node.is_leaf():
                grey_nodes.extend(node.children)
    
    
            # fetch all data that belongs to leaves underneath current node
            instance_set_retrain = list(helper.flatten([train_data[key] for key in node.get_data_keys()]))
            
            # shuffle instances
            numpy.random.shuffle(instance_set_retrain)

            # extract examples and labels
            examples = [inst.example for inst in instance_set_retrain]
            labels = [inst.label for inst in instance_set_retrain]


            print "FINAL TRAIN on " + node.name + " C=" + str(node.cost) + " B=" + str(node.edge_weight)
            predictor = self._train_inner_classifier(node, examples, labels, param, node.edge_weight, node.cost)
            
            # attach predictor to node
            node.predictor = predictor



        #####################################################
        #    Wrap things up    
        #####################################################
 
        # wrap up predictors for later use
        predictors = {}

        for leaf in root.get_leaves():

            assert(leaf.predictor!=None)
            
            predictors[leaf.name] = leaf.predictor
            

        # make sure we have the same keys (potentially in a different order)
        sym_diff_keys = set(train_data.keys()).symmetric_difference(set(predictors.keys()))
        assert len(sym_diff_keys)==0, "symmetric difference between keys non-empty: " + str(sym_diff_keys)  


        # save graph plot
        mypath = "/fml/ag-raetsch/share/projects/multitask/graphs/"
        filename = mypath + "graph_" + str(param.id)
        filename_perf = mypath + "performances_" + str(param.id)
        
        
        helper.save(filename_perf, result_dict["performances"])
        print "saving results to:", filename_perf 
        
        root.plot(filename, plot_cost=True, plot_B=True)


        return predictors

   
    def _perform_inner_xval(self, node, train_examples, train_labels, eval_examples, eval_labels, param):
        
        
        if TARGET_PARAM=="B":
            inner_edge_weights = RANGE
            inner_costs = [param.cost]
            
        if TARGET_PARAM=="C":
            inner_edge_weights = [param.transform]
            inner_costs = RANGE
        
        if TARGET_PARAM=="both":
            inner_edge_weights = RANGE
            inner_costs = RANGE
        
        
        if node.is_root():
            inner_edge_weights = [0.0]
        
        
        # set up variables to keep track of best values
        tmp_best_performance = -99999999
        tmp_best_edge_weight = -99999999
        tmp_best_cost = -99999999
        tmp_best_predictor = None
    
        
        param_performances = numpy.zeros((len(inner_edge_weights), len(inner_costs)))
        #debug_param_performances = numpy.zeros((len(inner_edge_weights), len(inner_costs)))


        #import expenv
        #mss = expenv.MultiSplitSet.get(384)
        #debug_test_set = mss.get_eval_data(-1)
        #debug_test_examples = [inst.example for inst in debug_test_set[node.name]]
        #debug_test_labels = [inst.label for inst in debug_test_set[node.name]]        
        
        for (k, inner_edge_weight) in enumerate(inner_edge_weights):
        
            for (m, inner_cost) in enumerate(inner_costs):
        
                            
                predictor = self._train_inner_classifier(node, train_examples, train_labels, param, inner_edge_weight, inner_cost)

                # set stuff temporarily
                #node.predictor = predictor
                #node.edge_weight = inner_edge_weight

                performance = self._inner_assessment(predictor, eval_examples, eval_labels)

                #predictor = self._train_inner_classifier(node, train_examples, train_labels, param, 10, 2); performance = self._inner_assessment(predictor, eval_examples, eval_labels); print performance
                ######
                #import copy
                #debug_examples = copy.copy(train_examples)
                #debug_examples.extend(eval_examples)
                
                #debug_labels = copy.copy(train_labels)
                #debug_labels.extend(eval_labels)
                
                #debug_predictor = self._train_inner_classifier(node, debug_examples, debug_labels, param, inner_edge_weight, inner_cost)

                
                #debug_performance = self._inner_assessment(debug_predictor, debug_test_examples, debug_test_labels)
                #debug_param_performances[k][m] = debug_performance
                
                
                # save keepers
                if performance > tmp_best_performance:
                    # save performance
                    tmp_best_performance = performance
                    
                    # save parameters
                    tmp_best_cost = inner_cost
                    tmp_best_edge_weight = inner_edge_weight
                    tmp_best_predictor = predictor
                    

                    print "new best:"
                    print "inner edge weight", inner_edge_weight
                    print "inner cost", inner_cost
                    print "performance", performance
                    print "-------------"
                    
                # save performance
                param_performances[k][m] = performance
        
            
        # double check results
        best_idx = numpy.argmax(param_performances) # best idx of flattened (!!) array        
        dims = param_performances.shape
        (best_idx_edge_weight, best_idx_cost) = numpy.unravel_index(best_idx, dims)
        
        
        assert (abs(tmp_best_cost - inner_costs[best_idx_cost]) < 0.01)
        assert abs(tmp_best_edge_weight - inner_edge_weights[best_idx_edge_weight]) < 0.01, "tmp: " + str(tmp_best_edge_weight) + ", inner: " + str(inner_edge_weights[best_idx_edge_weight]) + " best performance:" + str(tmp_best_performance) + "other best: " + str(numpy.max(param_performances))  
        assert (abs(tmp_best_performance - param_performances[best_idx_edge_weight][best_idx_cost]) < 0.01)
        
        
        # debug
        #TODO: clean this up
        if False and node.name==TARGET_TASK:
            
            if TARGET_PARAM=="C":
                self.debug_perf = param_performances[0,:]
                self.debug_best_idx = best_idx_cost
            if TARGET_PARAM=="B":
                self.debug_perf = param_performances[:,0]
                self.debug_best_idx = best_idx_edge_weight
            if TARGET_PARAM=="both":
                self.debug_perf = param_performances
                self.debug_best_idx = (best_idx_edge_weight, best_idx_cost)
        
        
        # prepare result (avoid order glitches)
        result = {}
        
        result["best_edge_weight"] = tmp_best_edge_weight
        result["best_inner_cost"] = tmp_best_cost
        result["best_predictor"] = tmp_best_predictor
        result["performances"] = param_performances
        
        
        return result
    

    def _train_inner_classifier(self, node, examples, labels, param, inner_param, inner_cost):


        # set up presvm
        if node.is_root():
            # no parent at root node
            parent_svm = None 
            svm = shogun_factory.create_initialized_svm(param, examples, labels)

        else:
            # regularize against parent predictor
            parent_svm = node.parent.predictor
            svm = shogun_factory.create_initialized_domain_adaptation_svm(param, examples, labels, parent_svm, inner_param)
            print "current edge_weight:", inner_param, " ,name:", node.name


        # create SVM object
            
        svm.train()
        
        return svm



    def _inner_assessment(self, predictor, eval_examples, eval_labels):
        
        feat = shogun_factory.create_features(eval_examples, self.param)

        # use predictor attached to current leaf
        out = predictor.classify(feat).get_labels()
                        
        
        # return performance measure
        if TARGET_MEASURE=="auPRC":
            return helper.calcprc(out, eval_labels)[0]
        elif TARGET_MEASURE=="auROC":
            return helper.calcroc(out, eval_labels)[0]
        else:
            assert(False), "unknown measure type"
        
            
    
    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: array
        @param examples: list of examples
        @type examples: list
        @param task_id: task identifier
        @type task_id: int
        
        @return: a performance value for each example
        @rtype: list<float>
        """


        feat = shogun_factory.create_features(examples, self.param)

        out = predictor.classify(feat).get_labels()        

        return out



def create_plot_regular(param, data_train, data_eval):
    """
    this will create a performance plot for manually set values
    """
    
    # train regular hierarchical
    import method_hierarchy_svm_new
    
    performances = []
    
    inner_range = [1.0]
    
    
    if TARGET_PARAM=="both":
        inner_range = RANGE
        performances = numpy.zeros((len(RANGE), len(RANGE)))
        
    
    for (i,value) in enumerate(RANGE):
        
        for (j,inner_value) in enumerate(inner_range):
        
            toy = [node for node in param.taxonomy.data.get_all_nodes() if node.name==TARGET_TASK][0]
            assert(toy.name==TARGET_TASK)
            
            # set new value
            if TARGET_PARAM=="C":
                toy.cost = value
            if TARGET_PARAM=="B":
                toy.edge_weight = value
            if TARGET_PARAM=="both":
                toy.cost=value
                toy.edge_weight=inner_value
                
        
            regular = method_hierarchy_svm_new.Method(param)
            regular.train(data_train)
            
            out = [a for a in regular.evaluate(data_eval).assessments if a.task_id==TARGET_TASK][0]
            assert(out.task_id==TARGET_TASK)
            
            print out
            
            assessment = {"auROC": out.auROC, "auPRC": out.auPRC}      
            
            out.destroySelf()

            if TARGET_PARAM=="both":
                performances[i][j] = assessment[TARGET_MEASURE]
            else:
                performances.append(assessment[TARGET_MEASURE])
            
        
    return performances

    
    
def create_plot_inner(param, data_train, data_eval):
    """
    this will create a performance plot for manually set values
    """


    # train hierarchical xval
    mymethod = Method(param)
    mymethod.train(data_train)
    out = [a for a in mymethod.evaluate(data_eval).assessments if a.task_id==TARGET_TASK][0]
    
    
    assessment = {"auROC": out.auROC, "auPRC": out.auPRC}
    out.destroySelf()
    
    return (mymethod.debug_perf, assessment[TARGET_MEASURE], mymethod.debug_best_idx)

 
   

def main():
    
    
    print "starting debugging:"
    

    from expenv import MultiSplitSet
    from helper import Options 
    from task_similarities import dataset_to_hierarchy
    
    # select dataset
    #multi_split_set = MultiSplitSet.get(317)
    multi_split_set = MultiSplitSet.get(432)
    #multi_split_set = MultiSplitSet.get(2) #small splicing
    #multi_split_set = MultiSplitSet.get(377) #medium splicing

    dataset_name = multi_split_set.description

    # flags
    flags = {}
    flags["normalize_cost"] = False
    flags["epsilon"] = 1.0 
    #0.005
    flags["kernel_cache"] = 1000
    flags["use_bias"] = False 

    # arts params
    flags["svm_type"] = "liblineardual"

    flags["degree"] = 24
    flags["degree_spectrum"] = 4
    flags["shifts"] = 0 #32
    flags["train_factor"] = 1
    flags["center_offset"] = 70
    flags["center_pos"] = 500


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

    (perf_xval, final_pred, best_idx_cost) = create_plot_inner(param, data_train, data_eval)
    perf_regular = create_plot_regular(param, data_train, data_eval)


    # plot performances
      
    import pylab
    
    if TARGET_PARAM=="both":


        #X,Y = pylab.meshgrid(range(len(RANGE)), range(len(RANGE)))
        
        cmap = pylab.cm.get_cmap('jet', 20)    # 10 discrete colors
        
        pylab.contourf(RANGE, RANGE, perf_xval, cmap=cmap)
        #im = pylab.imshow(perf_xval, cmap=cmap, interpolation='bilinear')
        pylab.axis('on')
        pylab.colorbar()
        
        pylab.title("mss:" + str(multi_split_set.id) + ", task:" + TARGET_TASK + " , param:" + TARGET_PARAM +  ", split:" + str(SPLIT_POINTER))
        
        pylab.show()
    
    else:
        
        pylab.semilogx(RANGE, perf_regular, "g-o")
        pylab.semilogx(RANGE, perf_xval, "b-o")
        #pylab.semilogx([a*0.66 for a in RANGE], perf_xval, "b-o")
        
        #pylab.plot(numpy.array(perf_regular) - numpy.array(perf_xval), "y-o")
        
        #pylab.plot([best_idx_cost], [final_pred], "r+")
        pylab.axhline(y=final_pred, color="r")
        pylab.axvline(x=RANGE[best_idx_cost], color="r")
        pylab.axvline(x=1.0, color="g")
        
        pylab.ylabel(TARGET_MEASURE)
        pylab.xlabel(TARGET_PARAM)
        
        pylab.legend( ("outer", "inner xval"), loc="best")
        pylab.title("mss:" + str(multi_split_set.id) + ", task:" + TARGET_TASK + " , degree:" + str(param.wdk_degree) +  ", split:" + str(SPLIT_POINTER))
        
        pylab.show()
        
    
    
if __name__ == "__main__":
    main()
    

