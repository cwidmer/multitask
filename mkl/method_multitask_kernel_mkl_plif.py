#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 06.01.2010
@author: Christian Widmer
@summary: Implementation of the augmented SVM multitask method
This methods uses a modified kernel such that tasks, 
which are close to each other are more similar by default.
"""


import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight, MKLClassification, ST_CPLEX, ST_NEWTON, ST_DIRECT, ST_GLPK
from shogun.Kernel import MultitaskKernelPlifNormalizer, KernelNormalizerToMultitaskKernelPlifNormalizer
from base_method import MultiMethod, PreparedMultitaskData
import task_similarities
import numpy
import helper


debug = False



class Method(MultiMethod):



    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        """
        
                
        # merge data sets
        data = PreparedMultitaskData(train_data, shuffle=False)
        
        # create shogun data objects
        base_wdk = shogun_factory.create_kernel(data.examples, param)
        lab = shogun_factory.create_labels(data.labels)

        # support
        support = numpy.linspace(0, 1, 5)

        # set normalizer
        normalizer = MultitaskKernelPlifNormalizer(support, data.task_vector_nums) 
        
        # fetch taxonomy from parameter object
        taxonomy = param.taxonomy.data

        taxonomy.plot()
        import os
        os.system("evince demo.png &")
        
        # compute distances
        distances = numpy.zeros((data.get_num_tasks(), data.get_num_tasks()))
        
        for (i,task_name_lhs) in enumerate(data.get_task_names()):
            for (j, task_name_rhs) in enumerate(data.get_task_names()):
                
                distances[i,j] = task_similarities.compute_hop_distance(taxonomy, task_name_lhs, task_name_rhs)

                
        # normalize distances
        distances = distances / numpy.max(distances)

        
        # set distances
        for (i,task_name_lhs) in enumerate(data.get_task_names()):
            for (j, task_name_rhs) in enumerate(data.get_task_names()):
                        
                normalizer.set_task_distance(i, j, distances[i,j])

        
        # assign normalizer
        base_wdk.set_normalizer(normalizer)
        base_wdk.init_normalizer()

        svm = None
        
        debug_weights = {}
                
        num_subk = base_wdk.get_num_subkernels()
        
        print "num subkernels:", num_subk
        
        #print "subkernel weights:", base_wdk.get_subkernel_weights()
        
        debug_weights["before"] = [normalizer.get_beta(i) for i in range(num_subk)]        
        
        print "using MKL:", (param.transform >= 1.0)
        
        if param.transform >= 1.0:
        
        
            num_threads = 4

            
            svm = MKLClassification()
            
            svm.set_mkl_norm(param.transform)
            #svm.set_solver_type(ST_CPLEX) #GLPK) #DIRECT) #NEWTON)#ST_CPLEX) 
        
            svm.set_C(param.cost, param.cost)
            
            svm.set_kernel(base_wdk)
            svm.set_labels(lab)
            
            svm.parallel.set_num_threads(num_threads)
            svm.set_linadd_enabled(False)
            svm.set_batch_computation_enabled(False)
            
            svm.train()
        
            #print "subkernel weights (after):", base_wdk.get_subkernel_weights()    
            
        else:
            
            # create SVM (disable unsupported optimizations)
            svm = SVMLight(param.cost, base_wdk, lab)
            svm.set_linadd_enabled(False)
            svm.set_batch_computation_enabled(False)
            
            svm.train()
        
        
        print "svm objective:", svm.get_objective()     
        


        debug_weights["after"] = [normalizer.get_beta(i) for i in range(num_subk)]            
        
        # debugging output
        print "debug weights (before/after):"
        print debug_weights["before"]
        print debug_weights["after"]
        print ""
        
        
        # wrap up predictors
        svms = {}
            
        # use a reference to the same svm several times
        for task_name in train_data.keys():
            svms[task_name] = (svm, data.name_to_id(task_name))


        return svms
    
    

    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor
        @type predictor: dict<str, tuple<SVMLight, Kernel, int> >
        @param examples: list of examples
        @type examples: list<str> 
        @param task_id: task id
        @type task_id: str
        """

        (svm, task_num) = predictor

        # shogun data
        feat = shogun_factory.create_features(examples)

        # fetch kernel normalizer
        normalizer = svm.get_kernel().get_normalizer()
        
        # cast using dedicated SWIG-helper function
        normalizer = KernelNormalizerToMultitaskKernelPlifNormalizer(normalizer)
        
        # set task vector
        normalizer.set_task_vector_rhs([task_num]*len(examples))

        # predict
        out = svm.classify(feat).get_labels()
        
        
        return out


        

def create_plot_inner(param, data_train, data_eval):
    """
    this will create a performance plot for manually set values
    """


    # train hierarchical xval
    mymethod = Method(param)
    mymethod.train(data_train)
    
    assessment = mymethod.evaluate(data_eval)
    
    print assessment
    
    assessment.destroySelf();
    


def main():
        
    print "starting debugging:"

    SPLIT_POINTER = 1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(379)

    dataset_name = multi_split_set.description

    print "dataset_name", dataset_name
    
    #create mock taxonomy object by freezable struct
    #taxonomy = Options()
    #taxonomy.data = taxonomy_graph.data
    #taxonomy.description = dataset_name
    #taxonomy.freeze()
    
    
    #create mock param object by freezable struct
    param = Options()
    param.kernel = "WeightedDegreeStringKernel"
    param.wdk_degree = 1
    param.cost = 1.0
    param.transform = 2.0
    param.taxonomy = multi_split_set.taxonomy
    param.id = 666
    
    param.freeze()
    

    data_train = multi_split_set.get_train_data(SPLIT_POINTER)
    data_eval = multi_split_set.get_eval_data(SPLIT_POINTER)

    # train hierarchical xval
    mymethod = Method(param)
    mymethod.train(data_train)
    
    assessment = mymethod.evaluate(data_eval)
    
    print assessment
    
    assessment.destroySelf();
    
    
    
if __name__ == "__main__":
    main()

