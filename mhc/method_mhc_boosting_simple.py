#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010 Christian Widmer
# Copyright (C) 2010 Max-Planck-Society

"""
Created on 4.06.2009
@author: Christian Widmer
@summary: Use boosting to combine MTL-kernel and dirac-based classifier
"""


from shogun.Kernel import MultitaskKernelNormalizer, KernelNormalizerToMultitaskKernelNormalizer
import shogun_factory_new as shogun_factory

from base_method import MultiMethod, PreparedMultitaskData
from helper import split_data
from boosting import solve_boosting, solve_nu_svm
import cvxmod
import numpy

debug = False



class Method(MultiMethod):
    """
    boosting based combination method
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

        # split data for training weak_learners and boosting
        (train_weak, train_boosting) = split_data(train_data, 4)
        
        # train on first part of dataset (evaluate on other)
        prepared_data_weak = PreparedMultitaskData(train_weak, shuffle=False)
        classifiers = self._inner_train(prepared_data_weak, param)

        # train on entire dataset
        prepared_data_final = PreparedMultitaskData(train_data, shuffle=False)
        final_classifiers = self._inner_train(prepared_data_final, param)


        print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        print "done training weak learners"

        #####################################################
        #    perform boosting and wrap things up    
        #####################################################

        # wrap up predictors for later use
        predictors = {}

        for task_name in train_boosting.keys():
            
            instances = train_boosting[task_name]
            
            N = len(instances)
            F = len(classifiers)
            
            examples = [inst.example for inst in instances]
            labels = [inst.label for inst in instances]
            
            # dim = (F x N)
            out = cvxmod.zeros((N,F))
            
            for i in xrange(F):
                    
                svm = classifiers[i]
                        
                tmp_out = self._predict_weak(svm, examples, prepared_data_weak.name_to_id(task_name), param)

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
            
            
            predictors[task_name] = (final_classifiers, weights, prepared_data_final.name_to_id(task_name), param)
            
            
            assert prepared_data_final.name_to_id(task_name)==prepared_data_weak.name_to_id(task_name), "name mappings don't match"
            
        
        #####################################################
        #    Some sanity checks
        ##################################################### 
        
        # make sure we have the same keys (potentiall in a different order)  
        sym_diff_keys = set(train_weak.keys()).symmetric_difference(set(predictors.keys()))
        assert len(sym_diff_keys)==0, "symmetric difference between keys non-empty: " + str(sym_diff_keys)  


        return predictors



    def _inner_train(self, prepared_data, param):
        """
        perform inner training by processing the tree
        """


        # init seq handler 
        
        classifiers = []


        #################
        # mtk
        normalizer = MultitaskKernelNormalizer(prepared_data.task_vector_nums)
        
        from method_mhc_rbf import SequencesHandlerRbf
        task_kernel = SequencesHandlerRbf(1, param.base_similarity, prepared_data.get_task_names(), param.flags["wdk_rbf_on"])
        

        # set similarity
        for task_name_lhs in prepared_data.get_task_names():
            for task_name_rhs in prepared_data.get_task_names():
                
                similarity = task_kernel.get_similarity(task_name_lhs, task_name_rhs)
                                
                normalizer.set_task_similarity(prepared_data.name_to_id(task_name_lhs), prepared_data.name_to_id(task_name_rhs), similarity)
           
        
        lab = shogun_factory.create_labels(prepared_data.labels)
        
        print "creating empty kernel"
        kernel = shogun_factory.create_kernel(prepared_data.examples, param)
        
        print "setting normalizer"
        kernel.set_normalizer(normalizer)
        kernel.init_normalizer()

        svm = shogun_factory.create_svm(param, kernel, lab)
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)

        # train SVM
        svm.train()
        
        classifiers.append(svm)

        #################
        # dirac             
            #import pdb
            #pdb.set_trace()
            

        svm_dirac = self._dirac_train(prepared_data, param)

        classifiers.append(svm_dirac)
        
        ##
        #union
        
        #svm_union = self._union_train(prepared_data, param)

        #classifiers.append(svm_union)
        


        return classifiers



    def _dirac_train(self, prepared_data, param):
        """
        perform inner training by processing the tree
        """

        
        normalizer = MultitaskKernelNormalizer(prepared_data.task_vector_nums)
        
        # set similarity
        for task_name_lhs in prepared_data.get_task_names():
            for task_name_rhs in prepared_data.get_task_names():
                
                if task_name_lhs == task_name_rhs:
                    similarity = 1.0
                else:
                    similarity = 0.0
                                
                normalizer.set_task_similarity(prepared_data.name_to_id(task_name_lhs), prepared_data.name_to_id(task_name_rhs), similarity)

                           
        lab = shogun_factory.create_labels(prepared_data.labels)
        
        print "creating empty kernel"
        kernel = shogun_factory.create_kernel(prepared_data.examples, param)
        
        print "setting normalizer"
        kernel.set_normalizer(normalizer)
        kernel.init_normalizer()

        svm = shogun_factory.create_svm(param, kernel, lab)
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)

        # train SVM
        svm.train()
        
        
        return svm


    def _union_train(self, prepared_data, param):
        """
        perform inner training by processing the tree
        """

    
        normalizer = MultitaskKernelNormalizer(prepared_data.task_vector_nums)
        
        # set similarity
        for task_name_lhs in prepared_data.get_task_names():
            for task_name_rhs in prepared_data.get_task_names():
                
                similarity = 1.0
                                
                normalizer.set_task_similarity(prepared_data.name_to_id(task_name_lhs), prepared_data.name_to_id(task_name_rhs), similarity)

        
        lab = shogun_factory.create_labels(prepared_data.labels)
        
        print "creating empty kernel"
        kernel = shogun_factory.create_kernel(prepared_data.examples, param)
        
        print "setting normalizer"
        kernel.set_normalizer(normalizer)
        kernel.init_normalizer()

        svm = shogun_factory.create_svm(param, kernel, lab)
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)

        # train SVM
        svm.train()


        return svm
    

    def _predict_weak(self, predictor, examples, task_id, param):
        """
        make prediction using a weak classifier

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
        feat = shogun_factory.create_features(examples, param)

        # fetch kernel normalizer & update task vector
        normalizer = predictor.get_kernel().get_normalizer()
        
        # cast using dedicated SWIG-helper function
        normalizer = KernelNormalizerToMultitaskKernelNormalizer(normalizer)
        
        # set task vector
        normalizer.set_task_vector_rhs([task_id]*len(examples))

        out = predictor.classify(feat).get_labels()        

        return out



    def _predict(self, predictor, examples, task_name):
        """
        make prediction using weights learned by boosting

        @param predictor: trained predictor
        @type predictor: SVMLight
        @param examples: list of examples
        @type examples: list
        @param task_id: task identifier
        @type task_id: int
        
        @return: svm output
        @rtype: list<float>
        """

        (classifiers, weights, task_id, param) = predictor
        

        assert(len(classifiers) == len(weights.tolist()))
        
        out = numpy.zeros(len(examples))

        for i in xrange(len(classifiers)):
            out += weights[i] * numpy.array(self._predict_weak(classifiers[i], examples, task_id, param))        


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
    param.kernel = "WeightedDegreeRBFKernel" #"WeightedDegreeStringKernel"#"PolyKernel" 
    param.wdk_degree = 2
    param.cost = 1.0
    param.transform = 0.2
    param.base_similarity = 1.0
    param.taxonomy = multi_split_set.taxonomy
    param.id = 666
    
    flags= {}
    #flags["boosting"] = "ones"
    #flags["boosting"] = "L1"
    flags["boosting"] = "L2"
    #flags["boosting"] = "L2_reg"
    flags["signum"] = False
    flags["normalize_cost"] = True
    flags["all_positions"] = False
    
    flags["wdk_rbf_on"] = False
    
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

