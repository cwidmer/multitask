#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010 Christian Widmer
# Copyright (C) 2010 Max-Planck-Society

"""
Created on 16.03.2010
@author: Christian Widmer
@summary: Implementation of MKL MTL with CombinedKernel
"""


import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight
from shogun.Kernel import MultitaskKernelNormalizer, KernelNormalizerToMultitaskKernelNormalizer, Pairii, PairiiVec
from shogun.Features import CombinedFeatures
from base_method import MultiMethod, PreparedMultitaskData

import pdb
import helper
from helper import SequencesHandler, split_data 


import numpy
import shogun
import cvxmod

from boosting import solve_boosting



class Method(MultiMethod):



    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        """

        # split for training weak_learners and boosting
        (train_weak, train_boosting) = split_data(train_data, 4)
          
        # merge data sets
        data = PreparedMultitaskData(train_weak, shuffle=True)
        
        # create shogun label
        lab = shogun_factory.create_labels(data.labels)
        


        ##################################################
        # define pockets
        ##################################################
        
        pockets = [0]*9
        
        pockets[0] = [1, 5, 6, 7, 8, 31, 32, 33, 34]
        pockets[1] = [1, 2, 3, 4, 6, 7, 8, 9, 11, 21, 31]
        pockets[2] = [11, 20, 21, 22, 29, 31]
        pockets[3] = [8, 30, 31, 32]
        pockets[4] = [10, 11, 30]
        pockets[5] = [10, 11, 12, 13, 20, 29]
        pockets[6] = [10, 12, 20, 22, 26, 27, 28, 29]
        pockets[7] = [12, 14, 15, 26]
        pockets[8] = [13, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26]
        
        pockets = []
        for i in xrange(35):
            pockets.append([i])


        #new_pockets = []
        
        # merge neighboring pockets
        #for i in range(8):
        #    new_pockets.append(list(set(pockets[i]).union(set(pockets[i+1]))))
            
        #pockets = new_pockets
        
        
        ########################################################
        print "creating a kernel:"
        ########################################################
        
        
        # init seq handler 
        pseudoseqs = SequencesHandler()

        
        classifiers = []


        for pocket in pockets:

            print "creating normalizer"
            #import pdb
            #pdb.set_trace()
            
            normalizer = MultitaskKernelNormalizer(data.task_vector_nums)
            
            print "processing pocket", pocket

            # set similarity
            for task_name_lhs in data.get_task_names():
                for task_name_rhs in data.get_task_names():
                    
                    similarity = 0.0
                    
                    for pseudo_seq_pos in pocket:
                        similarity += float(pseudoseqs.get_similarity(task_name_lhs, task_name_rhs, pseudo_seq_pos-1))
                    
                    # normalize
                    similarity = similarity / float(len(pocket))
                    
                    print "pocket %s (%s, %s) = %f" % (str(pocket), task_name_lhs, task_name_rhs, similarity)
                    
                    normalizer.set_task_similarity(data.name_to_id(task_name_lhs), data.name_to_id(task_name_rhs), similarity)
               

            print "creating empty kernel"
            kernel = shogun_factory.create_kernel(data.examples, param)
            
            print "setting normalizer"
            kernel.set_normalizer(normalizer)

            print "training SVM for pocket", pocket
            svm = self._train_single_svm(param, kernel, lab)

            classifiers.append(svm)
        
        
        print "done obtaining weak learners"
            
        
        # save additional info
        #self.additional_information["svm_objective"] = svm.get_objective()
        #self.additional_information["svm num sv"] = svm.get_num_support_vectors()
        #self.additional_information["post_weights"] = combined_kernel.get_subkernel_weights()
        
        #print self.additional_information 
        


        ##################################################
        # combine weak learners for each task
        ##################################################
        
        
        # set constants
        
        some = 0.9
        import cvxmod
        
        
        # wrap up predictors
        svms = {}
            
        # use a reference to the same svm several times
        for task_name in train_boosting.keys():
            
            instances = train_boosting[task_name]
            
            N = len(instances)
            F = len(pockets)
            
            examples = [inst.example for inst in instances]
            labels = [inst.label for inst in instances]
            
            # dim = (F x N)
            out = cvxmod.zeros((N,F))
            
            for i in xrange(F):
                svm = classifiers[i]
                tmp_out = self._predict_weak(svm, examples, data.name_to_id(task_name))

                out[:,i] = numpy.sign(tmp_out)
                #out[:,i] = tmp_out
            

            #TODO: fix
            helper.save("/tmp/out_sparse", (out,labels))
            pdb.set_trace()
            
            weights = solve_boosting(out, labels, some, solver="mosek")
            
            
            
            svms[task_name] = (data.name_to_id(task_name), svm)

        
        return svms


            


    def _train_single_svm(self, param, kernel, lab):
    

    
        kernel.set_cache_size(500)
        #lab = shogun_factory.create_labels(data.labels) 
        svm = SVMLight(param.cost, kernel, lab)

        # set up SVM
        num_threads = 8
        svm.io.enable_progress()
        svm.io.set_loglevel(shogun.Classifier.MSG_DEBUG)
        
        svm.parallel.set_num_threads(num_threads)
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)
            
        # normalize cost
        #norm_c_pos = param.cost / float(len([l for l in data.labels if l==1]))
        #norm_c_neg = param.cost / float(len([l for l in data.labels if l==-1]))

        #svm.set_C(norm_c_neg, norm_c_pos)
        
        
        # start training
        svm.train()

        return svm
    

    def _predict_weak(self, svm, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor (task_id, num_nodes, combined_kernel, predictor)
        @type predictor: tuple<int, int, CombinedKernel, SVM>
        @param examples: list of examples
        @type examples: list<object>
        @param task_name: task name
        @type task_name: str
        """

        # shogun data
        feat = shogun_factory.create_features(examples)
        
        # fetch kernel normalizer
        normalizer = svm.get_kernel().get_normalizer()
            
        # cast using dedicated SWIG-helper function
        normalizer = KernelNormalizerToMultitaskKernelNormalizer(normalizer)
            
        # set task vector
        normalizer.set_task_vector_rhs([task_id]*len(examples))

        # predict
        out = svm.classify(feat).get_labels()
        
        return out
    

    def _predict(self, predictor, examples, task_name):
        """
        make prediction on examples using trained predictor

        @param predictor: trained predictor (task_id, num_nodes, combined_kernel, predictor)
        @type predictor: tuple<int, int, CombinedKernel, SVM>
        @param examples: list of examples
        @type examples: list<object>
        @param task_name: task name
        @type task_name: str
        """

        (task_id, svm) = predictor

        # shogun data
        base_feat = shogun_factory.create_features(examples)
        
        # construct combined kernel
        feat = CombinedFeatures()
        
        for i in xrange(combined_kernel.get_num_subkernels()):
            feat.append_feature_obj(base_feat)

            # fetch kernel normalizer
            normalizer = combined_kernel.get_kernel(i).get_normalizer()
            
            # cast using dedicated SWIG-helper function
            normalizer = KernelNormalizerToMultitaskKernelNormalizer(normalizer)
            
            # set task vector
            normalizer.set_task_vector_rhs([task_id]*len(examples))


        combined_kernel = svm.get_kernel()
        combined_kernel.init(combined_kernel.get_lhs(), feat)
        
        # predict
        out = svm.classify().get_labels()
        
        return out




        


def solve_boosting(out, labels, nu, solver):
    '''
    solve boosting formulation used by gelher and novozin
    
    @param out: matrix (N,F) of predictions (for each f_i) for all examples
    @param y: vector (N,1) label for each example 
    @param p: regularization constant
    '''
    
    
    
    N = out.size[0]
    F = out.size[1]
    
    assert(N==len(labels))
    
    
    norm_fact = 1.0 / (nu * float(N))
    
    print norm_fact
    
    label_matrix = cvxmod.zeros((N,N))
    
    # avoid point-wise product
    for i in xrange(N):
        label_matrix[i,i] = labels[i] 
    
    
    #### parameters
    
    f = cvxmod.param("f", N, F)
    
    y = cvxmod.param("y", N, N, symm=True)
    
    norm = cvxmod.param("norm", 1) 
    
    #### varibales
    
    # rho
    rho = cvxmod.optvar("rho", 1)
    
    # dim = (N x 1)
    chi = cvxmod.optvar("chi", N)
    
    # dim = (F x 1)
    beta = cvxmod.optvar("beta", F)
    
    
    #objective = -rho + cvxmod.sum(chi) * norm_fact + square(norm2(beta)) 
    objective = -rho + cvxmod.sum(chi) * norm_fact
    
    print objective
    
    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))
    
    
    # create contraint for probability simplex
    #p.constr.append(beta |cvxmod.In| probsimp(F))
    p.constr.append(cvxmod.sum(beta)==1.0)
    #p.constr.append(square(norm2(beta)) <= 1.0)
    p.constr.append(beta >= 0.0)
    
    
    #    y       f     beta          y    f*beta      y*f*beta
    # (N x N) (N x F) (F x 1) --> (N x N) (N x 1) --> (N x 1)
    p.constr.append(y * (f * beta) + chi >= rho)
    
    
    ###### set values
    f.value = out
    y.value = label_matrix
    norm.value = norm_fact 
    
    p.solve(lpsolver=solver)
    

    weights = numpy.array(cvxmod.value(beta))
    
    #print weights
    
    cvxmod.printval(chi)
    cvxmod.printval(beta)
    cvxmod.printval(rho)
    

    return p
        

def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = -1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    #multi_split_set = MultiSplitSet.get(387)
    multi_split_set = MultiSplitSet.get(386)

    #dataset_name = multi_split_set.description

    
    # create mock param object by freezable struct
    param = Options()
    param.kernel = "WeightedDegreeStringKernel"#"PolyKernel" 
    param.wdk_degree = 1
    param.cost = 1
    param.transform = 2 #2.0
    param.taxonomy = multi_split_set.taxonomy
    param.id = 666
    
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

    
