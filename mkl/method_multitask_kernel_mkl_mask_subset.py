#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010 Christian Widmer
# Copyright (C) 2010 Max-Planck-Society

"""
Created on 30.01.2010
@author: Christian Widmer
@summary: Implementation of MKL MTL with CombinedKernel
"""


import shogun_factory_new as shogun_factory

from shogun.Classifier import SVMLight, MKLClassification, ST_CPLEX, ST_NEWTON, ST_DIRECT, ST_GLPK
from shogun.Kernel import MultitaskKernelMaskNormalizer, KernelNormalizerToMultitaskKernelMaskNormalizer, CombinedKernel
from shogun.Features import CombinedFeatures
from base_method import MultiMethod, PreparedMultitaskData

from helper import power_set

import shogun

class Method(MultiMethod):



    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: Data relevant to SVM training
        @type train_data: dict<str, list<instances> >
        @param param: Parameters for the training procedure
        @type param: ParameterSvm
        """

        import numpy
        numpy.random.seed(666)

          
        # merge data sets
        data = PreparedMultitaskData(train_data, shuffle=True)

                
        # create shogun label
        lab = shogun_factory.create_labels(data.labels)


        # assemble combined kernel
        combined_kernel = CombinedKernel()
        combined_kernel.io.set_loglevel(shogun.Kernel.MSG_DEBUG)    
        # set kernel cache
        if param.flags.has_key("cache_size"):
            combined_kernel.set_cache_size(param.flags["cache_size"])
        

        # create features
        base_features = shogun_factory.create_features(data.examples, param)
        
        combined_features = CombinedFeatures()

        
        ########################################################
        print "creating a masked kernel for possible subset:"
        ########################################################

        
        power_set_tasks = power_set(data.get_task_ids()) 
        
        
        for active_task_ids in power_set_tasks:
            
            
            print "masking all entries other than:", active_task_ids
            
        
            # create mask-based normalizer
            normalizer = MultitaskKernelMaskNormalizer(data.task_vector_nums, data.task_vector_nums, active_task_ids)
            
            # normalize trace
            if param.flags.has_key("normalize_trace") and param.flags["normalize_trace"]:
                norm_factor = len(data.get_task_ids()) / len(active_task_ids)
                normalizer.set_normalization_constant(norm_factor)               

            
            kernel = shogun_factory.create_empty_kernel(param)
            kernel.set_normalizer(normalizer)
            
            
            # append current kernel to CombinedKernel
            combined_kernel.append_kernel(kernel)
        
            # append features
            combined_features.append_feature_obj(base_features)

            print "------"
        

        combined_kernel.init(combined_features, combined_features)
        
                
        #combined_kernel.precompute_subkernels()
        
        self.additional_information["weights before trainng"] = combined_kernel.get_subkernel_weights()        
        print "subkernel weights:", combined_kernel.get_subkernel_weights()

        svm = None
                
        
        print "using MKL:", (param.flags["mkl_q"] >= 1.0)
        
        if param.flags["mkl_q"] >= 1.0:
            
            svm = MKLClassification()
            
            svm.set_mkl_norm(param.flags["mkl_q"])
            
            # set interleaved optimization
            if param.flags.has_key("interleaved"):
                svm.set_interleaved_optimization_enabled(param.flags["interleaved"])
            
            # set solver type
            if param.flags.has_key("solver_type") and param.flags["solver_type"]:
                if param.flags["solver_type"] == "ST_CPLEX":
                    svm.set_solver_type(ST_CPLEX)
                if param.flags["solver_type"] == "ST_DIRECT":
                    svm.set_solver_type(ST_DIRECT)
                if param.flags["solver_type"] == "ST_NEWTON":
                    svm.set_solver_type(ST_NEWTON)
                if param.flags["solver_type"] == "ST_GLPK":
                    svm.set_solver_type(ST_GLPK)
                    
                    
            svm.set_kernel(combined_kernel)
            svm.set_labels(lab)
            
        else:
            
            svm = SVMLight(param.cost, combined_kernel, lab)


        # optimization settings
        num_threads = 4
        svm.parallel.set_num_threads(num_threads)
        
        if param.flags.has_key("epsilon"):
            svm.set_epsilon(param.flags["epsilon"])
        
        
        # enable output        
        svm.io.enable_progress()
        svm.io.set_loglevel(shogun.Classifier.MSG_DEBUG)
        
        
        # disable unsupported optimizations (due to special normalizer)
        svm.set_linadd_enabled(False)
        svm.set_batch_computation_enabled(False)
        
        
        # set cost
        if param.flags["normalize_cost"]:
            
            norm_c_pos = param.cost / float(len([l for l in data.labels if l==1]))
            norm_c_neg = param.cost / float(len([l for l in data.labels if l==-1]))
            svm.set_C(norm_c_neg, norm_c_pos)
            
        else:
            
            svm.set_C(param.cost, param.cost)
        
        svm.train()
    
    
        # prepare mapping
        weight_map = {}
        weights = combined_kernel.get_subkernel_weights()
        for (i, pset) in enumerate(power_set_tasks):
            print pset
            subset_str = str([data.id_to_name(task_idx) for task_idx in pset])
            weight_map[subset_str] = weights[i]
        
        # store additional info
        self.additional_information["svm objective"] = svm.get_objective()
        self.additional_information["weight_map"] = weight_map

        
        # wrap up predictors
        svms = {}
            
        # use a reference to the same svm several times
        for task_name in train_data.keys():
            svms[task_name] = (data.name_to_id(task_name), len(power_set_tasks), combined_kernel, svm, param)

        
        return svms
    
    

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

        (task_id, num_nodes, combined_kernel, svm, param) = predictor

        # shogun data
        base_feat = shogun_factory.create_features(examples, param)
                
        # construct combined kernel
        feat = CombinedFeatures()
        
        for i in xrange(num_nodes):
            feat.append_feature_obj(base_feat)

            # fetch kernel normalizer
            normalizer = combined_kernel.get_kernel(i).get_normalizer()
            
            # cast using dedicated SWIG-helper function
            normalizer = KernelNormalizerToMultitaskKernelMaskNormalizer(normalizer)
            
            # set task vector
            normalizer.set_task_vector_rhs([task_id]*len(examples))


        combined_kernel.init(combined_kernel.get_lhs(), feat)
        
        # predict
        out = svm.classify().get_labels()

        # predict
        #out = svm.classify(feat).get_labels()
        
        
        return out


        

def main():
    
    
    print "starting debugging:"

    SPLIT_POINTER = -1

    from expenv import MultiSplitSet
    from helper import Options 
    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(399)

    #dataset_name = multi_split_set.description
    flags = {}
    flags["normalize_cost"] = False
    flags["epsilon"] = 0.05
    flags["cache_size"] = 7
    #flags["solver_type"] = "ST_DIRECT" #ST_CPLEX #ST_GLPK) #ST_DIRECT) #ST_NEWTON)
    flags["normalize_trace"] = True
    flags["interleaved"] = True
    
    
    #create mock param object by freezable struct
    param = Options()
    param.kernel = "WeightedDegreeStringKernel"
    param.wdk_degree = 1
    param.cost = 1
    param.transform = 1 #2.0
    param.taxonomy = multi_split_set.taxonomy
    param.id = 666
    
    
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
    


def load_saved_similarity():
    """
    load saved similarity file
    """

    import pandas
    fn = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/results/mhc/icml_sim.csv"
    df = pandas.io.parsers.parseCSV(fn)

    return df


def mkl_similarity_matrix(active_set, exp_id=4135):
    """
    convertes weight map to similarity matrix

    to be used in paper
    """

    import pandas
    import expenv
    import pylab
    import numpy

    print "new"

    e = expenv.Experiment.get(exp_id)
    print e.test_run.additional_information.keys()
    wm = e.test_run.additional_information["weight_map"]

    data = numpy.zeros((len(active_set), len(active_set))) 
    df = pandas.DataFrame(data=data, columns=active_set, index=active_set)

    print "num keys", len(wm.keys())

    for lhs in active_set:
        for rhs in active_set:
            for key in wm.keys():
                if key.find(lhs) != -1 and key.find(rhs) != -1:
                    df[lhs][rhs] += wm[key]
        # normalize
        #df[lhs] = df[lhs] / df[lhs].sum()


    print df.values

    print df.index

    pylab.imshow(df.values, interpolation="nearest")
    pylab.show()

    return df



def load_distances(active_set):
    """
    load distances from file
    """

    import numpy
    import pandas

    f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/All_PseudoSeq_Hamming.txt")
    #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_PseudoSeq_BlosumEnc_euklid.txt")
    #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_RAxML.txt")
    
    num_lines = int(f.readline().strip())
    task_distances = numpy.zeros((num_lines, num_lines))
    names = []

    for (i, line) in enumerate(f):
        tokens = line.strip().split("\t")
        name = str(tokens[0])
        names.append(name)

        entry = numpy.array([v for (j,v) in enumerate(tokens) if j!=0])
        assert len(entry)==num_lines, "len_entry %i, num_lines %i" % (len(entry), num_lines)
        task_distances[i,:] = entry
        
        
    df = pandas.DataFrame(index=names, columns=names, data=task_distances)
   
    # select subset of variables
    df = df.reindex(index=active_set, columns=active_set)

    return df


def compute_correlation(exp_id):
    """
    compute correlations between similarity measures
    """

    import scipy.stats as ss

    new_order = ['A_6901', 'A_0202', 'A_0203', 'A_0201', 'A_2301', 'A_2402', 'A_2403']

    df_mkl = load_saved_similarity()
    df_ext = load_distances(df_mkl.columns)

    df_mkl = df_mkl.reindex(columns=new_order, index=new_order)
    df_ext = df_ext.max().max() - df_ext.reindex(columns=new_order, index=new_order)

    v1 = df_mkl.values.flatten()
    v2 = df_ext.values.flatten()

    print "spearman", ss.spearmanr(v1, v2)
    print "pearson", ss.pearsonr(v1, v2)
    print df_mkl.corrwith(df_ext)

    print "detailed spearman"
    for col in df_mkl.index:
        print col, ss.spearmanr(df_mkl[col].values, df_ext[col].values)

    return df_ext, df_mkl


if __name__ == "__main__":
    main()

    
