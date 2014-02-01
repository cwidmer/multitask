#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 12.03.2009
@author: Christian Widmer
@summary: Runs experiments using the SQLObject-based expenv framework 

usage:
expenv_runner new <mss_id> <method> <comment>
expenv_runner run <run_id>
expenv_runner complete <experiment_id>
expenv_runner clean <experiment_id>
"""

#import std packages
import sys
import getopt
import time
import os
import numpy

#import extra packages
from pythongrid import KybJob, process_jobs

#import custom packages
import expenv
from expenv import Method, MultiSplitSet, MultiSourceExperiment, Taxonomy
from expenv import ParameterSvm, ParameterMultiSvm #, ParameterLog 
from expenv import execute_run, create_tables

from task_similarities import fetch_gammas, dataset_to_hierarchy


#some global variables
target = "auROC" #"auPRC"



def run_all_for_mss(mss_id, comment="", methods=None, cluster=True):

    cluster_flag = ""
    
    if cluster:
        cluster_flag = "-c"
    
    if methods == None:
        #methods = ["method_plain_svm", "method_union_svm", "method_hierarchy_svm", "method_xval_hierarchy_svm", "method_augmented_svm"]
        #methods = ["method_plain_svm", "method_union_svm", "method_hierarchy_svm", "method_augmented_svm", "method_pairwise_multitask"]
        methods = ["method_plain_svm"]
    
    
    for method in methods:
        prefix = "nohup python expenv_runner.py " + cluster_flag + " new "
        suffix = ' "' + comment + '"  2> /dev/null &'
        command = prefix + str(mss_id) + " " + method + suffix
        print command 
        
        os.system(command)
        
        time.sleep(4)


def run_all(mss_ids, comment="", methods=None, cluster=True):

    for mss_id in mss_ids:
        run_all_for_mss(mss_id, comment, methods, cluster)



def run_multi_example(dataset_idx, mymethod, comment):
    """
    sets up and runs experiment
    """

    
    #######################################
    # fix parameters
    #######################################

    flags= {}

    # general
    flags["normalize_cost"] = True #False
    flags["epsilon"] = 0.03
    flags["cache_size"] = 500
    
    # Boosting
    #flags["boosting"] = "ones"
    #flags["boosting"] = "L1"
    #flags["boosting"] = "L2"
    flags["boosting"] = "L2_reg"
    #flags["use_all_nodes"] = False
    flags["signum"] = False
    #flags["all_positions"] = True

    
    # MKL
    #flags["solver_type"] = "ST_DIRECT" #ST_CPLEX #ST_GLPK) #ST_DIRECT) #ST_NEWTON)
    #flags["normalize_trace"] = True
    #flags["interleaved"] = True
    #flags["mkl_q"] = 0
    
    #WDK_RBF
    flags["wdk_rbf_on"] = False
    
    # define parameter search space [float(numpy.power(10, 3.58))] #
    costs = [float(c) for c in numpy.exp(numpy.linspace(numpy.log(1000), numpy.log(100000), 8))]
    #costs = [float(c) for c in numpy.exp(numpy.linspace(numpy.log(float(numpy.power(10, 3))), numpy.log(10000), 4))]
    #costs =  [float(c) for c in numpy.exp(numpy.linspace(numpy.log(0.01), numpy.log(1000), 8))] 
    #[float(c) for c in numpy.exp(numpy.linspace(numpy.log(10), numpy.log(2000), 10))]
    costs.reverse()
    
    
    degrees = [1,2,3,4,5] #[1, 5, 10, 15, 20, 22]
    #print "WARNING: Degree is ONE"
    
    base_similarities = [200] #[float(c) for c in numpy.exp(numpy.linspace(numpy.log(1), numpy.log(1000), 8))]
    #base_similarities = [float(c) for c in numpy.linspace(1, 5000, 6)] #[1]
    #transform_params =  [float(c) for c in numpy.linspace(1, 10000, 6)] #[1] #1.5, 2.0, 2.5, 3.0] #, 3.5, 4.0, 4.5, 5.0]
    #transform_params = [float(c) for c in numpy.linspace(0.01, 0.99, 6)]
    transform_params = [0.99]
    
    generation_parameters = locals()
    
    
    #######################################
    # create experiment
    #######################################    
    
    # select dataset
    multi_split_set = MultiSplitSet.get(dataset_idx)


    dataset_name = multi_split_set.description

    print "method:", mymethod
    print "dataset:", dataset_name
    print "multi split set id:", dataset_idx

    experiment_description = dataset_name + " (" + mymethod + ") " + comment
    
    
    # allow different features/kernel types
    feature_type = multi_split_set.feature_type
    
    if feature_type == "string":
        kernel_type = "WeightedDegreeStringKernel"
    else:
        kernel_type = "PolyKernel"
    
    
    kernel_type = "WeightedDegreeRBFKernel"
    
    
    # create experiment
    experiment = MultiSourceExperiment(split_set = multi_split_set, 
                                       description = experiment_description, 
                                       method_name = mymethod,
                                       meta_data = generation_parameters)
    
    print "experiment id:", experiment.id
    

    
    #######################################
    # create runs
    #######################################    
    
    
    if multi_split_set.taxonomy==None:
        print "WARNING: NO taxonomy set, generating one for dataset " + dataset_name
        taxonomy = dataset_to_hierarchy(dataset_name)
    else:
        taxonomy = multi_split_set.taxonomy
        
    
    for cost in costs:
        for degree in degrees:
            for base in base_similarities:
                for transform in transform_params:

                    param = ParameterMultiSvm(cost=cost, 
                                              wdk_degree=degree, 
                                              base_similarity=base, 
                                              transform=transform, 
                                              taxonomy=taxonomy,
                                              kernel=kernel_type,
                                              flags=flags)

                    print param

                    Method(module_name=mymethod, param=param, experiment=experiment)
    

    # skip model selection if we only have one model
    if len(experiment.methods) > 1:
    
        # create evaluation runs based on splits and methods
        run_ids = [run.id for run in experiment.create_eval_runs()]
    
        # execute runs
        execute_runs(run_ids)


    # finally perform model selection and retrain
    select_best_and_test(experiment, target)
    #experiment.select_best_method(target)

    return experiment.id


def complete_experiment(experiment_id, mem, local, threads, force_rerun=False):
    """
    completes runs for experiment with id experiment_id
    """

    print "Warning: Overwriting assessment for experiment_id", experiment_id

    experiment = expenv.Experiment.get(experiment_id)

    # create evaluation runs based on splits and methods
    # run_ids = [run.id for run in experiment.eval_runs]
    if force_rerun:
        run_ids = [run.id for run in experiment.eval_runs]
    else:
        run_ids = [run.id for run in experiment.eval_runs if not run.assessment or not run.assessment_test]

    # execute runs
    execute_runs(run_ids, mem, local, threads)

    # finally perform model selection and retrain
    #experiment.select_best_method(target)
    select_best_and_test(experiment, target, mem, local, threads)


def select_best_retrain_test(experiment, target):
    """
    select best set of parameters from evaluation runs
    """

    # we select best method 
    experiment.select_best_method(target)

    # create test run
    test_run = experiment.create_test_run()
    print "test run id:", test_run.id
    execute_runs([test_run.id])
    
    
    print "##############################"
    print ""
    print "final assessment:"
    print test_run.assessment
    print ""
    print "Experiment", experiment.id, "done."


def select_best_and_test(experiment, target, mem, local, threads):
    """
    select best set of parameters from evaluation runs
    """

    # we select best method 
    experiment.select_best_method(target)

    # create test run
    test_run = experiment.create_test_run()

    # set flag indicating that this is test run
    flags = test_run.method.param.flags
    flags["is_test_run"] = True
    test_run.method.param.flags = flags

    # run
    print "test run id:", test_run.id
    execute_runs([test_run.id], mem, local, threads)
    
    
    print "##############################"
    print ""
    print "final assessment:"
    print test_run.assessment
    print ""
    print "Experiment", experiment.id, "done."


def re_run_run(run_id):
    """
    executes individual run
    """

    print "Warning: Overwriting assessment for run_id", run_id
       
    execute_run(run_id)


def execute_runs(run_ids, mem, local, threads):
    """
    takes a list of run ids and computes them
    """

    print "created", len(run_ids), " runs: ", run_ids
        
    # use pythongrid  
    jobs = []
    
    for run_id in run_ids:
        job = KybJob(expenv.execute_run, [run_id])
        job.h_vmem = mem
        jobs.append(job)
   
    #global local
    print "local", local
    print "maxNumThreads", threads
    
    finished_jobs = process_jobs(jobs, local=local, maxNumThreads=threads)

    MultiSourceExperiment._connection.expireAll()

    return finished_jobs



def main():
    """
    delegates work
    """

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hct:m:f", ["help", "cluster", "threads=", "mem=", "force"])

    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)


    # set default values
    local = True
    threads = 1
    mem = "1G"
    force_rerun = False

    print opts
    
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)

        if o in ("-c", "--cluster"):
            print "cluster flag set"
            local = False

        if o in ("-t", "--threads"):
            threads = int(a)

        if o in ("-m", "--mem"):
            mem = a

        if o in ("-f", "--force"):
            force_rerun = True



    if len(args) == 0:
        print "need at least one argument"

    elif args[0]=="init":
        create_tables()
        
    elif args[0]=="new" and len(args)==4:
        dataset_idx = int(args[1])
        mymethod = args[2]
        comment = args[3]
        run_multi_example(dataset_idx, mymethod, comment)

    elif args[0]=="run" and len(args)==2:
        run_id = int(args[1])
        re_run_run(run_id)

    elif args[0]=="complete" and len(args)==2:
        experiment_id = int(args[1])
        complete_experiment(experiment_id, mem, local, threads, force_rerun)

    elif args[0]=="clean" and len(args)==2:
        experiment_id = int(args[1])
        experiment = expenv.Experiment.get(experiment_id)
        experiment.clean_up()
  
        
if __name__ == "__main__":

    main()
    
