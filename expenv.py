#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009-2011 Christian Widmer
# Copyright (C) 2009-2011 Max-Planck-Society

"""
Created on 09.03.2009
@author: Christian Widmer
@summary: SQLObject-based expenv framework 


This module is meant to provide a general, easy-to-use and well-designed 
experimental framework, that helps to keep track of which data and methods 
led to which performance. It relies heavily on the package SQLObject, which
is used for database persistence. Expenv is meant to be used as a library, 
which can easily be extended by inheritance. The stand-alone interface 
provides an easy way to create and drop the tables defined within this module.


accepted argument values:

create\twill create all tables in module
drop\twill drop all tables in module
test\twill run simple test example (careful, tables will be dropped!!)
"""


import dbconnect

#import std packages
import time
import os
from random import choice
from collections import defaultdict

#import extra packages
import numpy
from sqlobject import SQLObject, StringCol, IntCol, FloatCol, BoolCol, TimestampCol, PickleCol, ForeignKey, RelatedJoin, MultipleJoin, SingleJoin
from sqlobject.inheritance import InheritableSQLObject


#import custom packages
from helper import split_list, Options
import helper



class Dataset(SQLObject):
    """
    dataset and some meta data
    """

    organism = StringCol(default="")
    comment = StringCol(default="")
    version = StringCol(default="")
    signal = StringCol(default="")
    
    instances = MultipleJoin("Instance")
    
    
    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        for instance in self.instances: 
            instance.clean_up()

        self.destroySelf()
    
    
    def create_split_set(self, num_splits, size_subset=0, size_testset=0, random=True):
        """
        Splits data in num_splits parts and creates
        the corresponding Split and SplitSet objects.
        For the last split in the split set, the is_test_set
        flag will be set.
        
        Performs a random permutation if the random flag is set.
        
        @param num_splits: number of desired splits (should be at least three)
        @type num_splits: int
        @param size_subset: if set to greater than zero only a subset of the dataset is used. 
        @type size_subset: int
        @param random: flag to determine, whether order of data is kept
        @type random: bool
        """
        

        #instances = self.instances
        instances = list(Instance.selectBy(dataset=self))
        
        if size_subset == 0:
            #we use all instances in dataset
            size_subset = len(instances)

        
        instances_idx = range(len(instances))
        
        #permute list of idx if random flag is set
        if random:
            numpy.random.shuffle(instances_idx)
        
        #only consider subset of dataset
        instances_idx = instances_idx[0:size_subset]


        print "total number of examples:", len(instances_idx)


        split_idx_lists = []
        
        if size_testset!=0:

            
            
            #append test set
            test_split_idx = instances_idx[0:size_testset] 
            split_idx_lists.append(test_split_idx)
            
            #split remaning instances equally
            instances_idx = instances_idx[size_testset:]
            split_idx_lists.extend(split_list(instances_idx, num_splits-1))
                        
            
        else:            
            #split list
            split_idx_lists = split_list(instances_idx, num_splits)
    
    
        try:
            assert(sum([len(s) for s in split_idx_lists])==size_subset)
        except AssertionError, detail:
            print detail
            print "=============================="
            print "sum split lengths:", sum([len(s) for s in split_idx_lists])
            print "size_subset:", size_subset
            raise AssertionError()
        
        
        #create new split set
        split_set = SplitSet(dataset=self, num_instances=size_subset)
        
        
        for (i,split_idx_list) in enumerate(split_idx_lists):
            
            print "adding ", str(i), " num_instances:", len(split_idx_list)

            
            #set testset flag for first split            
            split = Split(is_test_set=(i==0), split_set = split_set, num=i, num_instances=len(split_idx_list))
            
            #create split
            for idx in split_idx_list:
                
                instances[idx].split = split
                
                #TODO revert to many-to-many
                #split.addInstance(instances[idx])
            
            assert(len(split.instances)==len(split_idx_list))
            
        return split_set



class Instance(InheritableSQLObject):
    """
    A class to hold a labeled example.
    
    This class handles arbitrary feature types through a PickleCol field.
    """

    #split = RelatedJoin("Split")
    split = ForeignKey("Split", notNone=False, default=None)
    dataset = ForeignKey("Dataset")
    label = FloatCol(default=0.0)
    example = PickleCol(default=None)


    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        self.destroySelf()


class SplitSet(SQLObject):
    """
    A set of splits usually used in experiments
    """

    dataset = ForeignKey("Dataset")
    splits = MultipleJoin("Split")
    num_instances = IntCol(default=0)


    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        for split in self.splits:
            split.clean_up()

        self.dataset.clean_up()

        self.destroySelf()

    
    
    def get_task_id(self):
        """
        returns task identifier as string
        """

        return self.dataset.organism


    def get_test_set(self):
        """
        helper method to obtain test set
        """

        return self.splits[self.get_test_set_id()]


    def get_test_set_id(self):
        """
        helper method to obtain task id of test set
        """

        test_set_ids = [id for (id, split) in enumerate(self.splits) if split.is_test_set]

        # make sure that there is only one test set
        assert len(test_set_ids) == 1, "more than one test set"
        

        return test_set_ids[0]
            

    def get_eval_data(self, x_val_idx, random=False):
        """
        helper to fetch eval data
        """

        eval_split = None
        
        #return evaluation data 
        if x_val_idx!=-1:
            
            for split in self.splits:

                #exclude test non-eval data
                if not split.is_test_set and split.num==x_val_idx:
                                
                    print "getting eval data. split.num:", split.num, ", split.id:", split.id
                    #print [instance.id for instance in split.instances][0:5]
                    
                    eval_split = split
                    break
           
            assert(eval_split.is_test_set==False)
            
        elif x_val_idx==-1:
            
            for split in self.splits:
            
                if split.is_test_set:
                    
                    #print [instance.id for instance in split.instances][0:10]
                    print "getting test data. split.num:", split.num, ", split.id:", split.id
                    
                    eval_split = split
                    break
        
        
        # read from data file
        #if True == False:
        if split.data_file != None:
            
            instances = load_data_from_file(split.data_file)
        
        else:
        
            #optimization to only use one query
            instances = list(Instance.selectBy(split=eval_split).orderBy(["id"]))
    
            #sometimes the query fails, so we retry once:
            if len(instances)==0:
                #wait one second
                time.sleep(1)
                print "query failed, retrying once"
                instances = list(Instance.selectBy(split=eval_split).orderBy(["id"]))
            
        
        #randomly shuffle because they are sorted by id per default
        #if random:
        #    numpy.random.shuffle(instances)
        
        return instances
        
        
    def get_train_data(self, x_val_idx, random=False):
        """
        helper to fetch train data
        """
        
        data = []

        #concatenate datasets
        for split in self.splits:
            
            #exclude test and evaluation data
            if not split.is_test_set and split.num!=x_val_idx:
                
                print "getting train data. split.num:", split.num, ", split.id:", split.id


                # read from data in file
                if split.data_file != None:
                #if True == False:
                    
                    instances = load_data_from_file(split.data_file)
                    
                
                else:

                    #optimization to only use one query
                    instances = list(Instance.selectBy(split=split).orderBy(["id"]))
    
                    print "number of instances:", len(instances)
                    
                    #sometimes the query fails, so we retry once:
                    if len(instances)==0:
                        print "query failed, retrying once"
                        #wait one second
                        time.sleep(1)
                        instances = list(Instance.selectBy(split=split).orderBy(["id"]))
 
        
                #randomly shuffle because they are sorted by id per default
                #if random:
                #    numpy.random.shuffle(instances)
                
                data.extend(instances)
        
        #print "============================"
        
        return data


    def check_sanity(self):
        """
        make sure intersection between splits and between train&eval sets is empty
        """
        
        splits = self.splits
        
        num_splits = len(splits)
        
        for i in xrange(num_splits):
            
            for j in xrange(i+1, num_splits):
                
                idx_i = set([inst.id for inst in splits[i].instances])
                idx_j = set([inst.id for inst in splits[j].instances])
                
                assert(len(idx_i.intersection(idx_j))==0)
                

        for i in xrange(1, num_splits):
            
            train_idx = set([inst.id for inst in self.get_train_data(i)])
            eval_idx = set([inst.id for inst in self.get_eval_data(i)])
    
            assert(len(train_idx.intersection(eval_idx))==0)


def load_data_from_file(data_file):
    """
    load data file and return examples
    
    this implicitly defines the file format
    """

    #format: (str(seq_record.seq), 1)

    data = helper.load(data_file)

    instances = []            
    
    for item in data:
        
        # expand tuple
        example, label = item
    
        # create pseudo-object
        instance = Options()
        instance.example = example
        instance.label = label
        instance.freeze()

        instances.append(instance)

    # return subset    
    return instances



class MultiSplitSet(SQLObject):
    """
    For the multi-source scenario, we need to deal
    with several data sources. 
    """
    
    split_sets = RelatedJoin("SplitSet")
    description = StringCol(default="")
    feature_type = StringCol(default="") #possible values: string, real
    taxonomy = ForeignKey("Taxonomy", default=None)
    
    
    #general field to store data structures needed for the creation of the dataset
    generation_data = PickleCol(default=None) 
    
    generation_data_path= "/fml/ag-raetsch/share/projects/multitask/generation_data/"
    generation_file = ""
    
    
    def set_generation_data(self, generation_parameters):
        """
        trouble with db, thus save things on FS for now
        """
                
        self.generation_file = self.generation_data_path + "mss_" + str(self.id) + ".bz2" 
        helper.save(self.generation_file, generation_parameters)
    
    
    def get_generation_data(self):
        """
        trouble with db, thus save things on FS for now
        """
        
        gd = helper.load(self.generation_file)
        
        return gd 
        
        
        
    
    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        # clean up file
        if self.generation_file and os.path.exists(self.generation_file):
            os.remove(self.generation_file)


        for split_set in self.split_sets:
            split_set.clean_up()
            

        self.destroySelf()
    


    def get_eval_data(self, x_val_idx):
        """
        helper to fetch eval data
        returns a generator with elements (task_id, value)
        """

        
        #return multi data as a list
        for split in self.split_sets:

            print "split_id", split.id

            task_id = split.get_task_id()
            print "task_id", task_id 
            
            yield task_id, split.get_eval_data(x_val_idx)
        
        
        
    def get_train_data(self, x_val_idx):
        """
        helper to fetch train data,
        returns a dict{task_id, instances}
        """
        
        multi_data = {}
        
        #return multi data as a list
        for split in self.split_sets:            

            task_id = split.get_task_id()
            print "task_id", task_id 

            multi_data[task_id] = split.get_train_data(x_val_idx)
        
        for (i,instances) in enumerate(multi_data.values()):
            print "multi_split.get_train_data", i, len(instances)
        
        return multi_data
        
        

class Split(SQLObject):
    """
    A split is a subset of a dataset 
    """
    
    instances = MultipleJoin("Instance")
    #instances = RelatedJoin("Instance")
    split_set = ForeignKey("SplitSet")    
    is_test_set = BoolCol(default=False)
    num = IntCol(default=0)
    
    num_instances = IntCol(default=0)
    data_file = StringCol(default=None)

    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        self.destroySelf()



class Run(SQLObject):
    """
    Run is training and evaluation
    """

    experiment = ForeignKey("Experiment")
    assessment = ForeignKey("Assessment", notNone=False, default=None)
    assessment_test = ForeignKey("Assessment", notNone=False, default=None)
    
    method = ForeignKey("Method")
    
    additional_information = PickleCol(default=None)
    
    
    predictor_prefix = "/fml/ag-raetsch/share/projects/multitask/predictors/run_" 
    
    #important: points to the evaluation split
    x_val_idx = IntCol(default=-1)


    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        # clean up file
        predictor_fn = self.predictor_prefix + str(self.id)

        if os.path.exists(predictor_fn):
            print "deleting predictor file", predictor_fn
            os.remove(predictor_fn)

        if self.assessment:
            self.assessment.clean_up()

        if self.assessment_test:
            self.assessment_test.clean_up()

        self.destroySelf()


    def get_eval_data(self):
        """
        helper to fetch eval data
        """
        
        data = self.experiment.split_set.get_eval_data(self.x_val_idx)
                
        return data
        
        
    def get_train_data(self):
        """
        helper to fetch train data
        """
        
        data = self.experiment.split_set.get_train_data(self.x_val_idx)
        
        return data


    def execute(self):
        """
        train and predict using data and method
        """
        
        #connect framework to method implementation
        method_module = __import__(self.method.module_name)
        method = method_module.Method(self.method.param)
        
        method.train(self.get_train_data())
        
        
        # attach data from training
        self.additional_information = method.additional_information
 
        # check relevant flags
        is_test_run = self.method.param.flags.has_key("is_test_run") and self.method.param.flags["is_test_run"] == True
        save_predictor = self.method.param.flags.has_key("save_predictor") and self.method.param.flags["save_predictor"] == True 
 
        if is_test_run and save_predictor:
            predictor_fn = self.predictor_prefix + str(self.id) + ".gzip"
            print "saving predictor to", predictor_fn
            method.save_predictor(predictor_fn)

        
        self.assessment = method.evaluate(self.get_eval_data())
        
        # evaluate on test data right away
        test_data = self.experiment.split_set.get_eval_data(-1)
        self.assessment_test = method.evaluate(test_data)
        
        
        

    def load_predictor(self):
        """
        loads saved predictor from file system
        """

        predictor = None
        
        try:
            predictor_fn = self.predictor_prefix + str(self.id) + ".gzip"
            predictor =  helper.load(predictor_fn, "gzip")
            
        except Exception, detail:
            
            print "error loading predictor"
            print detail
        
        
        return predictor


    def __str__(self):
        """
        informal string representation
        """
        
        mystr = "run_id:" + str(self.id) + "\n"
        mystr += "Method: " + str(self.method) + "\n"
        mystr += "Info: " + str(self.additional_information) + "\n"
        mystr += "Assessment: " + str(self.assessment)
        
        return mystr



class Method(InheritableSQLObject):
    """
    Holds information about the method
    """
    
    experiment = ForeignKey("Experiment")
    param = ForeignKey("Parameter")
    name = StringCol(default="")
    module_name = StringCol(default="")
    svn_revision = IntCol(default=0)
   
    predictor = PickleCol(default=None)


    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        if self.param:
            self.param.destroySelf()

        self.destroySelf()
    

    def __str__(self):
        """
        informal string representation
        """
    
        mystr = self.module_name + " " + self.name + " (rev " + str(self.svn_revision) + ")\n"
        mystr += "Method parameters: " + str(self.param)
    
        return mystr



class Experiment(InheritableSQLObject):
    """
    Base class for Experiments
    
    An experiment includes ModelSelection and the
    final performance assessment on test data
    """

    name = StringCol(default="")
    description = StringCol(default="")
    method_name = StringCol(default="") #TODO: make this non-redundant
    timestamp = TimestampCol(default=None)
    
    runs = MultipleJoin("Run")
    test_run = ForeignKey("Run", notNone=False, default=None)
    
    methods = MultipleJoin("Method")
    
    #fields for model selection
    best_method = ForeignKey("Method", notNone=False, default=None)
    best_mean_performance = FloatCol(default=0.0)
    best_std_performance = FloatCol(default=0.0)


    # allow the storage of arbitrary meta data for expermiment
    meta_data = PickleCol(default=None)



    def get_eval_runs(self):
        """
        define python-style getter,
        only get runs which are not the test run
        """

        all_runs = list(self.runs)
        test_run = self.test_run

        if test_run and all_runs.count(test_run) > 0:
            all_runs.remove(test_run)

        return all_runs


    def set_eval_runs(self, x):
        """
        define python-style setter
        @param x: eval runs to be set
        @type x: None
        """

        print "eval_runs is read-only"


    # use python style getter to exclude test run from eval_runs
    eval_runs = property(get_eval_runs,
                              set_eval_runs)


    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        # clear up runs
        for run in self.runs:
            run.clean_up()

        # clean up methods
        for method in self.methods:
            method.clean_up()

        self.destroySelf()
  

    def __repr__(self):
        """
        prepare string representation
        """

        ret =  "=========================\n"
        ret += "Exper id:\t" + str(self.id) + "\n"
        ret += "comment:\t" + self.description + "\n"
        ret += "timestamp:\t" + str(self.timestamp) + "\n"
        ret += "number of runs:\t" + str(len(self.runs)) + "\n"
        ret += "Parameters:\t" + str(self.get_parameters()) + "\n"
        
        try:
            (best_method, best_target_eval, best_std_eval, best_target_test, best_std_test) = self.find_best_method("auROC") 
            ret += "Evaluation Performance by auROC: %.4f (%.4f)\n" % (best_target_eval, best_std_eval)
            ret += "Test Performance by auROC: %.4f (%.4f)\n\n" % (best_target_test, best_std_test)
    
            (best_method, best_target_eval, best_std_eval, best_target_test, best_std_test) = self.find_best_method("auPRC") 
            ret += "Evaluation Performance by auPRC: %.4f (%.4f)\n" % (best_target_eval, best_std_eval)
            ret += "Test Performance by auPRC: %.4f (%.4f)\n" % (best_target_test, best_std_test)
        except Exception, detail:
            print "detailed performances not ready"
            print detail
            

        if self.test_run!=None:
            ret += "Method Name:\t" + self.test_run.method.module_name + "\n"
            ret += "Best Parameters:\t" + str(self.test_run.method.param) + "\n"
            ret += "Eval perf:\t" + str(self.best_mean_performance) + "\n"

            if self.test_run.assessment!=None:
                ret += self.test_run.assessment.__repr__()
                
            ret += "Info:\t" + str(self.test_run)

        ret += "=================\n"
        #ret += "Detailed Run Information\n"
        
        #for run in self.eval_runs:
        #    ret += str(run)
            
        #TODO: give dataset statistics
        

        return ret
        


    def get_parameters(self):
        """
        helper method that looks through methods and 
        returns a dictionary with parameter names
        mapped to the list of used parameters
        """

        parameters = defaultdict(set)

        for method in self.methods:
            
            tmp_param = helper.get_member_dict(method.param, helper.get_sqlobject_member_list())
            
            for (key, value) in tmp_param.items():
                try:
                    parameters[key].add(value)
                except:
                    print "skipping non-hashable object", key, value
            
        return parameters 


    def find_best_method(self, target):
        """
        select best eval method, average over runs
        """


        best_method = 0
        best_target_eval = 0
        best_std_eval = 0
        best_target_test = 0
        best_std_test = 0  

        # we don't need to find best if there is only one method
        if len(self.methods) == 1:
            
            best_method = self.methods[0]
            
        else:
            
            for method in self.methods:
        
                candidate_runs = [run for run in self.eval_runs if run.method.id == method.id]
        
                for run in candidate_runs:
                    
                    print "run id:", run.id
                    print target + ":", getattr(run.assessment, target)
        
                tmp_score = float(numpy.mean([getattr(run.assessment, target) for run in candidate_runs]))
                tmp_std = float(numpy.std([getattr(run.assessment, target) for run in candidate_runs]))
                tmp_score_test = 0
                tmp_std_test = 0

                try:
                    # determine performance on test set, but select on eval set
                    tmp_score_test = float(numpy.mean([getattr(run.assessment_test, target) for run in candidate_runs]))
                    tmp_std_test = float(numpy.std([getattr(run.assessment_test, target) for run in candidate_runs]))
                    
                except Exception, detail:
                    print "assessment_test missing"
                    print detail
                
                if (tmp_score>best_target_eval):
                    
                    best_method = method
                    best_target_eval = tmp_score
                    best_std_eval = tmp_std
                    
                    best_target_test = tmp_score_test
                    best_std_test = tmp_std_test
        
        
        return (best_method, best_target_eval, best_std_eval, best_target_test, best_std_test) 



    def select_best_method(self, target):
        """
        write best method to database
        """

        (best_method, best_target_eval, best_std_eval, best_target_test, best_std_test) = self.find_best_method(target)
            
        self.best_method = best_method
        self.best_mean_performance = best_target_eval
        self.best_std_performance = best_std_eval
    
        return (best_method, best_target_eval, best_std_eval, best_target_test, best_std_test)

        

    def create_test_run(self):
        """
        creates Run object for test_run
        
        assumes, that field best_method is set
        """
        
        if self.best_method==None:    
            print "error: please determine best method first!"
            return None
        
        # we set the eval_set pointer to -1 
        self.test_run = Run(experiment=self, method=self.best_method, x_val_idx=-1)
                
        return self.test_run
    


class SingleSourceExperiment(Experiment):
    """
    Specialization for single-source experiments
    """

    split_set = ForeignKey("SplitSet")
    
    

class MultiSourceExperiment(Experiment):
    """
    Specialization for multi-source experiments
    """

    split_set = ForeignKey("MultiSplitSet")
    
        
    #TODO generalize to work for experiments
    def create_eval_runs(self):
        """
        creates Run objects based on splits
        """
        
        if len(self.eval_runs)!=0:    
            print "warning: eval_runs already exist!"
            return self.eval_runs
        
        
        num_splits = -1
        test_num = -1
        
        #sanity checks:
        for split_set in self.split_set.split_sets:
            
            counter = 0
                            
            for split in split_set.splits:
                
                if split.is_test_set:
                    if test_num != -1:
                        #all tasks have same test set num
                        assert(split.num==test_num)
                    test_num = split.num
                    
                counter += 1

            if num_splits != -1:
                #all tasks have same number of splits
                assert(num_splits == counter)
                
            num_splits = counter
                    
        
        
        #number of splits determines x-validation
        for method in self.methods:
                
            for split in self.split_set.split_sets[0].splits:
                
                if not split.is_test_set:
                    
                    run = Run(experiment=self, method=method, x_val_idx=split.num)
                
        return self.eval_runs


class Taxonomy(SQLObject):
    """
    A data structure relating the tasks
    """

    data = PickleCol()
    description = StringCol(default="")


    def __repr__(self):
        """
        pretty representation
        """
        
        ret = self.data.__repr__() + "\n"
        ret += "description:\t" + str(self.description) + "\n"

        return ret


 
class Parameter(InheritableSQLObject):
    """
    Base class for method Parameters
    """ 
    
    cost = FloatCol(default=0.0)
    flags = PickleCol(default=None)
    
    
    def fromDict(self, d):
        """
        provides an easy way to set model from dict
        
        @param dict: dict defining parameters
        @type dict: dict<str, object>
        """
        
        #TODO: implement check whether fields present
        #problem here is that we need to take the
        #fields of parents into account
        for (key, value) in d.items():
            
            try:
                self.__setattr__(key, value)
                
            except Exception, detail:                
                print "error setting parameter from dict"
                print detail
          
        return self

    
    def __eq__(self, other):
        """
        Comparator based on values, rather than identity.
        For future extensions see:  
        http://mail.python.org/pipermail/python-dev/2005-November/057914.html
        
        @param other: the object to compare against
        @type other: should be identical to self
        """
        
        #TODO this should be more general. See how to work this into a 
        #metaclass. In particular, attribute lists should NOT be 
        #computed on each comparison.
        
        if type(other)!=type(self):
            return False
        
        
        #initialize containers
        att_list_self = []
        att_list_other = []
        
        parent_attr_self = self
        parent_attr_other = other
        
        
        #take into account parents attributes
        while parent_attr_self != None:

            
            tmp_attr_self = [col.origName for col in parent_attr_self.sqlmeta.columnList]
            tmp_attr_other = [col.origName for col in parent_attr_other.sqlmeta.columnList]

            #remove unnecessary field
            tmp_attr_self.remove("childName")
            tmp_attr_other.remove("childName")
            
            #fetch column list
            att_list_self.extend(tmp_attr_self) 
            att_list_other.extend(tmp_attr_other)

            
            parent_attr_self = parent_attr_self.sqlmeta.parentClass
            parent_attr_other = parent_attr_other.sqlmeta.parentClass


        
        if att_list_other != att_list_self:
            return False
        
        print att_list_self

        #compare attributes based on attribute list
        for i in att_list_self:
            if getattr(self, i) != getattr(other, i):
                return False
            
        return True
    
        
class ParameterSvm(Parameter):
    """
    Parameters specific to the SVM
    """
    
    kernel = StringCol(default="")
    wdk_degree = IntCol(default=0)

    #TODO remove these, left here for legacy purposes
    regul = FloatCol(default=1.0)
    num_iterations = IntCol(default=0)


class ParameterMultiSvm(ParameterSvm):
    """
    Parameters specific to Multiclass methods
    """
    
    taxonomy = ForeignKey("Taxonomy")
    base_similarity = FloatCol(default=0)
    transform = FloatCol(default=0)
    extra = FloatCol(default=0)

    def __repr__(self):
        """
        pretty representation
        """

        ret = ParameterSvm.__repr__(self) + "\n"
        ret += "Taxonomy: \n"
        ret += self.taxonomy.__repr__() + "\n"
        ret += "base sim:\t" + str(self.base_similarity) + "\n"
        ret += "transform: \t" + str(self.transform) + "\n"
        ret += "extra param:\t" + str(self.extra)


        return ret


class ParameterLog(Parameter):
    """
    Parameters specific to Logistic Regression
    """
        
    shift = FloatCol(default=0.0)
    sharpness = FloatCol(default=0.0)

    

class Assessment(InheritableSQLObject):
    """
    Class for assessment of performance
    """

    task_id = StringCol(default="")

    auROC = FloatCol(default=0.0)
    auPRC = FloatCol(default=0.0)
    sensitivity = FloatCol(default=0.0)
    specificity = FloatCol(default=0.0)
    accuracy = FloatCol(default=0.0)
    f_value = FloatCol(default=0.0)

    #TODO: remove pred and lab
    pred = PickleCol(default=None)
    lab = PickleCol(default=None)

    timestamp = TimestampCol(default=None)
    
    #TODO: remove this, one-way should be enough
    multi_assessments = RelatedJoin("MultiAssessment")

    # define path to store pickled output (to avoid writing large amounts of data to DB) 
    output_path = "/fml/ag-raetsch/share/projects/multitask/assessment/"
    

    def get_output_and_label_file_name(self):
        """
        get file name for output and label file
        """

        return self.output_path + "assessment_lab-out_"+ str(self.id) + ".bz2" 
        #return self.output_path + "assessment_lab-out_"+ self.task_id + "_" + str(self.id) + ".bz2" 


    def save_output_and_labels(self, output, labels):
        """
        trouble with db, thus save things on FS for now
        """
        
        output_file = self.get_output_and_label_file_name()
        print "saving output and labels to", output_file
        tmp_dict = {"out": output, "lab": labels}
        helper.save(output_file, tmp_dict)
    
    
    def load_output_and_labels(self):
        """
        trouble with db, thus save things on FS for now
        """

        output_file = self.get_output_and_label_file_name()
        return helper.load(output_file)


    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        output_file = self.get_output_and_label_file_name()

        # clean up file
        if os.path.exists(output_file):
            os.remove(output_file)

        self.destroySelf()

 
    def __repr__(self):
        """
        string representation
        """

        ret = "task_id: " + self.task_id + ", auROC: " + str(self.auROC) + ", auPRC: " + str(self.auPRC)  + ", timestamp: " + str(self.timestamp) + "\n"
        return ret



    
class MultiAssessment(Assessment):
    """
    performance measured on each task
    
    averages are stored directly
    """
    

    assessments = RelatedJoin("Assessment")
    


    def clean_up(self):
        """
        method to recursively delete this entry from the database
        """

        for assessment in self.assessments:
            if assessment:
                assessment.clean_up()

        self.destroySelf()


    def compute_mean(self):
        """
        compute mean of assessments
        """

        self.auROC = float(numpy.mean([a.auROC for a in self.assessments]))
        self.auPRC = float(numpy.mean([a.auPRC for a in self.assessments]))
        self.sensitivity = float(numpy.mean([a.sensitivity for a in self.assessments]))
        self.specificity = float(numpy.mean([a.specificity for a in self.assessments]))
        self.accuracy = float(numpy.mean([a.accuracy for a in self.assessments]))
        self.f_value = float(numpy.mean([a.f_value for a in self.assessments]))


    def set_from_assessment(self, target):
        """
        set to values of the underlying assessments with index target
        
        @param target: index of target assessment
        @type target: int
        """

        self.auROC = self.assessments[target].auROC
        self.auPRC = self.assessments[target].auPRC
        self.sensitivity = self.assessments[target].sensitivity
        self.specificity = self.assessments[target].specificity
        self.accuracy = self.assessments[target].accuracy
        self.f_value = self.assessments[target].f_value


    def __repr__(self):
        """
        string representation
        """

        ret = "auPRC: " + str(self.auPRC) + ", timestamp: " + str(self.timestamp)
        ret += ", auROC: " + str(self.auROC) + "\n"
        ret += "subassessments:\n"
        for assessment in self.assessments:
            ret += "\t" + assessment.__repr__()

        return ret





def create_tables():
    """
    creates all tables
    """
    
    print "creating tables"
    
    Assessment.createTable()
    Experiment.createTable()
    Dataset.createTable()
    SingleSourceExperiment.createTable()
    Instance.createTable()
    Method.createTable()
    MultiAssessment.createTable()
    MultiSourceExperiment.createTable()
    MultiSplitSet.createTable()
    Parameter.createTable()
    ParameterLog.createTable()
    ParameterMultiSvm.createTable()
    ParameterSvm.createTable()
    Run.createTable()
    Split.createTable()
    SplitSet.createTable()
    Taxonomy.createTable()
    
    
def execute_run(run_id):
   
    #fetch from db
    run = Run.get(run_id)

    print "\n\n\n##################"
    print "run id:", run_id, "method:", run.method.name
    print "##################"

    
    #assess performance
    run.execute()
   
    #force reload
    Experiment._connection.expireAll()
   
    return "success"


#TODO write unittests

