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
@summary: Defines an extensible method scaffold. 

Here, we define the Method interface.
Basically, the different algorithms are hooked into
the experimental framework by extending this class.

"""

import copy
import numpy
import helper
from helper import Options
       
from expenv import Assessment, MultiAssessment       

    
class BaseMethod(object):
    """
    abstract baseclass for
    domain adaptation methods
    
    here, the method-specific train and test procedures are 
    implemented.
    
    When creating a new method, it should sub-class this class.
    """
    
    
    
    
    def __init__(self, param):
        
        self.param = param
        self.predictor = None
        self.additional_information = {}



    def train(self, instances):
        """
        train predictor
        """

        print "parameters:"
        print self.param
        print self.param.flags
        
        self.predictor = self._train(instances, self.param) 
        
        return self.predictor
    
    
    

    def evaluate(self, instances):
        """
        Evaluates trained method on evaluation data using the trained predictor
        """

        #make sure that the predictor was trained already
        if (self.predictor == None):
            raise Exception("predictor not trained, yet")

        
        #separate labels & examples
        examples = [inst.example for inst in instances]
        labels = [inst.label for inst in instances]


        #perform assessment
        assessment = self._predict_and_assess(self.predictor, examples, labels)

        print assessment
        
        return assessment
        


    def _train(self, train_data, param):
        """
        training procedure using training examples and labels
        
        @param train_data: instances
        @type train_data: e.g. list<intances>
        @param param: param for the training procedure
        @type param: Parameter
        """

        print "called abstract method, please override"

        return 1

    

    def _predict_and_assess(self, predictor, examples, labels, task_id):
        """
        Computes predictor outputs and computes Assessment.

        @param predictor: trained predictor
        @type predictor: obj
        @param examples: evaluation examples
        @type examples: list
        @param labels: evaluation labels
        @type labels: list
        @param task_id: task id
        @type task_id: int
        """

        #compute output
        out = self._predict(predictor, examples, task_id)


        #compute assessment
        assessement = self._assess(out, labels)

        
        return assessement



    def _assess(self, out, lab):
        """
        assessment only

        @param out: predictor output
        @type svm_out: list<float>
        @param lab: labels
        @type lab: list<float>
        """

        
        auROC = helper.calcroc(out, lab)[0]
        auPRC = helper.calcprc(out, lab)[0]
        
        
        assessment = Assessment(auROC=auROC, auPRC=auPRC, pred=None, lab=None)

        if self.param.flags.has_key("save_output") and self.param.flags["save_output"] == True:
            assessment.save_output_and_labels(out, lab)

        print assessment

        return assessment



    def _predict(self, predictor, examples, task_id):
        """
        make prediction on examples using trained predictor

        @param predictor: all information needed to predict
        @type predictor: obj
        @param examples: list of examples
        @type examples: list
        @param task_id: task id
        @type task_id: str
        """

        print "abstract method _predict, please override"
        
        out = []

        return out



    def clear(self):
        """
        reset predictor
        """
        
        del self.predictor


    def save_predictor(self, file_name):
        """
        saves predictor to file system for later use
        
        @param file_name: file name to save predictor
        @type file_name: str
        """
        
        print "saving predictor to", file_name
        print self.predictor
        
        try:
            helper.save(file_name, self.predictor, "gzip")
        except Exception, detail:
            print "error writing predictor"
            print detail

        


#########################################################
# Multi Stuff

class MultiMethod(BaseMethod):
    """
    Method baseclass to deal with more than one data source
    """
    


    def evaluate(self, eval_data, target_task=-1):
        """
        Evaluates trained method on evaluation data using the trained SVM
        
        @param eval_data: evaluation set containing examples and labels
        @type eval_data: dict<str, list<instances> >
        @param target_task: if set to -1, we consider average otherwise specific task
        @type target_task: int
        """


        #make sure that the predictor was trained already
        if (self.predictor == None):
            raise Exception("predictor not trained, yet")


        #assessment for each task
        multi_assessment = MultiAssessment()

        # we use generator to efficiently iterate through items
        for (task_id, instances) in eval_data:

            #print "eval split_set:", instances[0].dataset.organism

            #separate labels & examples
            examples = [inst.example for inst in instances]
            labels = [inst.label for inst in instances]
            
             
            #import pdb; pdb.set_trace()
            #perform assessment
            assessment = self._predict_and_assess(self.predictor[task_id], examples, labels, task_id)

            #attach task_id
            assessment.task_id = task_id

            multi_assessment.addAssessment(assessment)
            

        #set top-level assessment values
        if target_task == -1:
            multi_assessment.compute_mean()
            
        elif target_task >= 0:            
            multi_assessment.set_from_assessment(target_task)


        return multi_assessment




class PreparedMultitaskData(Options):
    '''
    Class to hold information for prepared multitask data.
    
    The main point is that it extracts examples, labels, task_names and task_nums
    and provides them as separate variables.
    
    Also, it is important that this is the place, where we provide
    the mapping from task_names to task_ids.
    
    Furthermore, it allows the shuffling of a data set on creation.
    
    Once, created the instance is frozen.
    '''
    

    
    def __init__(self, instance_set, shuffle=False):
        '''
        @param instance_set: Mulittask data structure
        @type instance_set: dict<str, list<instances> >
        
        @param shuffle: boolean to indicate whether to shuffle dataset
        @type shuffle: bool
        '''
    
        # create temp containers
        examples = []
        labels = []
        task_vector_names = []
        task_vector_nums = []
        
        self.__name_to_id = {}
        self.__id_to_name = {}
            
                
        # sort by task_name
        task_names = list(instance_set.keys())
        task_names.sort()
        
        
        # extract training data
        for (task_id, task_name) in enumerate(task_names):
            
            instances = instance_set[task_name]
            
            print "train task name:", task_name
            #assert(instances[0].dataset.organism==task_name)
            
            examples.extend([inst.example for inst in instances])
            labels.extend([inst.label for inst in instances])
            
            task_vector_names.extend([str(task_name)]*len(instances))
            task_vector_nums.extend([task_id]*len(instances))
            
            
            # add mapping information
            self.__name_to_id[task_name] = task_id
            self.__id_to_name[task_id] = task_name
        
        
        
        self.num_examples = len(examples)            
        

        # shuffle dataset if option is turned on
        if shuffle:
             
            # determine permutation
            idx = numpy.random.permutation(range(self.num_examples))
        
            # apply permutation to relevant vectors
            examples = numpy.array(examples)[idx].tolist()
            labels = numpy.array(labels)[idx].tolist()
            task_vector_names = numpy.array(task_vector_names)[idx].tolist()
            task_vector_nums = numpy.array(task_vector_nums)[idx].tolist()

            # save permutation
            self.permutation = idx 

                
        # sanity checks
        assert(isinstance(examples, list))
        assert(isinstance(labels, list))
        assert(isinstance(task_vector_names, list))
        assert(isinstance(task_vector_nums, list))
                
        assert(self.num_examples == len(labels))
        assert(self.num_examples == len(task_vector_names))
        assert(self.num_examples == len(task_vector_nums))
        
        for i in xrange(self.num_examples):
            assert(type(task_vector_names[i])==str)
            assert(type(task_vector_nums[i])==int)
        
        # make sure we have the same keys (potentially in a different order)
        sym_diff_keys_a = set(self.__name_to_id.keys()).symmetric_difference(set(self.__id_to_name.values()))
        assert len(sym_diff_keys_a)==0, "symmetric difference between keys non-empty: " + str(sym_diff_keys_a)
        
        sym_diff_keys_b = set(self.__name_to_id.values()).symmetric_difference(set(self.__id_to_name.keys()))
        assert len(sym_diff_keys_b)==0, "symmetric difference between keys non-empty: " + str(sym_diff_keys_b)  

        
        self.examples = examples
        self.labels = labels
        self.task_vector_names = task_vector_names
        self.task_vector_nums = task_vector_nums
        
        # disallow changes to this instance
        self.freeze()
        
    
    def get_task_names(self):
        '''
        get list of task names
        '''
    
        return copy.copy(self.__name_to_id.keys())
    
    
    def get_task_ids(self):
        '''
        get list of task names
        '''
    
        return copy.copy(self.__id_to_name.keys())
    
    
    def id_to_name(self, idx):
        '''
        map task id to task name
        
        @param idx: id to map
        @type idx: int
        '''
        
        return self.__id_to_name[idx]
    
    
    def name_to_id(self, name):
        '''
        map task name to assigned id
        
        @param name: name to mpa
        @type name: str
        '''
        
        return self.__name_to_id[name]
    
    
    def get_num_tasks(self):
        '''
        get number of tasks
        '''
        
        return len(self.__name_to_id.keys())
    
