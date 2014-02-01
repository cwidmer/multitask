
# import std packages
import os
import sys
import random
import getopt
import re
import copy
import time


import numpy.random
import scipy.io
import numpy

import expenv

#import datagen



from random import choice
from expenv import Dataset, Instance, MultiSplitSet
from collections import defaultdict
from helper import rand_seq
import helper    

#multi_split_set = prepare_multi_datasets(["pacificus", "remanei", "drosophila", "thaliana"], 6500, 500, 0)
#multi_split_set = generate_toy_data()

def load_data(organisms, size, subsample=0, prefix="", load_test=False):
    """
    Helper function to load simple dataset
    """

    dir_name = "/fml/ag-raetsch/share/projects/domain_adapt/data/paperdata/"
    test_dir_name = "/fml/ag-raetsch/share/projects/domain_adapt/data/testdata/"


    #load data from previously prepared files
    data = []

    for (i, organism) in enumerate(organisms):
    
        train_file_name = dir_name + "/" + prefix + organism + "_size=" + str(size) + "_subsample=" + str(subsample) + ".mat"
        test_file_name = test_dir_name + "/" + "TEST_" + organism  + ".mat"

        print "loading:", train_file_name

        dataset = parse_mat_file(train_file_name)

        print "size:", len(dataset["LT"])

        if load_test: 
            print "appending testset"
            print "loading:", test_file_name
            dataset_test = parse_mat_file(test_file_name)
            print "size testset:", len(dataset_test["LT"])
            dataset_test["XT"].extend(dataset["XT"])
            dataset_test["LT"].extend(dataset["LT"])
        
            
            dataset = dataset_test

        dataset["organism"] = organism

        data.append(dataset)


    return data

    
    
def parse_mat_file(fn):
    """
    parses single mat file and returns examples and lables as dictionary
    """

    mat_file = scipy.io.loadmat(fn)

    #fetch key from mat file (stored in dict)
    try:
        key = [key for key in mat_file.keys() if key.startswith("d")][0]
    except Exception:
        print "key not found, assuming different format"
        key = None

    dataset = {}

    dataset["file_name"] = fn
    
    if key==None:
        dataset["XT"] = numpy.array(mat_file["XT"]).tolist()
        dataset["LT"] = numpy.array(mat_file["LT"]).tolist()
    else:
        dataset["XT"] = numpy.array(mat_file[key].XT).tolist()
        dataset["LT"] = numpy.array(mat_file[key].LT).tolist()

    return dataset



def clone_mss(mss_id, num_splits=3, num_total=150, num_test=50, fraction_positives=0.5, write_db=False):
    '''
    Helper function to clone and subsample from existing MSS.    
    
    '''


    mss = expenv.MultiSplitSet.get(mss_id)

    if write_db:
        multi_split_set = MultiSplitSet()

        # copy fields with additional information
        multi_split_set.description = mss.description
        multi_split_set.feature_type = mss.feature_type
        multi_split_set.taxonomy = mss.taxonomy


    for ss in mss.split_sets:

        print ss

        task_id = ss.get_task_id()

        instances = []

        # merge dataset
        for split in ss.splits:
            instances.extend(split.instances)


        print "len instances", len(instances)
        assert(num_total <= len(instances))

        # extract labels
        labels = [inst.label for inst in instances]
        
        # subsample dataset
        keeper_idx = select_indices(num_total, fraction_positives, labels)

        # select keepers
        instances = numpy.array(instances)[keeper_idx].tolist()

        # create dataset
        if write_db:
            dataset = Dataset(organism=task_id, signal="TSS")
            print dataset
 
        for instance in instances:

            seq = instance.example
            label = instance.label

            # create instance object
            if write_db:
                Instance(dataset=dataset, example=seq, label=label)            

        print "=============="

        # create splitset
        if write_db:
            split_set = dataset.create_split_set(num_splits, size_testset=num_test, random=True)
            multi_split_set.addSplitSet(split_set)


    if write_db:
        print "created multi split set:"
        print multi_split_set

        return multi_split_set



    
def clone_cut(mss_id=386, num_splits=3, fraction_test=0.3, write_db=False):
    '''
    Helper function to clone and subsample from existing MSS.    
    
    '''


    mss = expenv.MultiSplitSet.get(mss_id)

    if write_db:
        multi_split_set = MultiSplitSet()

        # copy fields with additional information
        multi_split_set.description = mss.description + " 5 tasks"
        multi_split_set.feature_type = mss.feature_type
        multi_split_set.taxonomy = mss.taxonomy


    for ss in mss.split_sets:


        #if int(ss.get_task_id().replace("task_", ""))<20 or ss.num_instances < 400 or ss.get_task_id=="B_2705":
        if ss.get_task_id()=="B_2705":
            continue

        print ss
        print ss.get_task_id()

        task_id = ss.get_task_id()

        instances = []

        # merge dataset
        for split in ss.splits:
            instances.extend(split.instances)

        # extract labels
        labels = [inst.label for inst in instances]
                
        num_total = len(labels)
        num_test = int(float(num_total) * fraction_test)

        print "len instances", len(instances)
        assert(num_total == len(instances))

        # create dataset
        if write_db:
            dataset = Dataset(organism=task_id, signal="TSS")
            print dataset
 
        for instance in instances:

            seq = instance.example
            label = instance.label

            # create instance object
            if write_db:
                Instance(dataset=dataset, example=seq, label=label)            

        print "=============="

        # create splitset
        if write_db:
            split_set = dataset.create_split_set(num_splits, size_testset=num_test, random=True)
            multi_split_set.addSplitSet(split_set)


    if write_db:
        print "created multi split set:"
        print multi_split_set

        return multi_split_set


def save_mss_for_matlab(mss_id, target_dir=None, file_name=None):
    
    if target_dir == None:
        mypath = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mat/"
    else:
        mypath = target_dir
    
    mss = expenv.MultiSplitSet.get(mss_id)

    if file_name==None:
        file_name = mss.description
    
    train = mss.get_train_data(0)
    #test = mss.get_test_data(0)
    
    
    for (key, value) in train.items():
        
        # compile training set
        example_set = [inst.example for inst in value]
        label_set = [inst.label for inst in value]
        
        task = {"XT": example_set, "LT": label_set}
        
        scipy.io.savemat(mypath + file_name + "_" + str(key), task)
        


def compute_bayes_error_for_id(mss_id):
    """
    computes bayes error for mss with id mss_id
    """
    
    # load multiple split set from database
    mss = expenv.MultiSplitSet.get(mss_id)

    # load generation data from file system
    gen_dat = helper.load("/fml/ag-raetsch/share/projects/multitask/generation_data/mss_" + str(mss_id) + ".bz2")

    compute_bayes_error(mss, gen_dat["pwms"], gen_dat["length"], gen_dat["num_test"])



def compute_bayes_error(mss, pssms, length, num_test, comment=""):
    """
    adds bayes error to existing multi split set
    """


    # create dataset with plenty of datapoint and copy test set from previously created dataset
    
    mss_bayes = generate_toy_data_from_pssms(pssms, length, num_splits=3, num_train=10000, num_test=num_test, write_db=True, write_fasta=False)
    
    
    # copy test set
    for (i, ss_bayes) in enumerate(mss_bayes.split_sets):
        
        ss = mss.split_sets[i]
        
        assert(ss_bayes.get_task_id() == ss.get_task_id())

        s = ss.get_test_set()
        s_bayes = ss_bayes.get_test_set()
        
        assert(s.is_test_set)
        assert(s_bayes.is_test_set)
        
        instances = s.instances
        instances_bayes = s_bayes.instances

        print "len inst", len(instances)
        print "len bayes", len(instances_bayes)
        assert(len(instances) == len(instances_bayes))


        for (j, inst) in enumerate(instances):

            # overwrite example
            inst_bayes = instances_bayes[j]
            
            inst_bayes.example = inst.example
            inst_bayes.label = inst.label
        
        
    # compute Bayes error
    run_it(mss_bayes.id, methods=["method_plain_svm"], comment=comment+" [Bayes Error for mss=" + str(mss.id))



def do_it(comment="", length=10, total_num_train=1000, num_splits=5, bayes_error=False):
    
    
    num_train = total_num_train / (num_splits-1) 
    
    
    #pssms = fetch_hardcoded_pssms(length)
    #pssms = fetch_sampled_pssms(length)
    #name = "toy"
    
    #pssms = fetch_sampled_pssms_deep(length) 
    #name = "deep"
    
    pssm_creator = BinaryTreePSSMFactory(entropy=1.0, relative_entropy=0.40, length=length, epsilon=0.002)

    num_levels = 2
    pssms = pssm_creator.sample_PSSMs(num_levels, sigma = 0.4 / length)
    name = "deep" + str(len(pssms)) 
    
    
    mss = generate_toy_data_from_pssms(pssms, num_splits=num_splits, num_train=num_train, num_test=5000, write_db=True, write_fasta=False, name=name)
    
    
    if bayes_error:
        compute_bayes_error(mss, pssms, length, num_splits, num_train, name=name)    
        
    
    run_it(mss.id, comment)






def run_it(mss_id, comment="", methods=None, cluster=True):

    cluster_flag = ""
    if cluster:
        cluster_flag = "-c"
    
    if methods == None:
        #methods = ["method_plain_svm", "method_union_svm", "method_hierarchy_svm", "method_xval_hierarchy_svm", "method_augmented_svm"]
        methods = ["method_plain_svm", "method_union_svm", "method_hierarchy_svm", "method_augmented_svm", "method_pairwise_multitask"]
    
    
    for method in methods:
        prefix = "nohup python expenv_runner.py " + cluster_flag + " new "
        suffix = ' "' + comment + '"  1.0 2> /dev/null &'
        command = prefix + str(mss_id) + " " + method + suffix
        print command 
        
        os.system(command)
        
        time.sleep(4)



class PSSM(object):
    """
    scoring matrix
    """
    
    
    def __init__(self, alphabet, length):
        """
        
        @param alphabet: alphabet to be used
        @type alphabet: list<str>
        @param length: length of pwd
        @type length: int 
        """
        
        self.alphabet = alphabet
        self.length = length
        self.data = []
        
        uniform_p = 1.0/float(len(self.alphabet))
        
        for i in xrange(length):
            
            column = dict()
            
            for c in alphabet:
                column[c] = uniform_p
                
            self.data.append(column)
        

    def __repr__(self):
        """
        string representation
        """
        
        line = ""
        
        for (i, c) in enumerate(self.alphabet):
            line += c + ": "
            
            for j in xrange(self.length):
                
                line += str(numpy.round(self.data[j][c], 2)) + "  "
                
            line += "\n"
            
            
        return line
            
        
    def set_column(self, column_id, column):
        """
        
        @param column_id: id of which column to set
        @type column_id: int
        @param column: column data
        @type column: dict<str, float> 
        """
        
        if self.column_sane(column):
        
            self.data[column_id] = column

        else:
            
            print "warning: column not sane, not set" 


    def column_sane(self, column, epsilon = 0.01):
        """
        check if column is correct
        
        @param column: column data
        @type column: dict<str, float> 
        @return: boolean to indicate whether probabilities add up to 1
        @rtype: boolean
        """
                
        mysum = sum(column.values())
        
        issane = (mysum > 1.0 - epsilon) and (mysum < 1.0 + epsilon)

        return issane 


    def sample_sequences(self, num_seq):
        """
        generate sequences
         
        @param num_seq: number of sequences
        @type num_seq: int
        @return: list of samples sequences
        @rtype: list<str>
        """
        
        
        #sanity check
        for column in self.data:
            assert(self.column_sane(column))
        
        
        seqs = []
        
        for i in xrange(num_seq):
            seqs.append(self.sample_sequence())
            
        return seqs
    
    

    def sample_sequence(self):
        """
        generate sequence based on PSSM
         
        @return: character
        @rtype: str
        """
        
                
        seq = ""
        
        for i in range(self.length):
            
            seq += self._sample_char(i)
            
            
        return seq

    

    def _sample_char(self, column_id):
        """
        
        @param column_id: id of which column to sample from
        @type column_id: int 
        @return: character
        @rtype: str
        """
        
        
        last_value = 0        
        boundaries = []
        
        for c in self.alphabet:
            
            p = self.data[column_id][c]
            current_value = last_value + p
            boundaries.append(current_value)            
            last_value = current_value


        rand_num = random.random()
        
        lower_bound = 0
        for (i, upper_bound) in enumerate(boundaries):
            
            if rand_num >= lower_bound and rand_num < upper_bound:
                return self.alphabet[i]
            
            lower_bound = upper_bound
        



    def mutate_column(self, column_id, sigma=0.1):
        """
        
        mutates a pair of probabilities
        
        @param column_id: id of which column to set
        @type column_id: int
        @param sigma: standard deviation
        @type sigma: int
        
        @return: mutated column
        @rtype: dict<str, float>
        """
        
        column = self.data[column_id]
        
        #edit_keys = random.sample(column.keys(), 2)
        edit_keys = column.keys()
        
        
        for key in edit_keys:
            
            difference = random.gauss(0,sigma)
            
            #TODO we only randomly chose the sign before
            #difference = sigma*numpy.sign(random.gauss(0,sigma))
            
            column[key] += difference
        
        column = self.normalize_column(column)
        
        return column




    def mutate(self, sigma=0.1):
        """
        
        mutates PSSM by mutating every column
        
        @param sigma: standard deviation
        @type sigma: int
        
        @return self
        @rtype PSSM 
        """
        
        for i in xrange(self.length):
            
            self.mutate_column(i, sigma)
        

        return self

        
        
    def normalize_column(self, column):
        
        
        for (key, value) in column.items():
            
            if value <= 0.0:
                column[key] = 0.05
                
            if value >= 1.0:
                column[key] = 0.95
                
        mysum = sum(column.values())
        
        norm_factor = 1.0 / mysum
        
        for (key, value) in column.items():
            column[key] *= norm_factor
         
         
        return column



    def compute_entropy_column(self, column):
        """
        compute entropy for single column
        
        @param column
        @type dict<str, float>
        
        @return entropy for column
        @rtype float
        """
        
        entropy = 0
        
        for value in column.values():
            
            entropy += - value * numpy.log2(value)
        
        
        return entropy
        
        
        
    def compute_entropy(self, left_idx=0, right_idx=0, normalize=False):
        """
        compute average entropy of entire PSSM
        
        @param left_idx: left index of interval over which entropy is computed
        @type left_idx: int
        @param right_idx: right index of interval over which entropy is computed
        @type right_idx: int
        @param normalize: flag if entropy is to be averaged over columns
        @type normalize: bool
        @return average entropy
        @rtype float
        """
        
        total_entropy = 0
        
        
        if right_idx == 0:
            right_idx = self.length
        
        
        for i in xrange(left_idx, right_idx):
            
            column = self.data[i]
            
            total_entropy += self.compute_entropy_column(column)
        
        
        if normalize:
            entropy = total_entropy / (right_idx - left_idx)
        else:
            entropy = total_entropy
        
        return entropy
                
    
    def compute_information_content(self, left_idx=0, right_idx=0, normalize=False):
        """
        compute sum_i 2 - entropy_i over columns 
        
        @param left_idx: left index of interval over which entropy is computed
        @type left_idx: int
        @param right_idx: right index of interval over which entropy is computed
        @type right_idx: int
        @param normalize: flag if entropy is to be averaged over columns
        @type normalize: bool
        @return total information content
        @rtype float
        """
        
        information_content = 0
        
        
        if right_idx == 0:
            right_idx = self.length
        
        
        for i in xrange(left_idx, right_idx):
            
            column = self.data[i]
            
            information_content += 2 - self.compute_entropy_column(column)
        
        
        if normalize:
            information_content = information_content / (right_idx - left_idx)
        
        
        return information_content
        
                
                
    def compute_relative_entropy_columns(self, column1, column2):
        """
        compute relative entropy of column1 given column2
        KL(column1 | column2) = KL(p | q) = - sum p(x) * log2 (q(x) / p(x))
        
        @param column1: column 1
        @type column1: dict<str, float>
        @param column2: column 2, assumed as given
        @type column2: dict<str, float>
        @return relative entropy of Kullback-Leibler divergence
        @rtype float
        """
        
        #sanity checks
        assert(len(column1.values()) == len(column2.values()))
        assert(column1.keys() == column2.keys())
        
        
        relative_entropy = 0
        
        for key in column1.keys():
            
            value1 = column1[key]
            value2 = column2[key]
            
            #avoid div zero
            value1 += 0.00001
            
            relative_entropy += - value1 * numpy.log2(value2 / value1 )
        
        
        return relative_entropy
        
        
        
        
        
    def compute_relative_entropy(self, pssm, left_idx=0, right_idx=0, normalize=False):
        """
        compute relative entropy of self given another PSSM

        @param pssm: other pssm w.r.t. which the relative entropy is computed
        @type pssm: PSSM
        @param left_idx: left index of interval over which entropy is computed
        @type left_idx: int
        @param right_idx: right index of interval over which entropy is computed
        @type right_idx: int
        @param normalize: flag if entropy is to be averaged over columns
        @type normalize: bool
        @return average relative entropy
        @rtype float
        """
        
        total_entropy = 0
        
        
        if right_idx == 0:
            right_idx = self.length
        
        
        for i in xrange(left_idx, right_idx):
            
            column1 = self.data[i]
            column2 = pssm.data[i]
            
            total_entropy += self.compute_relative_entropy_columns(column1, column2)
        
        if normalize:
            entropy = total_entropy / (right_idx - left_idx)
        else:
            entropy = total_entropy
        
        return entropy 
        
                
                

def write_to_fasta(file_name, sequences, labels):
    
    print "writing fasta file:", file_name    
    
    f = file(file_name, "a")
    
    for (i,seq) in enumerate(sequences):
        
        label = labels[i]
        
        f.write(">" + str(label) + "\n")
        f.write(seq + "\n")

    f.close()
    print "done writing fasta"



def fetch_sampled_pssms_deep(length):

    target_entropy = 1.0
    relative_entropy_near = 0.40
    
    relative_entropy_middle = 0.20
    relative_entropy_far = 1.0
    
    
    #fix entropy of parent
    root = optimize_pssm_by_sampling(length, entropy=target_entropy, epsilon=0.001, sigma=0.01)
    
    epsilon = 0.002
    sigma = 0.4 / length
    
    left = optimize_mutation_by_sampling(root, target_entropy, relative_entropy_near, epsilon, sigma)
    right = optimize_mutation_by_sampling(root, target_entropy, relative_entropy_near, epsilon, sigma)
    
    toy_0 = optimize_mutation_by_sampling(left, target_entropy, relative_entropy_near, epsilon, sigma)
    toy_0_a = optimize_mutation_by_sampling(toy_0, target_entropy, relative_entropy_near, epsilon, sigma)
    toy_0_b = optimize_mutation_by_sampling(toy_0, target_entropy, relative_entropy_near, epsilon, sigma)
    
    toy_1 = optimize_mutation_by_sampling(left, target_entropy, relative_entropy_near, epsilon, sigma)
    toy_1_a = optimize_mutation_by_sampling(toy_1, target_entropy, relative_entropy_near, epsilon, sigma)
    toy_1_b = optimize_mutation_by_sampling(toy_1, target_entropy, relative_entropy_near, epsilon, sigma)
    
    toy_2 = optimize_mutation_by_sampling(right, target_entropy, relative_entropy_near, epsilon, sigma)
    toy_2_a = optimize_mutation_by_sampling(toy_2, target_entropy, relative_entropy_near, epsilon, sigma)
    toy_2_b = optimize_mutation_by_sampling(toy_2, target_entropy, relative_entropy_near, epsilon, sigma)
    
    toy_3 = optimize_mutation_by_sampling(right, target_entropy, relative_entropy_near, epsilon, sigma)
    toy_3_a = optimize_mutation_by_sampling(toy_3, target_entropy, relative_entropy_near, epsilon, sigma)
    toy_3_b = optimize_mutation_by_sampling(toy_3, target_entropy, relative_entropy_near, epsilon, sigma)
    

    return [toy_0_a, toy_0_b, toy_1_a, toy_1_b, toy_2_a, toy_2_b, toy_3_a, toy_3_b]




def fetch_sampled_pssms(length):
    """
    
    to have perfect control over PSSMs, we sample with fixed entropy and relative entropy to parents
    
    @param length: length of pssm
    @type length: int
    
    @return: list of hardcoded pssms (4)
    @rtype: list<PSSM>
    """    
    
    target_entropy = 1.0
    relative_entropy_near = 1.0
    
    #relative_entropy_middle = 0.20
    #relative_entropy_far = 1.0
    
    
    #fix entropy of parent
    root = optimize_pssm_by_sampling(length, entropy=target_entropy, epsilon=0.001, sigma=0.01)
    
    #return [root, root]
    
    epsilon = 0.002
    sigma = 0.3 / length
    
    left = optimize_mutation_by_sampling(root, target_entropy, relative_entropy_near, epsilon*2, sigma*2)
    right = optimize_mutation_by_sampling(root, target_entropy, relative_entropy_near, epsilon*2, sigma*2)
    
    return [left, right]
    
    toy_0 = optimize_mutation_by_sampling(left, target_entropy, relative_entropy_near, epsilon, sigma)
    toy_1 = optimize_mutation_by_sampling(left, target_entropy, relative_entropy_near, epsilon, sigma)
    
    toy_2 = optimize_mutation_by_sampling(right, target_entropy, relative_entropy_near, epsilon, sigma)
    
    
    #note: last leaf has different relative entropy!!
    toy_3 = optimize_mutation_by_sampling(right, target_entropy, relative_entropy_near*3, epsilon*2, sigma*2)
    #toy_3 = optimize_mutation_by_sampling(right, target_entropy, relative_entropy_near, epsilon, sigma)
    
    #mutate the hell out of this one
    #toy_3.mutate()
    #toy_3.mutate()
    #toy_3.mutate()


    return [toy_0, toy_1, toy_2, toy_3]



def fetch_hardcoded_pssms(length):
    """
    
    for debugging purposes, we manually generate a set of pssms
    
    @param length: length of pssm
    @type length: int
    
    @return: list of hardcoded pssms (4)
    @rtype: list<PSSM>
    """
    

    pwm = PSSM(["a", "g", "c", "t"], length)
    
    col1 = {"a": 0.4, "g": 0.25, "c": 0.25, "t": 0.1}
    
    pwm.set_column(0, col1)
    pwm.set_column(1, col1)
    pwm.set_column(2, col1)
    #pwm.set_column(3, col2)
    #pwm.set_column(4, col2)
    #pwm.set_column(5, col2)    
    
    pwm_a = copy.deepcopy(pwm)
    pwm_a.set_column(0, {"a": 0.45, "g": 0.20, "c": 0.25, "t": 0.1})
    pwm_a.set_column(1, {"a": 0.45, "g": 0.20, "c": 0.25, "t": 0.1})
    pwm_a.set_column(2, {"a": 0.45, "g": 0.20, "c": 0.25, "t": 0.1})
    

    pwm_a1 = copy.deepcopy(pwm_a)
    pwm_a1.set_column(1, {"a": 0.5, "g": 0.15, "c": 0.25, "t": 0.1})
    
    pwm_a2 = copy.deepcopy(pwm_a)
    pwm_a2.set_column(1, {"a": 0.45, "g": 0.20, "c": 0.15, "t": 0.2})

    
    pwm_b = copy.deepcopy(pwm)
    pwm_b.set_column(0, {"a": 0.3, "g": 0.35, "c": 0.25, "t": 0.1})
    pwm_b.set_column(1, {"a": 0.3, "g": 0.35, "c": 0.25, "t": 0.1})
    pwm_b.set_column(2, {"a": 0.3, "g": 0.35, "c": 0.25, "t": 0.1})
    
    
    pwm_b1 = copy.deepcopy(pwm_b)
    pwm_b1.set_column(0, {"a": 0.2, "g": 0.45, "c": 0.25, "t": 0.1})
    pwm_b1.set_column(1, {"a": 0.2, "g": 0.45, "c": 0.25, "t": 0.1})
    
    pwm_b2 = copy.deepcopy(pwm_b)
    pwm_b2.set_column(0, {"a": 0.2, "g": 0.55, "c": 0.05, "t": 0.2})
    pwm_b2.set_column(1, {"a": 0.3, "g": 0.35, "c": 0.15, "t": 0.2})
    pwm_b2.set_column(2, {"a": 0.2, "g": 0.55, "c": 0.05, "t": 0.2})
    
    pwm_b2.mutate()
    
    print "entropy:"
    print "======================="
    print "pwm_a1:", pwm_a1.compute_entropy(0, 3)
    print "pwm_a2:", pwm_a2.compute_entropy(0, 3)
    print "-----------------------"
    print "pwm_b1:", pwm_b1.compute_entropy(0, 3)
    print "pwm_b2:", pwm_b2.compute_entropy(0, 3)
    print "=======================\n\n"
    
    print "relative entropy:"
    print "======================="
    print "KL(pwm_a | pwm):", pwm_a.compute_relative_entropy(pwm, 0, 3)
    print "KL(pwm_b | pwm):", pwm_b.compute_relative_entropy(pwm, 0, 3)
    print "-----------------------"
    print "KL(pwm_a1 | pwm_a):", pwm_a1.compute_relative_entropy(pwm_a, 0, 3)
    print "KL(pwm_a2 | pwm_a):", pwm_a2.compute_relative_entropy(pwm_a, 0, 3)
    print "-----------------------"
    print "KL(pwm_b1 | pwm_b):", pwm_b1.compute_relative_entropy(pwm_b, 0, 3)
    print "KL(pwm_b2 | pwm_b):", pwm_b2.compute_relative_entropy(pwm_b, 0, 3)    
    print "======================="    
    pwms = [pwm_a1, pwm_a2, pwm_b1, pwm_b2]


    return pwms
    
    
    
def generate_toy_data_from_pssms(pssms, num_splits=4, num_train=300, num_test=5000, ratio_pos_neg=0.5, write_db=False, write_fasta=False, name="", seed = 52342342349):

    
    mypath = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/toy/"

    
    length = pssms[0].length
    neg = PSSM(["a", "g", "c", "t"], length)
    
    num_organisms = len(pssms)
    
    numpy.random.seed = seed

    # save parameters relevant for PSSM generation
    #generation_parameters = locals()
    generation_parameters = 1
    #TODO: change back!
    


    if write_db:
        
        multi_split_set = MultiSplitSet(feature_type="string")
        
        multi_split_set.set_generation_data(generation_parameters)
        #multi_split_set = MultiSplitSet()

        if name=="":
            multi_split_set.description = "toy" + str(multi_split_set.id)
        else: 
            multi_split_set.description = name + "_" + str(multi_split_set.id)
   

    for i in xrange(num_organisms):

        
        org_name = "toy_" + str(i)

        #create dataset
        if write_db:
            dataset = Dataset(organism=org_name, signal="acc")
            print dataset
         
        #create instances           
        instances = []
        
        for j in xrange(num_splits):
            
            out_file = ""
            
            if j==0:
                num_inst = num_test
                out_file = mypath + org_name + "_test.fasta" 
            else:
                num_inst = num_train
                out_file = mypath + org_name + "_train.fasta"


            #fetch matrix
            tmp_pwm = pssms[i]

            seqs = tmp_pwm.sample_sequences(int(num_inst*ratio_pos_neg))
            
            print "len pos seqs", len(seqs)
            labels = [1]*int(num_inst*ratio_pos_neg)

            neg_seqs = neg.sample_sequences(int(num_inst*(1-ratio_pos_neg)))
            
                
            seqs.extend(neg_seqs)
            
            labels.extend([-1]*int(num_inst*(1-ratio_pos_neg)))
     
            num_examples = len(seqs)
            
            #shuffle
            idx = numpy.random.permutation(num_examples)
            seqs = numpy.array(seqs)[idx].tolist()
            labels = numpy.array(labels)[idx].tolist()


            if write_fasta:
                write_to_fasta(out_file, seqs, labels)

            # print stuff
            #for mys in [s+": "+ str(labels[k]) for (k,s) in enumerate(seqs)]:
            #    print mys

            for k in xrange(num_examples):

                seq = seqs[k].upper()
                label = labels[k]

                #create instance object
                if write_db:
                    instance = Instance(dataset=dataset, example=seq, label=label)            
                    instances.append(instance)

            print "--------------"
        print "=============="

        #create splitset
        if write_db:
            split_set = dataset.create_split_set(num_splits, size_testset=num_test, random=False)
            multi_split_set.addSplitSet(split_set)

    if write_db:
        print "created multi split set:"
        print multi_split_set

        return multi_split_set
    
    

def generate_toy_data_from_motif(num_splits=4, num_train=300, num_test=5000, ratio_pos_neg=0.5, length=50, write_db=False, mut_prob=0.7, name="", seed = 52342342349):

    numpy.random.seed = seed

    # motifs in different distances to each other


    #motifs = ["GGGGG", "GGGGT", "GGAGT", "GGATG"]
    #motifs = ["GAATAG", "GAAGAG", "GAAATG", "GAACTG"]
    #motifs = ["ATGTAA", "ATCTGA", "GTAACG", "AAGGCG"]
    #motifs = ["TAGA", "TTAA", "GGAA", "TGAA"]
    motifs = ["TCAAA", "GCAAA"] #, "AACTA", "AACGA"]
    #motifs = ["AAAA", "AAAA", "AAAA", "AAAA"]
    


    if write_db:
        
        multi_split_set = MultiSplitSet()

        if name=="":
            multi_split_set.description = "toy" + str(multi_split_set.id)
        else: 
            multi_split_set.description = name
   

    for (i, motif) in enumerate(motifs):

        #create dataset
        if write_db:
            dataset = Dataset(organism="toy_"+str(i), signal="acc")
            print dataset
         
        #create instances           
        instances = []
        
        for j in xrange(num_splits):
            
            if j==0:
                num_inst = num_test
            else:
                num_inst = num_train
        

            print " j:", j, motif

            seqs = datagen.motifgen(motif, (num_inst*ratio_pos_neg), length, length, 10, 10, mut_prob)
            seqs = seqs[1]
            print "len seqs", len(seqs)
            labels = [1]*int(num_inst*ratio_pos_neg)

            seqs.extend(datagen.motifgen(motif, (num_inst*(1-ratio_pos_neg)), length, length, 10, 10, 1.0)[1])
            labels.extend([-1]*int(num_inst*(1-ratio_pos_neg)))
     
            num_examples = len(seqs)
            idx = numpy.random.permutation(num_examples)
            seqs = numpy.array(seqs)[idx].tolist()
            labels = numpy.array(labels)[idx].tolist()

            # print stuff
            for mys in [s+": "+ str(labels[k]) for (k,s) in enumerate(seqs)]:
                print mys

            for k in xrange(num_examples):

                seq = seqs[k].upper()
                label = labels[k]

                #create instance object
                if write_db:
                    instance = Instance(dataset=dataset, example=seq, label=label)            
                    instances.append(instance)

            print "--------------"
        print "=============="

        #create splitset
        if write_db:
            split_set = dataset.create_split_set(num_splits, size_testset=num_test, random=False)
            multi_split_set.addSplitSet(split_set)

    if write_db:
        print "created multi split set:"
        print multi_split_set

        return multi_split_set




def generate_toy_data_simple(num_datasets=4, num_splits=3, num_train=1000, num_test=500, ratio_pos_neg=0.5, length=20):


    print "creating toy dataset"
    
    #creating dataset
    alphabet = ["A", "C", "G", "T"]
    candidate_labels = [-1.0, 1.0]
    
    num_pos = 10
    num_train = 1000
    num_test = 1000
    
    multi_split_set = MultiSplitSet()
    
    
    for i in xrange(num_datasets):

        #create dataset
        dataset = Dataset(organism="dodo_"+str(i), signal="acc")
        print dataset
         
        #create instances           
        instances = []
        
        for j in xrange(num_splits):
            
            if j==0:
                num_inst = num_test
            else:
                num_inst = num_train
        
            for k in xrange(num_inst):
                #TODO attempt to correlate patterns with labels
                #it would be nice to generate data from POIM
                #seq = alphabet[j]*num_pos
                #seq = alphabet[j]*num_pos + rand_seq(alphabet, 50)
                seq = rand_seq(alphabet, num_pos)
                label = choice(candidate_labels)
                
                #create instance object
                instance = Instance(dataset=dataset, example=seq, label=label)            
                instances.append(instance)

        #create splitset
        split_set = dataset.create_split_set(num_splits, size_testset=num_test, random=False)
        multi_split_set.addSplitSet(split_set)

    return multi_split_set


def prepare_multi_datasets_old(organisms, total_size, size_testset, subsample, comment=""):
    """
    loads dataset and stores it in database
    """


    data = load_data(organisms, total_size, subsample, load_test=True)
    
    prepare_multi_datasets(organisms, data, size_testset, comment)



def prepare_multi_datasets(data, fraction_test_set, num_splits=4, description="", feature_type="string", write_db=False, random=False):
    """
    insert data structure into database
    
    @param data: data[task_id]["XT" or "LT"] --> list of examples or labels
    @type data: dict<str, dict<str, list>> 
    @param size_testset: fraction of datapoints to be assigned to test set [0.0-1.0] 
    @type size_testset: float
    @num_splits: number of splits 
    @type size_testset: int
    @param description: comment as dataset description
    @type description: str

    @return: MutiSplitSet which was written into database
    @rtype: MultiSplitSet
    """

    # perform a few sanity checks
    assert(type(fraction_test_set)==float)
    assert(fraction_test_set > 0.01)
    assert(fraction_test_set < 0.99)
    

    if write_db:
        multi_split_set = MultiSplitSet(feature_type=feature_type, description=description)
    
        print "mss_id:", multi_split_set.id


    for (task_name, dataset) in data.items():
        
        print "task name:", task_name
       
        assert(len(dataset["LT"])==len(dataset["XT"]))

        if write_db:
            # create new dataset in db
            dataset_db = Dataset(organism=task_name, signal="", comment="")
        
            print "dataset_id:", dataset_db.id
     
        # add instances
        for (i,example) in enumerate(dataset["XT"]):
        
            label = float(dataset["LT"][i])
            
            if label == 0:
                label = -1
            
            if feature_type=="string":
                tmp_example = str(example)
            else:
                tmp_example = example  
     
            if write_db:
                Instance(dataset=dataset_db, example=tmp_example, label=label)

        
        # get number of test examples
        size_testset = int(float(len(dataset["LT"]))*fraction_test_set)

        if write_db:
            # create splitset
            split_set = dataset_db.create_split_set(num_splits, size_testset=size_testset, random=random)
        
            multi_split_set.addSplitSet(split_set)


    if write_db:
        return multi_split_set



def prepare_multi_datasets_uniform(data, num_splits=5, description="", feature_type="string", write_db=False):
    """
    insert data structure into database
    
    @param data: data[task_id]["XT" or "LT"] --> list of examples or labels
    @type data: dict<str, dict<str, list>> 
    @param size_testset: fraction of datapoints to be assigned to test set [0.0-1.0] 
    @type size_testset: float
    @num_splits: number of splits 
    @type size_testset: int
    @param description: comment as dataset description
    @type description: str

    @return: MutiSplitSet which was written into database
    @rtype: MultiSplitSet
    """


    if write_db:
        multi_split_set = MultiSplitSet(feature_type=feature_type, description=description)
    
        print "mss_id:", multi_split_set.id


    for (task_name, dataset) in data.items():
        
        print "task name:", task_name
       
        assert(len(dataset["LT"])==len(dataset["XT"]))

        if write_db:
            # create new dataset in db
            dataset_db = Dataset(organism=task_name, signal="", comment="")
        
            print "dataset_id:", dataset_db.id
     
        # add instances
        for (i,example) in enumerate(dataset["XT"]):
        
            label = float(dataset["LT"][i])
            
            if label == 0:
                label = -1
            
            if feature_type=="string":
                tmp_example = str(example)
            else:
                tmp_example = example  
     
            if write_db:
                Instance(dataset=dataset_db, example=tmp_example, label=label)

        

        if write_db:
            # create splitset
            split_set = dataset_db.create_split_set(num_splits, random=False)
        
            multi_split_set.addSplitSet(split_set)


    if write_db:
        return multi_split_set



def add_mhc_benchmark_data(test_split_id, write_db=False):
    """
    insert data structure into database


    @return: MutiSplitSet which was written into database
    @rtype: MultiSplitSet
    """

        
    import pickle
    from expenv import SplitSet, Split

    tmp_data = pickle.load(file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/iedb_benchmark/iedb_data_pca.pickle"))
    
    num_splits = 5 
    assert(test_split_id < num_splits)

    #    for (task_name, dataset) in tmp_data.items():
    #        for split_id in range(5):                
    #            for i in range(len(dataset.peptides[split_id])):                
    #                instance = helper.Options()
    #                instance.example = dataset.peptides[split_id][i]
    #                instance.label = dataset.class_labels[split_id][i]
    #                
    #                self.data[task_name][split_id].append(instance)
                    
                    
    if write_db:
        #multi_split_set = MultiSplitSet(feature_type="string", description="MHC benchmark test_split_id=" + str(test_split_id))
        multi_split_set = MultiSplitSet(feature_type="string", description="MHC benchmark SUBSET4")
    
        print "mss_id:", multi_split_set.id


    for (task_name, dataset) in tmp_data.items():
        
    
        active_tasks = ["A_0202", "A_0203", "A_6901", "A_0201", "A_2403", "A_2301", "A_2402"]
        
        if not task_name in active_tasks:
            continue
        
        print "task name:", task_name

        if write_db:
            # create new dataset in db
            dataset_db = Dataset(organism=task_name, signal="", comment="")
            split_set = SplitSet(dataset=dataset_db, num_instances=0)
            
            print "dataset_id:", dataset_db.id

        total_num_instances = 0
        
        for split_id in range(num_splits):
            
            num_examples_split = len(dataset.peptides[split_id])
            total_num_instances += num_examples_split
            
            if write_db:
                #set testset flag for first split
                            
                split = Split(is_test_set=(split_id==test_split_id), split_set = split_set, num=split_id, num_instances=num_examples_split)
            
            
            for i in range(num_examples_split):

                label = float(dataset.class_labels[split_id][i])
                example = str(dataset.peptides[split_id][i])
                  
                if write_db:
                    Instance(dataset=dataset_db, example=example, label=label, split=split)

            if write_db:
                # add number, after we've counted 
                split_set.num_instances = total_num_instances

        if write_db:
        
            multi_split_set.addSplitSet(split_set)


    if write_db:
        return multi_split_set




def split_fasta(in_file_name, length=500, max_num_N=20, min_number_seqs=10, write_db=False):


    f_in = file(in_file_name)


    #>EP17001 (+) Pv snRNA U1; range -9999 to 6000.
    center_pos = 10000

    #padding to negative examples
    space_left = 500
    space_right = 500

    idx_begin = center_pos - length/2
    idx_end = center_pos + length/2
    
   
    tmp_seq = ""
    org_id = ""
    
    
    org_to_sequence = defaultdict(list)


    ##########################
    # parse fasta
    ##########################

    for line in f_in:
    
        if line.startswith(">"):
            
            if org_id != "":
                org_to_sequence[org_id].append(tmp_seq)
                tmp_seq = ""
            
            #fetch org id, remove brackets
            org_id = re.sub("\[.+\]", "", line.split(" ")[2]).replace(";","")

        else:
            
            tmp_seq += line[:-1]                
    
        
 
    counter = defaultdict(int)
    num_Ns_list = []


    #store final sequences    
    positives_all = defaultdict(list)
    negatives_all = defaultdict(list)


    ##########################
    # extract examples
    ##########################
    
    for (org_id, tmp_seqs) in org_to_sequence.items():


        if len(tmp_seqs) < min_number_seqs or org_id=="Bm":
            continue


        for tmp_seq in tmp_seqs:

            prefix = tmp_seq[0:idx_begin-space_left]
            infix = tmp_seq[idx_begin:idx_end]
            suffix = tmp_seq[idx_end+space_right:]
   
            num_Ns_list.append(infix.count("N"))
            num_Ns = infix.count("N")

            #we allow for certain number of Ns
            if num_Ns <= max_num_N:

                #print "org_id:", org_id, ", number of Ns:", num_Ns

                #replace Ns
                infix = infix.replace("N", "A")

                counter[org_id] += 1
                num_Ns_list.append(num_Ns)

                positives_all[org_id].append(infix)


            #try about 100 times (upstream)
            for j in xrange(100):
          
                idx = random.randint(0, len(prefix)-length)
                
                negative = prefix[idx:idx+length]
                
                if negative.count("N") <= max_num_N/2:
                    negative = negative.replace("N", "A")
                    negatives_all[org_id].append(negative)
                    break

                
            #try about 100 times (downstream)
            for j in xrange(100):
          
                idx = random.randint(0, len(suffix)-length)
                
                negative = suffix[idx:idx+length]
                
                if negative.count("N") <= max_num_N/2:
                    negative = negative.replace("N", "A")
                    negatives_all[org_id].append(negative)
                    break

        print org_id, "\tp:", len(positives_all[org_id]), "   \tn:", len(negatives_all[org_id])



    chars = set()

    print "positives"
    for (org_id, seqs) in positives_all.items():
        for seq in seqs:

            for c in seq:
                if not chars.issuperset(c):
                    print org_id, "adding", c
                    chars.add(c)

    print chars

    chars = set()

    print "negatives:"
    for (org_id, seqs) in negatives_all.items():

        for seq in seqs:

            for c in seq:
                if not chars.issuperset(c):
                    print org_id, "adding", c
                    chars.add(c)

    print chars
    ##########################
    # write to database
    ##########################

    if write_db:
        multi_split_set = MultiSplitSet()
        print "creating MultiSplitSet:", multi_split_set

        for (org_id, pos_sequences) in positives_all.items():

            neg_sequences = negatives_all[org_id]

            if len(pos_sequences) >= 9 and len(neg_sequences) >= 9 and len(pos_sequences) <= 1000 and len(neg_sequences) <= 1000:

                dataset_db = Dataset(organism=org_id, signal="TSS")
                print "creating dataset:", dataset_db

                
                print "creating pos instances:", len(pos_sequences)
                for seq in pos_sequences:
                    instance = Instance(dataset=dataset_db, example=str(seq), label=1.0)

                print "creating neg instances:", len(neg_sequences)
                for seq in neg_sequences:
                    instance = Instance(dataset=dataset_db, example=str(seq), label=-1.0)

                #create splitset
                split_set = dataset_db.create_split_set(6, random=True)
                print "creating SplitSet:", split_set

                multi_split_set.addSplitSet(split_set)


            else:
                print "warning: too few sequences for organisms", org_id


        return multi_split_set


    return (counter, num_Ns_list)




def optimize_pssm_by_sampling(length=10, entropy=1.85, epsilon=0.001, sigma=0.01):
    """
    
    compute pssm with desired entropy by a simple hillclimbing algorithm
    """
    
    
    contraints_active = True
    

    keeper = PSSM(["a", "g", "c", "t"], length)


    difference_entropy = 10000000.0
    
    
    while contraints_active:
        
        tmp_pssm = copy.deepcopy(keeper)
    
        # takes already care of respecting boundaries
        tmp_pssm.mutate(sigma)
        
        # compute entropy
        #tmp_entropy = tmp_pssm.compute_entropy(normalize=True)
        tmp_entropy = tmp_pssm.compute_information_content(normalize=False)
        
    
    
        # if we get closer to desired area, we keep the new one
        if numpy.abs(tmp_entropy - entropy) < difference_entropy:
             
            difference_entropy = numpy.abs(tmp_entropy - entropy) 
        
            # keep current pssm
            keeper = tmp_pssm

        
        # checks if contraints are active        
        if tmp_entropy > entropy-epsilon and \
            tmp_entropy < entropy+epsilon:
            
            contraints_active = False
            
            print "entropy:", tmp_entropy
            print "delta entropy:", difference_entropy
            print "--------"

            print keeper
 
    
    
    return tmp_pssm



def optimize_mutation_by_sampling(parent_pssm=None, entropy=1.85, relative_entropy=0.04, epsilon=0.001, sigma=0.01):
    """
    
    compute optimal mutation by a simple hill-climbing algorithm
    
    @param paremt_pssm:
    @param entropy:
    @param relative_entropy:
    @param epsilon:
    
    """
    
    
    contraints_active = True
    

    keeper = copy.deepcopy(parent_pssm)


    difference_entropy = 10000000.0
    difference_relative_entropy = 10000000.0
    
    
    steps = 0
    
    while contraints_active and steps < 1000000:
        
        steps += 1
        
        tmp_pssm = copy.deepcopy(keeper)
    
        # takes already care of boundary contraints
        tmp_pssm.mutate(sigma)
        
        # compute entropy and relative entropy
        #tmp_relative_entropy = tmp_pssm.compute_relative_entropy(parent_pssm, normalize=True)
        tmp_relative_entropy = tmp_pssm.compute_relative_entropy(parent_pssm, normalize=False)
        tmp_entropy = tmp_pssm.compute_information_content(normalize=False)
    

        # if we get closer to desired area, we keep the new one (hillclimber)
        if numpy.abs(tmp_entropy - entropy) < difference_entropy and \
            numpy.abs(tmp_relative_entropy - relative_entropy) < difference_relative_entropy:
            
            
            print "entropy:", tmp_entropy
            print "delta entropy:", difference_entropy
            print "relative entropy:", tmp_relative_entropy
            print "delta relative entropy:", difference_relative_entropy
            print "--------"
            sys.stdout.flush()
            
            difference_entropy = numpy.abs(tmp_entropy - entropy)
            difference_relative_entropy = numpy.abs(tmp_relative_entropy - relative_entropy) 
        
            keeper = tmp_pssm

        
        # checks if contraints are active        
        if tmp_relative_entropy > relative_entropy-epsilon and \
            tmp_relative_entropy < relative_entropy+epsilon and \
            tmp_entropy > entropy-epsilon and \
            tmp_entropy < entropy+epsilon:
            
            contraints_active = False
    
    
            print "entropy:", tmp_entropy
            print "delta entropy:", difference_entropy
            print "relative entropy:", tmp_relative_entropy
            print "delta relative entropy:", difference_relative_entropy
            print "--------"
            
            print keeper

        
    
    return tmp_pssm




def copy_test_set(split_set_source, split_set_destination):
    '''
    helper to copy split sets (for instance to ensure the same test data)
    
    @param split_set_source:
    @param split_set_destination:
    '''
    
    
    test_source = [split for split in split_set_source.splits if split.is_test_set==True][0]
    test_dest = [split for split in split_set_destination.splits if split.is_test_set==True][0]

    print "overwriting data from split", test_dest.id, "with data from split", test_source.id 
    
    instances_source = test_source.instances
    instances_dest = test_dest.instances
    
    assert(len(instances_source) == len(instances_dest))
    
    for (j, inst_source) in enumerate(instances_source):
        
        inst_dest = instances_dest[j]
        
        inst_dest.example = inst_source.example
        inst_dest.label = inst_source.label


    print len(instances_source), "data points copied"
    



class BinaryTreePSSMFactory(object):
    """
    PSSM Factory along binary tree
    """

    leaf_counter = 0    
    leaves = None
    nodes = None

    entropy = 0
    relative_entropy = 0
    length = 0

    

    def __init__(self, entropy=1.0, relative_entropy=0.40, length=20, epsilon=0.002):
        """
        constructor 
        
        @param entropy: entropy of positive class (negative class is uniform)
        @type entropy: float
        @param relative_entropy: KL-divergence to parent
        @type relative_entropy: float
        @param length: length of sequence
        @type length: int 
        @param epsilon: maximal deviation from solution
        @type epsilon: float
        """
        
        self.entropy = entropy
        self.relative_entropy = relative_entropy
        self.length = length
        self.epsilon = epsilon
        
        self.sigma = 0.4 / float(length)

        # will be used to store pssms at all levels in binary tree
        self.nodes = defaultdict(list)
    

    
    def sample_PSSMs(self, num_levels, sigma=None):
        """
        creates binary tree with num_levels and 2^n leaves
        
        @param num_levels: number of levels
        @type num_levels: int
        
        @return: pssms at all levels, e.g. pssms[level_num][pssm_num_at_level]
        @rtype: dict<int, list<pssm>>
        """


        print "creating binary tree with", pow(2, num_levels), "tasks"

        # init
        self.leaf_counter = 0
        self.leaves = []

        # this is quite critical for convergence
        if sigma != None:
            self.sigma = sigma

        # fix entropy of parent
        root = optimize_pssm_by_sampling(self.length, entropy=self.entropy, epsilon=self.epsilon, sigma=0.01)

        # recursive call
        self.__create_subtree(root, num_levels-1)
        self.__create_subtree(root, num_levels-1)


        # return root node
        return self.nodes



    def __create_subtree(self, parent_pssm, level):
        """
        recursive call to generate binary tree
        
        @param parent_pssm: parent node
        @type parent_pssm: TreeNode
        @param level: current level
        @type level: int
        """

        pssm = optimize_mutation_by_sampling(parent_pssm, self.entropy, self.relative_entropy, self.epsilon, self.sigma)

        # append pssm for current level
        self.nodes[level].append(pssm)

        if level == 0:
            self.leaves.append(pssm)
            self.leaf_counter += 1

        else:

            self.__create_subtree(pssm, level-1)
            self.__create_subtree(pssm, level-1)




def select_indices(size, fraction_positives, labels):
    '''
    helper to subsample from dataset at certain positive negative ratio
    
    @param size: number of examples to sample
    @type size: int 
    @fraction_positives: real number indicating fraction of positives (e.g. 0.1)
    @type fraction_positives: float
    @param labels: vector of labels
    @type labels: list<int>
    
    @return: list of indices
    @rtype: list<int> 
    '''
    
    # some sanity checks
    assert fraction_positives > 0, "fraction positives has to be greater than 0 not " + str(fraction_positives)
    assert fraction_positives < 1, "fraction positives has to be smaller than 1 not " + str(fraction_positives)
    assert size > 1, "size has to be greater than 1, not " + str(size)
    assert size < len(labels), "size has to be smaller than original dataset (" + str(len(labels)) + ") " + str(size) 
    
    
    # determine how many data points are to be created
    num_positives = int(float(size) * fraction_positives)
    num_negatives = size - num_positives
    
    print "num_positives: ", num_positives
    print "num_negatives: ", num_negatives 
    
    assert num_positives > 0, "should have at least one positive example, not " + str(num_positives)
    assert num_negatives > 0, "should have at least one negative eexample, not " + str(num_negatives)
    
    
    # shuffle data
    idx = range(len(labels))
    idx = numpy.random.permutation(idx)
    
    # use in loop
    keepers=[]
    counter_positives = 0
    counter_negatives = 0
    
    for i in idx:
    
        # positive
        if (labels[i]==1 and counter_positives < num_positives):
            keepers.append(i)
            counter_positives += 1
        
        # negative
        if (labels[i]==-1 and counter_negatives < num_negatives):
            keepers.append(i)
            counter_negatives += 1
        
        # check if we have enough data points
        if (counter_positives + counter_negatives == size):
            break
    
    
    # check for completeness
    if (counter_positives + counter_negatives != size):
        raise Exception("dataset can not be assembled")


    assert sum([-labels[i] for i in keepers if labels[i] == -1])==num_negatives, "num negatives doesn't match"
    assert sum([labels[i] for i in keepers if labels[i] == 1])==num_positives, "num positives doesn't match" 
    
    return keepers



def load_splice_data(num_examples):
    
    mydir = "/fml/ag-raetsch/share/projects/multitask/raw_data/"

    f = file(mydir + "organisms.txt")
    
    organisms = set()
    
    for line in f:
        token = line.split("\t")[0]
        if token != "\n":
            organisms.add(token)

    f.close()

    data = {}

    for org in organisms:
        
        org_fn = mydir + org + "/acc.mat"
        #new_fn = mydir + org + "/acc.bz2"
        
        print "loading:", org_fn
        
        # load mat file
        mat_file = scipy.io.loadmat(org_fn)
    
        all_labels = list(mat_file["LT"])
        
        num_positives = sum(numpy.array(all_labels)==1)
        print "all positives", num_positives
        num_negatives = sum(numpy.array(all_labels)==-1)
        print "all negatives", num_negatives
        
        # subsample to avoid loading all data
        fraction_positives = 1.0 / 100.0
        idx = select_indices(num_examples, fraction_positives, all_labels)
        
        # shuffle
        idx = numpy.random.permutation(idx)
        
        # pick keepers
        labels = numpy.array(all_labels)[idx]
        
        features = []
        #examples
        char_array = mat_file["XT"]
        
        for i in idx:
    
            feature = ""
            
            seq_length = char_array.shape[0]
            
            left = (seq_length / 2) - 80
            right = (seq_length / 2) + 60
            
            assert(left > 0)
            assert(right < seq_length)
            
            
            # concat features
            for j in xrange(left, right):
                feature += chr(char_array[j][i]).upper().replace("N", "A")

            features.append(feature)
            
        num_positives = sum(numpy.array(labels)==1)
        print "num_positives:", num_positives
        print "total:", len(features)
        print "legth seqs:", len(features[0])
        
        tokens = re.findall("(\w+)_(\w+)", org)[0]
        org_key = tokens[0][0].upper() + "." + tokens[1]
        data[org_key] = {"XT": features, "LT": labels}
        
        
    return data


def load_landmine_data(file_name):
    '''
    helper function to load landmine dataset into database
    
    @param file_name: path to file
    '''
    
    data = scipy.io.loadmat(file_name)
    
    features = data["feature"]
    labels = data["label"]
    
    num_tasks = len(features)
    
    print "num tasks:", num_tasks
    
    assert(len(labels) == num_tasks)
    
    ret = defaultdict(dict)
    
    
    for i in xrange(num_tasks):
        
        tmp_examples = features[i]
        tmp_labels = labels[i]
        
        # format name
        task_name = "task_" + "%01d" % (i)
        
        ret[task_name]["XT"] = tmp_examples
        
        ret[task_name]["LT"] = tmp_labels
        
    return ret


def create_landmine_taxonomy():
    '''
    helper to create simple taxonomy for mss
    '''
        
    from task_similarities import TreeNode
    root = TreeNode("root")
    dessert = TreeNode("dessert")
    other = TreeNode("other")
    range_dessert = range(0, 15)
    range_other = range(15, 29)
    root.add_child(dessert)
    root.add_child(other)
    for i in range_dessert:
        task_name = "task_" + "%01d" % (i)
        node = TreeNode(task_name)
        dessert.add_child(node)
    
    
    for i in range_other:
        task_name = "task_" + "%01d" % (i)
        node = TreeNode(task_name)
        other.add_child(node)
    
    
    root.plot(plot_cost=True, plot_B=True)
    import os
    os.system("evince demo.png")

    return root



def load_mhc_benchmark(data, write_db=False):
    """
    insert data structure into database
    
    @return: MutiSplitSet which was written into database
    @rtype: MultiSplitSet
    """


    new_data = defaultdict(dict)


    for (task_name, dataset) in data.items():
        
        new_data[task_name]["XT"] = list(helper.flatten(dataset.peptides))
        new_data[task_name]["LT"] = list(helper.flatten(dataset.class_labels))
        
        
    prepare_multi_datasets_uniform(new_data, num_splits=5, description="mhc benchmark", feature_type="string", write_db="True")
    


def create_mhc_benchmark_taxonomy(tax_fn):
    
    import pickle
    from task_similarities import TreeNode
                                                                                                                                                                       
    a = pickle.load(file(tax_fn))
    id_to_node = {}
                     
    for node in a:
        node_id = node[0]
        id_to_node[node_id] = TreeNode(str(node_id))
        
    for node in a:
        if node[1]!= -1:
            tree_node = id_to_node[node[0]]
            parent_node = id_to_node[node[1]]
            parent_node.add_child(tree_node)

    for node in a:
        if len(node[2])==1:
            id_to_node[node[0]].name = node[2][0].replace("*","_")
    
    root = id_to_node[2]        
    root.name = "root"
    
    return root
    
    


def main():

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs=", ["help", "sizes="])

    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)

    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)

        if o in ("-s", "--sizes"):
            sizes = a.split(",")


    if len(args) != 1:
        print "need exactly one argument"

    split_fasta(args[0])

        
if __name__ == "__main__":

    main()

