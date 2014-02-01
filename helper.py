#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 11.03.2009
@author: Christian Widmer
@summary: This module carries some useful helper functions. 

"""


import sys
import types
import random
import gzip
import bz2
import cPickle
import inspect


from types import *
import re
import numpy
import string, os
from copy import copy, deepcopy


class Struct(object):
    """Class to create 'struct's on the fly.
    Example: o = Struct()
             o.i = 2
             o.x = 'ababab'
             a = Struct({'i':2,'x':'ababab'})
             b = Struct(i=2,x='ababab')
             In all three cases, the result will be the same.
             __str__ method is redefined so that printing the object of this type
             will show all attributes which are not represented as 'instance at 0xXXXXXX'.
             """

    strStyle = "p" # p - print "pretty", s - print as one string

    def __init__(self,*lw,**kw):
        for dict in lw:
            for key in dict.keys():
                setattr(self,key,dict[key])
        for key in kw.keys():
            setattr(self,key,kw[key])

    def __str__(self):
        if self.strStyle == "p":
            return self.strPretty()
        else:
            return self.strDense()

    def __getitem__(self,key):
        try:
            return getattr(self,key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self,key,value):
        setattr(self,key,value)
    
    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self,key):
        try:
            delattr(self,key)
        except AttributeError:
            raise KeyError(key)

    def __iter__(self):
        return self.__dict__.iterkeys()

    def iterkeys(self):
        return self.__dict__.iterkeys()

    def setdefault(self,*l):
        try:
            return getattr(self,l[0])
        except AttributeError:
            if len(l) >= 2:
                setattr(self,l[0],l[1])
                return getattr(self,l[0])
            else:
                raise KeyError(l[0])

    def pop(self,key):
        return self.__dict__.pop(key)

    def update(self,other):
        if isinstance(other,Struct):
            o = other.__dict__
        else:
            o = other
        self.__dict__.update(o)

    def updateOtherMissing(self,other):
        """Update in other keys that are not yet present where"""
        if isinstance(other,Struct):
            o = other.__dict__
        else:
            o = other
        s = self.asDict()
        for (key,value) in s.items():
            o.setdefault(key,value)

    def updateFromOtherExisting(self,other):
        """Update in self keys that already present in self"""
        if isinstance(other,Struct):
            o = other.__dict__
        else:
            o = other
        s = self.asDict()
        for key in list(s.keys()):
            if key in o:
                s[key] = o[key]

    def asDict(self):
        return self.__dict__

    def keys(self):
        return self.__dict__.keys()
    
    def has_key(self,key):
        return self.__dict__.has_key(key)

    def get(self,*l):
        try:
            return getattr(self,*l)
        except AttributeError:
            raise KeyError(l[0])
            
    def strDense(self):
        keys = self.keys()
        keys.sort()
        pairs = []
        for key in keys:
            obj = self.__dict__[key]
            s_obj = str(obj)
            if self.isPrintable(s_obj):
                pairs.append((key,s_obj))
        return 'Struct('+`pairs`+')'

    def __repr__(self):
        return self.__str__()

    def isPrintable(self,reprObj):
        return not re.match("^\<.+ instance at 0x[0-9a-z]+\>$",reprObj)

    def strPretty(self):
        keys = self.keys()
        keys.sort()
        s = '\n'
        for key in keys:
            obj = self.__dict__[key]
            s_obj = str(obj)
            if self.isPrintable(s_obj):
                # add to TAB to all rows of attribute's representation
                lines = s_obj.split('\n')
                s_obj = '\n\t'.join(lines)
                s = s + key + '\t=\t' + s_obj + '\n'
        return s

    def scalars(self):
        """Return dictionary mapping names of "scalar" attributes to values.
        "Scalar" attributes are non-sequence primitive types, such as Int, Float, String, None."""
        r = {}
        for key in self.keys():
            val = self.__dict__(key)
            if type(val) in (NoneType,BooleanType,IntType,LongType,FloatType,StringType):
                r[key] = val
        return r

    def copy(self):
        return copy(self)


class Options(Struct):
    
    def copy(self):
        """Deep copy semantics"""
        return deepcopy(self)

    def keys(self):
        """Will ignore all attributes that start with _"""
        return [ k for k in Struct.keys(self) if not k.startswith("_") ]

    def freeze(self):
        """Make this object read-only"""
        Struct.__setattr__(self,"_is_frozen",True)
        for name in self.keys():
            val = getattr(self,name)
            if isinstance(val,Options):
                val.freeze()

    def unfreeze(self):
        """Make this object mutable again after previous call to freeze()"""
        try:
            Struct.__delattr__(self,"_is_frozen")
        except AttributeError:
            pass
        for name in self.keys():
            val = getattr(self,name)
            if isinstance(val,Options):
                val.unfreeze()

    def __setattr__(self,name,value):
        if getattr(self,"_is_frozen",False):
            raise AttributeError(name)
        else:
            Struct.__setattr__(self,name,value)

    def __delattr__(self,name):
        if getattr(self,"_is_frozen",False):
            raise AttributeError(name)
        else:
            Struct.__delattr__(self,name)


def int2bin(n, count=24):
    """returns the binary of integer n, using count number of digits"""
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def power_set(orignal_list):
    '''
    PowerSet of a List
    
    @param orignal_list: list from which to construct a powerset
    '''
    list_size = len(orignal_list)
    num_sets = 2**list_size
    powerset = []
    # Don't include empty set
    for i in range(num_sets)[1:]:
        subset = []
        binary_digits = list(int2bin(i,list_size))
        list_indices = range(list_size)
        for (bit,index) in zip(binary_digits,list_indices):
            if bit == '1':
                subset.append(orignal_list[index])
        powerset.append(subset)
    return powerset



def flatten(lst):
    """
    flattens nested list
    
    """
    for elem in lst:
        if type(elem) in (tuple, list):
            for i in flatten(elem):
                yield i
        else:
            yield elem


def split_list(mylist, num_parts):
    """
    Takes a list and a desired number of parts
    and returns a partition as a list of lists
    
    @param mylist: the old list to split
    @type mylist: list<object>
    @param num_parts: number of splits
    @type num_parts: int
    """

    newlist = []
    splitsize = 1.0/num_parts*len(mylist)
    for i in range(num_parts):
        newlist.append(mylist[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        
    return newlist



def rand_seq(alphabet, length):
    """
    generates a random sequence of length over alphabet
    
    @param alphabet: alphabet from which to choose characters
    @type alphabet: list<str>
    @param length: length of random string
    @type length: int
    """

    for c in alphabet:
        
        if len(c)>1:
            print "warning: individual tokens of length > 1: " + c

    seq = "".join([random.choice(alphabet) for j in range(length)])

    return seq
    



def save(filename, myobj, compression_format="bz2"):
    """
    save object to file using pickle
    
    @param filename: name of destination file
    @type filename: str

    @param myobj: object to save (has to be pickleable)
    @type myobj: obj

    @param compression_format: either bz2 or gzip
    @type compression_format: str
    """

    try:
        if compression_format == "gzip":
            f = gzip.GzipFile(filename, 'wb')
        else:
            f = bz2.BZ2File(filename, 'wb')

    except IOError, details:
        sys.stderr.write('File ' + filename + ' cannot be written\n')
        sys.stderr.write(details)
        return

    cPickle.dump(myobj, f, protocol=2)
    f.close()



def load(filename, compression_format="bz2"):
    """
    Load from filename using pickle
    
    @param filename: name of file to load from
    @type filename: str

    @param compression_format: either bz2 or gzip
    @type compression_format: str
    """
    
    try:
        if compression_format == "gzip":
            f = gzip.GzipFile(filename, 'rb')
        else:
            f = bz2.BZ2File(filename, 'rb')

    except IOError, details:
        sys.stderr.write('File ' + filename + ' cannot be read\n')
        sys.stderr.write(details)
        return

    myobj = cPickle.load(f)
    f.close()
    return myobj




def get_member_dict(obj, exclude_list=set()):
    """
    get dict of object members
    """
        
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('_') and not inspect.ismethod(value) and not name in set(exclude_list):
            pr[name] = value
    return pr


def get_sqlobject_member_list():
    """
    provides a list of SQLObject specific class members
    (that do not start with _)"
    """

    member_list = ['SelectResultsClass', 'childName', 'q', 'dirty', 'id', 'sqlmeta', "taxonomyID", "taxonomy"]
 
    return member_list



def calcprc(output, labels):
    """
    calculates PRC

    @param output: svm output
    @type output: list of doubles
    @param labels: true labels
    @type labels: list of doubles
    """


    output = numpy.array(output)
    labels = numpy.array(labels)

    mapping = numpy.double(numpy.array(labels)==1)

    idx = numpy.argsort(-output)
    #s = output[idx]
    mapping = mapping[idx]

    tp = numpy.cumsum(mapping)/numpy.double(sum(labels==1))
    tdr = numpy.cumsum(mapping)/(numpy.double(range(len(labels)))+1.)

    t = tp[1:] - tp[0:-1]
    score = sum(0.5 * (tdr[0:-1] + tdr[1:]) * t)

    if numpy.isnan(score):
        score = 0.0

    return (float(score), tp, tdr)



def calcroc(predout, labels, n = None, targetClass = 1, normalize = True) :
    """returns the true positive rate and the false positive rate (the ROC curve),
    and also the area under the curve

    Parameters:
    - predout - the values of the prediction output
    - labels - the true labels
    - n - the number of false positives to take into account (roc_n)
    - targetClass - the positive class (default = 1)
    - normalize whether to normalize the roc curve (default: True)
      when this is set to False, TP/FP counts are output rather than TP/FP rates
    """
    from numpy import random, sum, equal, not_equal, array, argsort

    if n is not None and n < 1 :
        n = int(n * sum(not_equal(labels, targetClass)))
    I = range(len(predout))
    random.shuffle(I)
    predout = [predout[i] for i in I]
    labels = [labels[i] for i in I]
    f = array(predout)

    tp = [0.0]
    fp = [0.0]
    I = argsort(-f)

    for patternIdx in I :
        if labels[patternIdx] == targetClass :
            tp[-1] += 1
        else :
            tp.append(tp[-1])
            fp.append(fp[-1] + 1.0)
        if n is not None and fp[-1] >= n :
            break

    numTP = sum(equal(labels, targetClass))

    if normalize :
        for i in range(len(tp)):
            #if tp[-1] > 0 : tp[i] /= float(tp[-1])
            if tp[-1] > 0 : tp[i] /= float(numTP)
        for i in range(len(fp)) :
            if fp[-1] > 0 : fp[i] /= float(fp[-1])

        area = sum(tp) / len(tp)

    else :
        area = sum(tp) / (len(tp) * numTP)

    return (float(area), tp, fp)
 


def plot_roc_curve(out, labels):
    """
    plot roc curve using pylab
    """

    import pylab

    roc_auc, tpr, fpr = calcroc(out, labels)

    print "Area under the ROC curve : %f" % roc_auc

    # Plot ROC curve
    pylab.figure(-1)
    pylab.clf()
    pylab.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.xlim([0.0,1.0])
    pylab.ylim([0.0,1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('Receiver operating characteristic example')
    pylab.legend(loc="lower right")
    pylab.show()


 
def get_refcounts():
    d = {}
    sys.modules
    # collect all classes
    for m in sys.modules.values():
        for sym in dir(m):
            o = getattr (m, sym)
            if type(o) is types.ClassType:
                d[o] = sys.getrefcount (o)
    # sort by refcount
    pairs = map (lambda x: (x[1],x[0]), d.items())
    pairs.sort()
    pairs.reverse()
    return pairs


def print_top_100():
    for n, c in get_refcounts()[:100]:
        print '%10d %s' % (n, c.__name__)



#############################
#TODO: move this to data class eventually 
 

def split_data(train_data, FOLD):
    '''
    
    split data in two sets, one with union of (n-1) folds and one with examples from one fold
    
    @param train_data: training data
    @type train_data:  
    @param FOLD: number of folds to split
    @type FOLD: int
    
    @return two dicts in same format of input, split according to FOLD
    @rtype: tuple< dict<task_name, list<instances> >, dict<task_name, list<instances> > >  
    '''

    
    numpy.random.seed(1)
    
    inner_train_data = {}
    inner_eval_data = {}
    
    for task_id in train_data.keys():
        
        idx = range(len(train_data[task_id]))
        
        idx_pos = [idx for idx in range(len(train_data[task_id])) if train_data[task_id][idx].label == 1]
        idx_neg = [idx for idx in range(len(train_data[task_id])) if train_data[task_id][idx].label == -1]
        
        numpy.random.shuffle(idx_pos)
        numpy.random.shuffle(idx_neg)
        
        splits_pos = split_list(idx_pos, FOLD)
        splits_neg = split_list(idx_neg, FOLD)
    
        eval_split_id = 0
        train_idx_pos = list(flatten([splits_pos[j] for j in xrange(FOLD) if j!=eval_split_id]))
        train_idx_neg = list(flatten([splits_neg[j] for j in xrange(FOLD) if j!=eval_split_id]))
        
        train_idx = train_idx_pos
        train_idx.extend(train_idx_neg)
        numpy.random.shuffle(train_idx)
        
        
        eval_idx_pos = splits_pos[eval_split_id]
        eval_idx_neg = splits_neg[eval_split_id]
        
        eval_idx = eval_idx_pos
        eval_idx.extend(eval_idx_neg)
        
        numpy.random.shuffle(eval_idx)
        
        # make sure idx lists are disjoint
        assert(len(set(train_idx).intersection(set(eval_idx))) == 0)
        
        # select data sets
        inner_train_data[task_id] = numpy.array(train_data[task_id])[train_idx].tolist()
        inner_eval_data[task_id] = numpy.array(train_data[task_id])[eval_idx].tolist()


    return (inner_train_data, inner_eval_data)



def gen_features(examples):
    """
    computes vector representation of strings
    
    @param examples: sequence examples
    @type examples: list<str>
    
    @return: examples in explicit feature representation
    @rtype: array<array<float>>
    """
    
    import numpy
    
    val = 0.08421519
    #val = 1
    
    ret = numpy.zeros((len(examples),len(examples[0])*4))
    
    for (i,example) in enumerate(examples):
        
        for (j,c) in enumerate(example.upper()):
            
            idx = j*4
            
            if c=="A":
                ret[i][idx] = val
            elif c=="C":
                ret[i][idx+1] = val
            elif c=="G":
                ret[i][idx+2] = val
            elif c=="T":
                ret[i][idx+3] = val              
        
        #ret[i] = ret[i]/(len(example))
        
    return ret



class SequencesHandler(object):
    '''
    Convenience class to keep track of mapping task_name, task_id.
    
    Provides support to nicely unify data form different tasks.
    '''
       
    
    
    def __init__(self):
        '''
        loads data into handler
        '''
        
        fn = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHCsequenzen/pseudo.txt"
        
        tmp_key = ""
        
        self.seqs = {}
        self.seq_length = 0
        
        # parse file
        for line in file(fn):
            
            if line.startswith(">"):
                tmp_key = line.strip()[1:]
            else:
                self.seqs[tmp_key] = line.strip()
                self.seq_length = len(self.seqs[tmp_key])
        
        #print self.seqs.keys()
                
                
                
    def get_similarity(self, task_name_lhs, task_name_rhs, pos):
        '''
        computes position specific similarities between task pseudo-sequences
        
        @param task_name_lhs: name of task on left hand side
        @param task_name_rhs: name of task on right hand side
        @param pos: position to take into account
        
        @return: kronecker delta
        '''
        
        seq_lhs = self.seqs[task_name_lhs]
        seq_rhs = self.seqs[task_name_rhs]
        
        if seq_lhs[pos] == seq_rhs[pos]:
            return 1
        else:
            return 0 



def assess(out, labels, target):
    """
    simple wrapper for performance determination
    """

    # return performance measure
    if target=="auPRC":
        return calcprc(out, labels)[0]
    elif target=="auROC":
        return calcroc(out, labels)[0]
    else:
        assert(False), "unknown measure type"



def find_in_list(mylist, element):
    """find position of element in list (same as string.find)"""
  
    for idx, list_element in enumerate(mylist):
        if list_element == element:
            return idx
        
    return -1
    

        
def sanitize_sequence(seq, verbose=True):
    """
    sanitizes a piece of DNA sequence
    """

    seq_upper = seq.upper()

    accepted_seq = ("A", "C", "G", "T")

    for seq_char in seq_upper:
        if not seq_char in accepted_seq:
            seq_upper = seq_upper.replace(seq_char, "A")
            
            if verbose:
                print "warning, replacing %s with A" % (seq_char)
            
        
    return seq_upper


    

def product(*args, **kwds):
    """
    product from itertools
    """
    
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def iter_grid(param_grid):
    """Generators on the combination of the various parameter lists given

    Parameters
    -----------
    kwargs: keyword arguments, lists
        Each keyword argument must be a list of values that should
        be explored.

    Returns
    --------
    params: dictionary
        Dictionnary with the input parameters taking the various
        values succesively.

    Examples
    ---------
    >>> from scikits.learn.grid_search import iter_grid
    >>> param_grid = {'a':[1, 2], 'b':[True, False]}
    >>> list(iter_grid(param_grid))
    [{'a': 1, 'b': True}, {'a': 1, 'b': False}, {'a': 2, 'b': True}, {'a': 2, 'b': False}]

    """
    if hasattr(param_grid, 'has_key'):
        param_grid = [param_grid]
    for p in param_grid:
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(p.items())
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params



class SimpleLabeledArray(object):
    """
    simple labeled array, that keeps track of row and col labels

    THIS IS OBSOLETE, USE PANDAS FOR STUFF LIKE THIS
    """
    
    def __init__(self, data, row_labels, col_labels=None):
        """
        constructor
        """
        
        if not col_labels:
            col_labels = row_labels
        
        assert(data.shape[0] == len(row_labels))
        assert(data.shape[1] == len(col_labels))
        
        self.data = data
        self.row_labels = row_labels
        self.col_labels = col_labels
        
        self.row_label_to_idx = dict([(label, i) for (i, label) in enumerate(row_labels)])
        self.col_label_to_idx = dict([(label, i) for (i, label) in enumerate(col_labels)])
    
    
    def get_row(self, row_label):
        """
        gets values by label
        """
        
        row_idx = self.row_label_to_idx[row_label]
        
        return self.data[row_idx,:]


    def get_col(self, col_label):
        """
        gets values by label
        """
        
        col_idx = self.col_label_to_idx[col_label]
        
        return self.data[:,col_idx]
        
        
    def get_value(self, row_label, col_label):
        """
        gets value by label
        """
        
        row_idx = self.row_label_to_idx[row_label]
        col_idx = self.col_label_to_idx[col_label]
        
        return self.data[row_idx, col_idx]
    
    
    def set_value(self, row_label, col_label, value):
        """
        sets value by label
        """
        
        row_idx = self.row_label_to_idx[row_label]
        col_idx = self.col_label_to_idx[col_label]
        
        self.data[row_idx, col_idx] = value
   

def assert_symdiff_empty(lhs, rhs):
    """
    check if symmetric difference is empty
    """

    symm_diff = set(lhs).symmetric_difference(set(rhs))
    assert len(symm_diff) == 0, symm_diff


def assert_intersetion_empty(lhs, rhs):
    """
    check if symmetric difference is empty
    """

    intersection = set(lhs).intersection(set(rhs))
    assert len(intersection) == 0, "intersection not empty: %s" % (str(intersection))


def coshuffle(*args):
    """
    will shuffle target_list and apply
    same permutation to other lists

    >>> helper.coshuffle([2, 1, 3], [4, 2, 8], [6, 3, 12])
    ([5, 3, 2, 1, 4], [5, 3, 2, 1, 4], [5, 3, 2, 1, 4])
    """ 

    assert len(args) > 0, "need at least one list"

    num_elements = len(args[0])

    for arg in args:
        assert len(arg) == num_elements, "length mismatch"

    idx = range(num_elements)
    random.shuffle(idx)

    new_lists = []

    for arg in args:
        new_lists.append([arg[i] for i in idx])

    return tuple(new_lists)



def file_len(fname):
    """
    determine num of lines in a file
    """
    
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


