#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2010-2011 Christian Widmer
# Copyright (C) 2010-2011 Max-Planck-Society

"""
module to create shogun data objects according to given parameters
"""

import shogun
from shogun.Classifier import SVMLight, LibLinear, L2R_LR, L2R_L2LOSS_SVC_DUAL, DomainAdaptationSVM, DomainAdaptationSVMLinear
from shogun.Kernel import WeightedDegreeStringKernel, LinearKernel, PolyKernel, GaussianKernel, CTaxonomy
from shogun.Kernel import CombinedKernel, WeightedDegreeRBFKernel
from shogun.Features import StringCharFeatures, RealFeatures, CombinedFeatures, StringWordFeatures
from shogun.Features import DNA, PROTEIN, Labels
from shogun.PreProc import SortWordString
from shogun.Kernel import WeightedDegreeStringKernel, CombinedKernel, WeightedCommWordStringKernel, WeightedDegreePositionStringKernel
from shogun.Features import StringCharFeatures, DNA, StringWordFeatures, CombinedFeatures
from shogun.Features import CombinedDotFeatures, HashedWDFeatures, HashedWDFeaturesTransposed, WDFeatures, ImplicitWeightedSpecFeatures, StringByteFeatures
from shogun.PreProc import SortWordString


import cPickle
import numpy


#TODO: refactor this into a class and inherit in art2_factory



def create_taxonomy(taxonomy):
    """
    creates shogun taxonomy from python taxonomy
    
    @param taxonomy: taxonomy encoded in python (root node)
    @type taxonomy: TreeNode
    
    @return: wrapped C++ taxonomy object
    @rtype: shogun::taxonomy 
    """

    shogun_taxonomy = CTaxonomy()
    
    
    grey_nodes = taxonomy.children
    
    sum_edge_weights = sum([float(node.edge_weight) for node in taxonomy.get_all_nodes()])
    
    tmp_root_weight = 1.0
    
    
    # add one if root is set to zero by default
    if taxonomy.edge_weight == 0:
        print "adding one because root edge weight is set to zero"
        sum_edge_weights += tmp_root_weight
    else:
        tmp_root_weight = taxonomy.edge_weight
        
        
    shogun_taxonomy.set_root_beta(tmp_root_weight / sum_edge_weights)
    #shogun_taxonomy.set_root_beta(1.0)
    
    
    while len(grey_nodes)>0:

        node = grey_nodes.pop(0)
        
        # append children
        if not node.is_leaf():
            grey_nodes.extend(node.children)
    
        normalized_weight = float(node.edge_weight) / sum_edge_weights 
        print "inserting:", node.parent.name, ",", node.name, ",", normalized_weight
        # insert node
        shogun_taxonomy.add_node(node.parent.name, node.name, normalized_weight)   
        #shogun_taxonomy.add_node(node.parent.name, node.name, 0.0)


    return shogun_taxonomy



def create_initialized_domain_adaptation_svm(param, examples, labels, presvm, weight):
    """
    create kernel/featues and initialize svm with it
    """

    # create data object (either kernel or features)    
    if param.flags.has_key("svm_type") and param.flags["svm_type"] == "liblineardual":
        data = create_features(examples, param)
    else:
        # create shogun data objects
        data = create_kernel(examples, param)

    lab = create_labels(labels)

    # create svm object
    svm = create_domain_adaptation_svm(param, data, lab, presvm, weight)

    return svm



def create_domain_adaptation_svm(param, k, lab, presvm, weight):
    '''
    create SVM object with standard settings
    
    @param param: parameter object
    @param k: kernel
    @param lab: label object
    
    @return: svm object
    '''


    # create SVM
    if param.flags.has_key("svm_type") and param.flags["svm_type"] == "liblineardual":
        svm = DomainAdaptationSVMLinear(param.cost, k, lab, presvm, weight)
    else:
        svm = DomainAdaptationSVM(param.cost, k, lab, presvm, weight)
    
    return set_svm_parameters(svm, param)



def create_initialized_svm(param, examples, labels):
    """
    create kernel/featues and initialize svm with it
    """

    # create data object (either kernel or features)    
    if param.flags.has_key("svm_type") and param.flags["svm_type"] == "liblineardual":
        data = create_features(examples, param)
    else:
        # create shogun data objects
        data = create_kernel(examples, param)

    lab = create_labels(labels)

    # create svm object
    svm = create_svm(param, data, lab)

    return svm


def create_svm(param, data, lab):
    """
    create SVM object with standard settings
    
    @param param: parameter object
    @param data: kernel or feature object (for kernelized/linear svm)
    @param lab: label object
    
    @return: svm object
    """


    # create SVM
    if param.flags.has_key("svm_type") and param.flags["svm_type"] == "liblineardual":
        print "creating LibLinear object"
        svm = LibLinear(param.cost, data, lab)
        svm.set_liblinear_solver_type(L2R_L2LOSS_SVC_DUAL)

        # set solver type
        if param.flags.has_key("solver_type") and param.flags["solver_type"] == "L2R_LR":
            print "setting linear solver type to: L2R_LR" 
            svm.set_liblinear_solver_type(L2R_LR)

    else:
        print "creating SVMLight object"
        svm = SVMLight(param.cost, data, lab)

    
    return set_svm_parameters(svm, param)



def set_svm_parameters(svm, param):
    """
    set svm paramerers based on param object (same for all svms)
    """

    lab = svm.get_labels()

    # set cost
    if param.flags["normalize_cost"]:
        
        norm_c_pos = param.cost / float(len([l for l in lab.get_labels() if l==1]))
        norm_c_neg = param.cost / float(len([l for l in lab.get_labels() if l==-1]))
        svm.set_C(norm_c_neg, norm_c_pos)
        
    else:
                
        svm.set_C(param.cost, param.cost)

    # set epsilon
    if param.flags.has_key("epsilon"):
        svm.set_epsilon(param.flags["epsilon"])

    # show debugging output
    if not param.flags.has_key("debug_off"):
        svm.io.enable_progress()
        svm.io.set_loglevel(shogun.Classifier.MSG_DEBUG)
    
    # optimization settings
    if param.flags.has_key("threads"):    
        svm.parallel.set_num_threads(param.flags["threads"])

    # set bias
    if param.flags.has_key("use_bias"):
        svm.set_bias_enabled(param.flags["use_bias"])
    
    return svm



def create_empty_kernel(param):
    """
    kernel factory
    
    @param param: parameter object
    @type param: Parameter
    
    @return subclass of shogun Kernel object
    @rtype: Kernel
    """

    
    kernel = None
    

    if param.kernel == "WeightedDegreeStringKernel":        
        kernel = WeightedDegreeStringKernel(param.wdk_degree)
        
        
    elif param.kernel == "LinearKernel":
        kernel = LinearKernel()
        
    
    elif param.kernel == "PolyKernel":
        kernel = PolyKernel(10, 1, False)        
        
        
    elif param.kernel == "GaussianKernel":
        kernel = GaussianKernel(10, param.sigma)
    
    elif param.kernel == "WeightedDegreeRBFKernel":
        size_cache = 50
        nof_properties = 5 #20
        sigma = param.transform
        kernel = WeightedDegreeRBFKernel(size_cache, sigma, param.wdk_degree, nof_properties)
     
    
    else:
        
        raise Exception, "Unknown kernel type:" + param.kernel
    
    if hasattr(param, "flags") and param.flags.has_key("cache_size"):
        kernel.set_cache_size(param.flags["cache_size"])

    if param.flags.has_key("debug"):    
        kernel.io.set_loglevel(shogun.Kernel.MSG_DEBUG)
    
    return kernel


    

def create_kernel(examples, param):
    """
    kernel factory
    
    @param examples: list/array of examples
    @type examples: list
    @param param: parameter object
    @type param: Parameter
    
    @return subclass of shogun Kernel object
    @rtype: Kernel
    """


    # first create feature object of correct type
    feat = create_features(examples, param)
    
    
    kernel = None
    

    if param.kernel == "WeightedDegreeStringKernel":        
        kernel = WeightedDegreeStringKernel(feat, feat, param.wdk_degree)
        kernel.set_cache_size(200)
        
        
    elif param.kernel == "LinearKernel":
        kernel = LinearKernel(feat, feat)
        
    
    elif param.kernel == "PolyKernel":
        kernel = PolyKernel(feat, feat, 1, False)        
        
        
    elif param.kernel == "GaussianKernel":
        kernel = GaussianKernel(feat, feat, param.sigma)
    
    
    elif param.kernel == "WeightedDegreeRBFKernel":
        size_cache = 200
        nof_properties = 20
        sigma = param.base_similarity
        kernel = WeightedDegreeRBFKernel(feat, feat, sigma, param.wdk_degree, nof_properties, size_cache)

    elif param.kernel == "Promoter":
        kernel = create_promoter_kernel(examples, param.flags)

    
    else:
        raise Exception, "Unknown kernel type."
    
    
    if hasattr(param, "flags") and param.flags.has_key("cache_size"):
        kernel.set_cache_size(param.flags["cache_size"])
    
    if param.flags.has_key("debug"):    
        kernel.io.set_loglevel(shogun.Kernel.MSG_DEBUG)
    
    return kernel



def encode_peptide(peptide, encoding):
    """
    peptide encoding for WDKRBF
    """

    encoded_peptide = []
    for i in range(len(peptide)):
        encoded_peptide.extend(encoding[peptide[i]])

    return encoded_peptide


#TODO refactor into its own MHC subclass

def create_encoded_features(peptides):
    """
    get feature object for WDKRBF
    """
    source_path = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/"
    
    fid = open(source_path + "BLOSUM50_encoding.cPickle")
    #fid = open(source_path + "pca_encoding.pickle")
    encoding = cPickle.load(fid)
    fid.close()

    #encode peptides
    blosum_encoded_peptides = []
    
    for peptide in peptides:
        blosum_encoded_peptides.append(encode_peptide(peptide, encoding))
        
    feat = RealFeatures(numpy.array(blosum_encoded_peptides).transpose())
    

    return feat

    
def create_features(examples, param):
    """
    factory for features
    
    @param examples: list/array of examples
    @type examples: list

    @return subclass of shogun Features object
    @rtype: Features
    """


    assert(len(examples) > 0)

    feat = None    

    #TODO: refactor!!!    
    if param and param.flags.has_key("svm_type") and param.flags["svm_type"] == "liblineardual":
        # create hashed promoter features
        return create_hashed_promoter_features(examples, param.flags)

    if param and param.kernel == "WeightedDegreeRBFKernel":
        # use BLOSSUM encoding
        return create_encoded_features(examples)

    if param and param.kernel == "Promoter":
        print "creating promoter features"
        # create promoter features
        return create_promoter_features(examples, param.flags)
     
     
    #auto_detect string type
    if type(examples[0]) == str:
        
        # check what alphabet is used
        longstr = ""
        num_seqs = min(len(examples), 20)
        for i in range(num_seqs):
            longstr += examples[i]
            
        if len(set([letter for letter in longstr]))>5:
            feat = StringCharFeatures(PROTEIN)
            if param and param.flags.has_key("debug"):
                print "FEATURES: StringCharFeatures(PROTEIN)"
        else:
            feat = StringCharFeatures(DNA)
            if param and param.flags.has_key("debug"):
                print "FEATURES: StringCharFeatures(DNA)"
            
        feat.set_features(examples)

    else:
        
        # assume real features
        examples = numpy.array(examples, dtype=numpy.float64)
        
        examples = numpy.transpose(examples)
        
        feat = RealFeatures(examples)
       
        if param and param.flags.has_key("debug"):
            print "FEATURES: RealFeatures"
        
    
    return feat
        
        
        
def create_labels(labels):
    """
    create shogun labels
    
    @param labels: list of labels
    @type labels: list<float>
    
    @return: labels in shogun format
    @rtype: Labels
    """
    
    
    lab = Labels(numpy.double(labels))
    
    
    return lab



def get_spectrum_features(data, order=3, gap=0, reverse=True):
    """
    create feature object used by spectrum kernel
    """

    charfeat = StringCharFeatures(data, DNA)
    feat = StringWordFeatures(charfeat.get_alphabet())
    feat.obtain_from_char(charfeat, order-1, order, gap, reverse)
    preproc = SortWordString()                                            
    preproc.init(feat)
    feat.add_preprocessor(preproc)
    feat.apply_preprocessor()

    return feat


    
def get_wd_features(data, feat_type="dna"):
    """
    create feature object for wdk
    """
    
    if feat_type == "dna":
        feat = StringCharFeatures(DNA)
        
    elif feat_type == "protein":
        feat = StringCharFeatures(PROTEIN)
        
    else:
        raise Exception("unknown feature type")
                
    feat.set_features(data)
    
    return feat



def create_combined_wd_features(instances, feat_type):
    """
    creates a combined wd feature object
    """

    num_features = len(instances[0])
    
    # contruct combined features
    feat = CombinedFeatures()
        
    for idx in range(num_features): 
    
        # cut column idx
        data = [instance[idx] for instance in instances]
    
        seq_len = len(data[0])
        for seq in data:
            if len(seq) != seq_len:
                print "warning, seq lengths differ", len(seq), seq_len, "in", idx, "num_feat", num_features
    
        tmp_feat = get_wd_features(data, feat_type)
        feat.append_feature_obj(tmp_feat)

    
    return feat



def create_combined_spectrum_features(instances, feat_type):
    """
    creates a combined spectrum feature object
    """

    num_features = len(instances[0])
    
    # contruct combined features
    feat = CombinedFeatures()
        
    for idx in range(num_features): 
    
        # cut column idx
        data = [instance[idx] for instance in instances]
    
        tmp_feat = get_spectrum_features(data, feat_type)
        feat.append_features(tmp_feat)

    
    return feat



def create_combined_wd_kernel(instances, param):
    """
    creates a combined wd kernel object
    """

    num_features = len(instances[0])

    # contruct combined features
    kernel = CombinedKernel()
    
    for idx in range(num_features): 
        
        param.kernel = "WeightedDegreeStringKernel"
        
        tmp_kernel = create_empty_kernel(param)
        kernel.append_kernel(tmp_kernel)

    combined_features = create_combined_wd_features(instances, feat_type="dna")

    kernel.init(combined_features, combined_features)
    
    return kernel


def create_empty_promoter_kernel(param):
    """
    creates an uninitialized promoter kernel
   
    @param param:
    """


    # centered WDK/WDK-shift
    if param["shifts"] == 0:
        kernel_center = WeightedDegreeStringKernel(param["degree"])
    else:
        kernel_center = WeightedDegreePositionStringKernel(10, param["degree"])
        shifts_vector = numpy.ones(param["center_offset"]*2, dtype=numpy.int32)*param["shifts"]
        kernel_center.set_shifts(shifts_vector)

    kernel_center.set_cache_size(param["kernel_cache"]/3)

    # border spetrum kernels
    size = param["kernel_cache"]/3
    use_sign = False
    kernel_left = WeightedCommWordStringKernel(size, use_sign)
    kernel_right = WeightedCommWordStringKernel(size, use_sign)

    # assemble combined kernel
    kernel = CombinedKernel()
    kernel.append_kernel(kernel_center)
    kernel.append_kernel(kernel_left)
    kernel.append_kernel(kernel_right)


    return kernel



def create_promoter_kernel(examples, param):
    """
    creates a promoter kernel
    
    @param examples:
    @param param:
    """
    
    # create uninitialized kernel
    kernel = create_empty_promoter_kernel(param)

    # get features
    feat = create_promoter_features(examples, param)

    # init combined kernel
    kernel.init(feat, feat)

    return kernel
    
    

def create_promoter_features(data, param):
    """
    creates promoter combined features
    
    @param examples:
    @param param:
    """

    print "creating promoter features"

    (center, left, right) = split_data_promoter(data, param["center_offset"], param["center_pos"])

    # set up base features
    feat_center = StringCharFeatures(DNA)
    feat_center.set_features(center)
    feat_left = get_spectrum_features(left)
    feat_right = get_spectrum_features(right)

    # construct combined features
    feat = CombinedFeatures()
    feat.append_feature_obj(feat_center)
    feat.append_feature_obj(feat_left)
    feat.append_feature_obj(feat_right)

    return feat




def split_data_promoter(data, center_offset, center_pos):
    '''
    split promoter data in three parts
    @param data:
    '''


    center = [seq[(center_pos - center_offset):(center_pos + center_offset)] for seq in data]
    left = [seq[0:center_pos] for seq in data]
    right = [seq[center_pos:] for seq in data]

    #print left, center, right

    return (center, left, right)


########################################################
# linear stuff
########################################################


def create_hashed_promoter_features(data, param):
    """
    creates a promoter feature object
    """

    print "creating __hashed__ promoter features (for linear SVM)"

    (center, left, right) = split_data_promoter(data, param["center_offset"], param["center_pos"])

    # set up base features
    feats_center = create_hashed_features_wdk(param, center)
    feats_left = create_hashed_features_spectrum(param, left)
    feats_right = create_hashed_features_spectrum(param, right)

    # create combined features
    feats = CombinedDotFeatures()
    feats.append_feature_obj(feats_center)
    feats.append_feature_obj(feats_left)
    feats.append_feature_obj(feats_right)

    return feats


def create_hashed_features_wdk(param, data):
    """
    creates hashed dot features for the wdk
    """

    # fix parameters
    start_degree = 0
    hash_bits = 12
    degree = param["degree"]
    order = 1
    gap = 0
    reverse = True

    #print "test", data[0]

    # create raw features
    feats_char = StringCharFeatures(data, DNA)
    feats_raw = StringByteFeatures(DNA)
    feats_raw.obtain_from_char(feats_char, order-1, order, gap, reverse)

    # finish up
    feats = HashedWDFeaturesTransposed(feats_raw, start_degree, degree, degree, hash_bits)
    #feats = HashedWDFeatures(feats_raw, start_degree, degree, degree, hash_bits)
    #feats = WDFeatures(feats_raw, 1, 8)#, degree, hash_bits)

    return feats



def create_hashed_features_spectrum(param, data):
    """
    creates hashed dot features for the spectrum kernel
    """

    # extract parameters
    order = param["degree_spectrum"]

    # fixed parameters
    gap = 0
    reverse = True 
    normalize = True

    # create features
    feats_char = StringCharFeatures(data, DNA)
    feats_word = StringWordFeatures(feats_char.get_alphabet())
    feats_word.obtain_from_char(feats_char, order-1, order, gap, reverse)

    # create preproc
    preproc = SortWordString()
    preproc.init(feats_word)
    feats_word.add_preproc(preproc)
    feats_word.apply_preproc()

    # finish 
    feats = ImplicitWeightedSpecFeatures(feats_word, normalize)

    return feats




