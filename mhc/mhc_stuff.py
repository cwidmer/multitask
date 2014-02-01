#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 25.11.2009
@author: Christian Widmer
@summary: Extracts data structure for MHC stuff 
"""


#import packages
import sys
import getopt
import scipy.io
from task_similarities import TreeNode
from collections import defaultdict
from data_processing import prepare_multi_datasets
from expenv import Taxonomy


def load_data():
    """
    creates graph from adjancancy matrix
    """
    
    data_fn = "/fml/ag-raetsch/home/toussaint/mhc_binding/iedb_9mers.mat"


    d = scipy.io.loadmat(data_fn)
    
    # get task identifiers
    task_ids = [str(name) for name in d["all_alleles"]]

    # fill data stucture, with examples and labels mapped to respective task
    data = defaultdict(dict)
    
    for (i,task_id) in enumerate(task_ids):
        data[task_id]["XT"] = [str(seq) for seq in d["data"][i].textdata]
        data[task_id]["LT"] = [int(seq) for seq in d["class_labels"][i]]

    return data


def write_data_to_database():
    """
    loads data and writes it to database
    """
    
    data = load_data()
    
    return prepare_multi_datasets(data, 0.3, 4, "mhc")


def write_graph_to_database():
    """
    creates taxonomy and writes it to database
    """
    
    root = create_graph()
    taxonomy = Taxonomy(data=root, description="mhc")
    
    return taxonomy


def create_graph():
    """
    creates graph from adjancancy matrix
    """
    
    
    info = "/fml/ag-raetsch/home/raetsch/siebenundvierzig_allele/labels_and_features_pca_47.dat"

    id_to_name = {}
    name_to_id = {}

    for line in file(info):
        
        if line.startswith("#"):
            tokens = line.split(" ")
            idx = int(tokens[2])
            name = tokens[3][1:-1].replace("*", "_")
            
            # store both directions
            id_to_name[idx] = name
            name_to_id[name] = idx
    
    
    ################################
    # construct adjacency matrix
    ################################
    
    
    fn = "/fml/ag-raetsch/home/raetsch/siebenundvierzig_allele/adjacency_matrix_pca_47.dat"
    
    nodes = []
    rows = []
    
    for (line_num, line) in enumerate(file(fn)):
        
        rows.append([int(token) for token in line.split(" ")])
        nodes.append(TreeNode(str(line_num)))
    
    num_nodes = len(nodes)
    
    
    ################################
    # construct adjacency matrix
    ################################
    
    for (row_id, row) in enumerate(rows):
        for col_id in xrange(row_id, num_nodes):
            
            if row[col_id]==1 and row_id!=col_id:
                nodes[col_id].add_child(nodes[row_id])

        
    ################################
    # remove unnecessary nodes
    ################################

    
    fn_35 = "/fml/ag-raetsch/home/toussaint/mhc_binding/iedb_9mers.mat"
    dat = scipy.io.loadmat(fn_35)

    keeper_names = [str(name) for name in dat["all_alleles"]]
    keeper_idx = [name_to_id[name] for name in keeper_names]
  
    
    keep_nodes = []
    
    for idx in keeper_idx:
        node = nodes[idx]
        keep_nodes.append(node)
        keep_nodes.extend(node.get_path_root())
    
    keep_nodes = set(keep_nodes)
    
    for node in nodes:
        if not node in keep_nodes:
            node.parent.children.remove(node)
    
    
    ################################
    # wrap things up
    ################################

                   
    # add name tags to leaves
    for node in keep_nodes:
        if node.is_leaf():
            node.name = id_to_name[int(node.name)]
            
            
    print "keeping", len(keep_nodes), "of",  len(nodes), "nodes" 
  
    root = [node for node in keep_nodes if node.is_root()][0]
    
    root.plot(file_name="/tmp/mhc_graph")
  
    return root


def compute_task_distances_pandas(active_ids):
    

    import numpy
    import pandas
    
    
    # load data
    #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_PseudoSeq_BlosumEnc_pearson.txt")
    f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/All_PseudoSeq_Hamming.txt")
    #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_PseudoSeq_BlosumEnc_euklid.txt")
    #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_RAxML.txt")
    
    num_lines = int(f.readline().strip())
    task_distances = numpy.zeros((num_lines, num_lines))
    name_to_id = {}

    for (i, line) in enumerate(f):
        tokens = line.strip().split("\t")
        name = str(tokens[0])
        name_to_id[name] = i

        entry = numpy.array([v for (j,v) in enumerate(tokens) if j!=0])
        assert len(entry)==num_lines, "len_entry %i, num_lines %i" % (len(entry), num_lines)
        task_distances[i,:] = entry
        
        
    
    # cut relevant submatrix
    tmp_distances = task_distances[active_ids, :]
    tmp_distances = tmp_distances[:, active_ids]
    print "distances ", tmp_distances.shape

    
    # normalize distances
    task_distances = task_distances / numpy.max(tmp_distances)
    

    


def compute_task_distances(mss_id):
    

    import numpy
    import expenv
    
    mss = expenv.MultiSplitSet.get(mss_id)
    
    # load data
    #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_PseudoSeq_BlosumEnc_pearson.txt")
    f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/All_PseudoSeq_Hamming.txt")
    #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_PseudoSeq_BlosumEnc_euklid.txt")
    #f = file("/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/mhc/MHC_Distanzen/MHC_Distanzen/ALL_RAxML.txt")
    
    num_lines = int(f.readline().strip())
    task_distances = numpy.zeros((num_lines, num_lines))
    name_to_id = {}

    for (i, line) in enumerate(f):
        tokens = line.strip().split("\t")
        name = str(tokens[0])
        name_to_id[name] = i

        entry = numpy.array([v for (j,v) in enumerate(tokens) if j!=0])
        assert len(entry)==num_lines, "len_entry %i, num_lines %i" % (len(entry), num_lines)
        task_distances[i,:] = entry
        
        
    
    # cut relevant submatrix
    active_ids = [name_to_id[ss.dataset.organism] for ss in mss.split_sets] 
    tmp_distances = task_distances[active_ids, :]
    tmp_distances = tmp_distances[:, active_ids]
    print "distances ", tmp_distances.shape

    
    # normalize distances
    task_distances = task_distances / numpy.max(tmp_distances)
    
    id_to_name = [ss.dataset.organism for ss in mss.split_sets]

    return (tmp_distances, id_to_name)



def perform_clustering(mss_id):

    import numpy
    import expenv
    
    mss = expenv.MultiSplitSet.get(mss_id)
    


    from method_mhc_mkl import SequencesHandler
    from shogun.Distance import EuclidianDistance, HammingWordDistance
    from shogun.Features import StringCharFeatures, StringWordFeatures, PROTEIN
    from shogun.Clustering import Hierarchical
    from shogun.PreProc import SortWordString
    
    order = 1
    gap = 0
    reverse = False
    
    seq_handler = SequencesHandler()
    
    data = [seq_handler.get_seq(ss.dataset.organism) for ss in mss.split_sets] 

    charfeat=StringCharFeatures(PROTEIN)
    charfeat.set_features(data)
    feats=StringWordFeatures(charfeat.get_alphabet())
    feats.obtain_from_char(charfeat, order-1, order, gap, reverse)
    preproc=SortWordString()
    preproc.init(feats)
    feats.add_preproc(preproc)
    feats.apply_preproc()

    
    use_sign = False

    distance = HammingWordDistance(feats, feats, use_sign)
    #distance = EuclidianDistance()
    
    merges=4
    hierarchical=Hierarchical(merges, distance)
    hierarchical.train()

    hierarchical.get_merge_distances()
    hierarchical.get_cluster_pairs()
    
    
    return hierarchical



def perform_orange_clustering(mss_id):

    import orange
    from task_similarities import TreeNode
    import helper

    #(dist_full, id_to_name) = compute_task_distances(mss_id)
    p = '/fml/ag-raetsch/home/cwidmer'
    (dist_full, id_to_name) = helper.load(p + "/dist")

    l = []                 
    for i in range(len(dist_full)):
        l.append([])       
        for j in range(i+1,len(dist_full)):
            l[i].append(dist_full[i,j]) 
    l.reverse()
    
    m = orange.SymMatrix(l)
    
    
    root = orange.HierarchicalClustering(m, linkage=orange.HierarchicalClustering.Average)
    root_node = TreeNode("root")
    
    clusters = [root]
    nodes = [root_node]
    
    
    while len(clusters) > 0:
        
        cluster = clusters.pop(0)
        node = nodes.pop(0)
    
        # append nodes if non-empty
        if cluster.left:
            clusters.append(cluster.left)

            
            name = str(tuple(cluster.left))

            if len(tuple(cluster.left))==1:
                name = id_to_name[tuple(cluster.left)[0]]
            print name            
            # create nodes
            left_node = TreeNode(name)
            node.add_child(left_node, 1.0)
            nodes.append(left_node)
            
        # append nodes if non-empty
        if cluster.right:
            clusters.append(cluster.right)

            
            name = str(tuple(cluster.right))

            if len(tuple(cluster.right))==1:
                name = id_to_name[tuple(cluster.right)[0]]
            print name            
            # create nodes
            right_node = TreeNode(name)
            node.add_child(right_node, 1.0)
            nodes.append(right_node)   
    
    
    return root_node
 
 
def get_position_weight():
    
    for cost in unique_costs:
        runs_cost = [r for r in L1_runs if r.method.param.cost==cost]
        print len(runs_cost)
        tmp_perf = numpy.mean([r.assessment.auPRC for r in runs_cost])
        print tmp_perf, cost
        if tmp_perf > best_performance:
            best_performance = tmp_perf; print tmp_perf
            best_run = cost
            print tmp_perf
    
    
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

    
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)


    create_graph()
          
        
if __name__ == "__main__":
    main()
    
