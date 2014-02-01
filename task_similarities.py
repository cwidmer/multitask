#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
Created on 12.07.2009
@author: Christian Widmer
@summary: This class implements all ideas regarding distance measures between tasks,
          including similarity matrices and taxonomies.

"""

import numpy
import yapgvb
import os
import re

from expenv import Taxonomy


def dataset_to_hierarchy(dataset_name):
    """
    
    helper to construct taxonomy
    
    @param dataset_name: task id 
    @type dataset_name: str
    @param train_data: training data
    @type train_data: dict<str, list<instances> > 
    @return: A hierarchy of tasks with data attached to the leaves
    @rtype: TreeNode
    """
    
    
    
    #factory
    mymap = {"motif": create_hard_hierarchy_motif2,
             "motif2": create_hard_hierarchy_motif2,
             "promoter": create_hard_promoter,
             "tiny promoter": create_hard_promoter,
             #"medium splicing": create_hard_splicing,
             "large splicing": create_hard_splicing,
             #"broad_splicing": create_broad_splicing,
             "complex promoter": create_hard_complex_promoter}


    root = None


    #TODO refactor this crappy code, really

    if mymap.has_key(dataset_name):        
        #fetch hierarchy with appropriate hierarchy function
        root = mymap[dataset_name]()
        
        return root
        #some hack to create trivial hierarchy
        #task_ids = root.get_data_keys()
        #root = create_simple_hierarchy(task_ids)

    elif dataset_name.find("toy377") != -1:
        return Taxonomy.get(18087)

    elif dataset_name.find("toy") != -1:
        root = create_hard_hierarchy_motif_two() 

    elif dataset_name.find("debug") != -1:
        return Taxonomy.get(18087)

    elif dataset_name.find("small splicing") != -1:
        return Taxonomy.get(18092)

    elif dataset_name.find("medium splicing") != -1:
        return Taxonomy.get(18097)
    
    elif dataset_name.find("broad_splicing") != -1:
        return Taxonomy.get(18101)
     
    elif dataset_name.find("deep") != -1:
        
        # we assume that the number of tasks is encoded in dataset name
        num_levels = int(numpy.log2(int(re.findall("deep(\w+)_", dataset_name)[0])))
        btf = BinaryTreeFactory()
        root = btf.create_binary_tree(num_levels)

    elif dataset_name == "mhc":
        # load pre-computed taxonomy
        return Taxonomy.get(18076) #all edge_weights one
        #return Taxonomy.get(18077) #all edge_weights zero


    taxonomy = Taxonomy(data=root, description=dataset_name)
    return taxonomy  



class TreeNode(object):
    """
    Simple graph implemenation for hierarchical multitask
    """
    
    
    def __init__(self, name=""):
        """
        define fields
        
        @param name: node name, for leaves this defines the dataset identifier
        @type name: str
        """

        self.name = name        
        self.parent = None
        self.children = []
        self.predictor = None
        self.edge_weight = 0
        self.cost = 1.0


        
    def add_child(self, node, weight=1.0):
        """
        add node as child of current leaf
        
        @param node: child node
        @type node: TreeNode
        """
        
        node.parent = self
        node.edge_weight = weight
        
        self.children.append(node)


    
    def get_data_keys(self):
        """
        fetch dataset names that are connected to leaves below the current node
        this can be used as key to data structures
        
        @return: list of dataset names
        @rtype: list<str>
        """
        
        return [node.name for node in self.get_leaves()]
        
        
        
        
    def get_leaves(self):
        """
        fetch all leaves with breadth first search from node
        
        @return: list of leaves
        @rtype: list<TreeNode>
        """
        
        leaves = []
        
        
        grey_nodes = [self]
        
        while len(grey_nodes)>0:
            
            
            node = grey_nodes.pop(0) #pop first item (go python!)
            
            if len(node.children) == 0:
                leaves.append(node)
            else:
                grey_nodes.extend(node.children)
                
        return leaves
  

    def get_nearest_neighbor(self):
        """
        
        """

        leaves = [leaf for leaf in self.parent.get_leaves() if leaf!=self]

        leftmost = leaves[0]

        return leftmost

   
   
    def get_all_nodes(self):
        """
        fetch all nodes with breadth first search from node
        
        @return: list of nodes
        @rtype: list<TreeNode>
        """
        
        nodes = []
        
        
        grey_nodes = [self]
        
        while len(grey_nodes)>0:
            
            node = grey_nodes.pop(0) #pop first item (go python!)
            nodes.append(node)
            grey_nodes.extend(node.children)
            
                
        return nodes


    def get_node(self, node_name):
        """
        get node from subtree rooted at self by name
        
        @param node_name: name of node to get
        @type node_name: str
        
        @return: node with name node_name
        @rtype: TreeNode
        """
        
        candidates = [node for node in self.get_all_nodes() if node.name==node_name]
        
        assert(len(candidates)==1)
        
        return candidates[0]
        
        

   
    def get_path_root(self):
        """
        fetch all ancesters of current node (excluding self) 
        until root is reached including root  
        
        @return: list of nodes on the path to root
        @rtype: list<TreeNode>
        """
        
                
        nodes_on_path =[]
        
        node = self

        while node != None:                    
            nodes_on_path.append(node)
            node = node.parent

        return nodes_on_path
    
    
    
    def is_root(self):
        """
        returns true if self is the root node  
        
        @return: indicator if self is root
        @rtype: bool
        """
        
        if self.parent == None:
            return True
        
        else:
            return False


    def is_leaf(self):
        """
        returns true if self is a leaf  
        
        @return: indicator if self is root
        @rtype: bool
        """
        
        if len(self.children) == 0:
            return True
        
        else:
            return False

    
    
    def clear_predictors(self):
        """
        removes predictors from all nodes
        """
        
        all_nodes = self.get_all_nodes()
        
        for node in all_nodes:
            node.predictor = None
            
        
        

    def plot(self, file_name="demo", force_num=False, plot_cost=False, plot_B=False):
        """
        visualizes taxonomy with help of the yetanothergraphvizbinding package
        
        a png is created and tried to open with evince (yes, hardcoded for now)   
        
        @return: graph data structure in yapgvb format
        @rtype: yapgvb.Digraph
        """

        
        graph = yapgvb.Digraph("my_graph")
    
        #graph.ranksep = 3
        #graph.ratio = "auto"
    
        grey_nodes = [self]
        
        
        counter = 0
        
        name = ""
        if self.name=="" or force_num:
            name = "root" #str(counter) + ": " + self.name
        else:
            name = self.name
            
        
        self.node = graph.add_node(label = name)
        self.node.color = "gray95"
        
        while len(grey_nodes)>0:
           
            node = grey_nodes.pop(0) #pop first item
                
                    
            print node.name

            #enqueue children
            if node.children != None:
                
                grey_nodes.extend(node.children)
        
                #add edges
                for child_node in node.children:
                    
                    counter += 1

                    child_name = ""
                    if child_node.name=="" or force_num:
                        child_name = str(counter) + ": " + child_node.name
                    else:
                        child_name = child_node.name
                    
                    child_node.node = graph.add_node(label = child_name)
                    
                    child_node.node.style = "filled"
                    
                    if child_node.is_leaf():
                        child_node.node.color = "gray80"
                        child_node.node.shape = "doubleoctagon" #"doublecircle"
                        
                    else:
                        child_node.node.color = "gray95"
                    
                    edge = node.node >> child_node.node
                    
                    
                    tmp_label = ""
                    
                    if plot_cost:
                        try:
                            tmp_label += "C=" + str(child_node.cost)
                        except Exception:
                            print "cost attribute not set"
                        
                    if plot_B:
                        tmp_label += "B=" + str(child_node.edge_weight)
                        
                    edge.label = tmp_label


    
        print "Using dot for graph layout..."
        graph.layout(yapgvb.engines.dot)
        #graph.layout(yapgvb.engines.neato)
        #graph.layout(yapgvb.engines.twopi)
        #graph.layout(yapgvb.engines.circo)
        
        
        demo_formats = [
            yapgvb.formats.png
            #yapgvb.formats.ps
            #yapgvb.formats.xdot
        ]
        
        for myformat in demo_formats:
            filename = file_name + ".%s" % myformat
    
            print "  Rendering .%s ..." % filename
    
            graph.render(filename)            
        
        
        #os.system("evince " + filename + " &")
        
        return graph



def create_distance_structure_from_tree(root):
    """
    @param root: root of the tree
    @type root: TreeNode

    @return: structure containing distance between tasks
    @rtype: dict<tuple(str,str), float>
    """
    
    task_ids = [node.name for node in root.get_leaves()]
    num_tasks = len(task_ids)
    task_distance = {}
    
    for i in xrange(num_tasks):
        for j in xrange(i, num_tasks):
    
            task1_id = task_ids[i]
            task2_id = task_ids[j]
            
            task_distance[(task1_id, task2_id)] = compute_hop_distance(root, task1_id, task2_id)
            
            
    return task_distance



def compute_hop_distance(root, task1_id, task2_id):
    """
    computes hop distance between two tasks according to a tree
    
    @param root: root of the tree
    @type root: TreeNode
    @param task1_id: task name of first task
    @type task1_id: str
    @param task2_id:
    @type task2_id: str:
    
    @return: distance between tasks
    @rtype: float 
    """
    
    if task1_id == task2_id:
        return 0
    
    
    leaves = root.get_leaves()
    
    task1 = [leaf for leaf in leaves if leaf.name == task1_id][0]
    task2 = [leaf for leaf in leaves if leaf.name == task2_id][0]

    path1 = task1.get_path_root()
    path2 = task2.get_path_root()
    
    path_identical = True
    
    while path_identical and len(path1) > 0 and len(path2) > 0:
        
        tmp_node1 = path1[-1]
        tmp_node2 = path2[-1]
        
        #print "======"
        #print "path1:", [node.name for node in path1]
        #print "path2:", [node.name for node in path2]
        
        
        if tmp_node1 != tmp_node2:
            path_identical = False
        else:
            path1.pop()
            path2.pop()
            
            
    distance = float(len(path1) + len(path2) + 2)
    
    #print "d(%s,%s)=%f" % (task1_id, task2_id, distance)
    
    return distance




def create_simple_hierarchy(task_ids):
    """
    create trivial hierarchy for debugging purposes
    
        0
       /|\
      / | \
     1  2  3
    
    @param instance_sets: task_ids
    @type instance_sets: list<str>
    @return: root node of simple tree
    @rtype: TreeNode
    """
    
    
    root = TreeNode()
    
    #add all leaves directly to root node
    for task_id in task_ids:
        child = TreeNode(task_id)

        root.add_child(child)
        
        
    return root



def create_hard_hierarchy_motif_two():
    """
    creates hard-coded hiearchy with two tasks
   
    @param instance_sets: train data
    @type instance_sets: dict<task_id, list<Instance> >
    @return: root node of simple tree
    @rtype: TreeNode
    """
    
    
    root = TreeNode("root")
    root.cost = 1.0
    
    child1 = TreeNode("toy_0")
    root.add_child(child1, weight=1.0)
    child1.cost = 1.0
    
    child2 = TreeNode("toy_1")
    root.add_child(child2, weight=1.0)
    child2.cost = 1.0
    
    
    return root



def create_hard_hierarchy_motif2(remove_me):
    """
    creates hard-coded hiearchy with four tasks
   
    @param instance_sets: train data
    @type instance_sets: dict<task_id, list<Instance> >
    @return: root node of simple tree
    @rtype: TreeNode
    """
    
    
    root = TreeNode("root")
    
    child1 = TreeNode("inner1")
    root.add_child(child1, weight=0.0)
    
    toy_0 = TreeNode("toy_0")
    child1.add_child(toy_0, weight=1.0)

    toy_1 = TreeNode("toy_1")
    child1.add_child(toy_1, weight=1.0)
    
    child2 = TreeNode("inner2")
    root.add_child(child2, weight=0.0)
    
    toy_2 = TreeNode("toy_2")
    child2.add_child(toy_2, weight=1.0)

    toy_3 = TreeNode("toy_3")
    child2.add_child(toy_3, weight=remove_me)
    
    
    root.plot("/tmp/mygraph")
    os.system("evince /tmp/mygraph.ps &")
    
    return root





class BinaryTreeFactory(object):
    """
    Binary Tree Factory
    """

    node_counter = 0
    inner_node_counter = 0
    leaf_counter = 0    

    nodes = None
    leaves = None


    def create_binary_tree(self, num_levels):
        """
        creates binary tree with num_levels and 2^n leaves
        
        @param num_levels: number of levels
        @type num_levels: int
        
        @return: root node with binary tree attached
        @rtype: TreeNode
        """

        print "creating binary tree with", pow(2, num_levels), "tasks"

        # init
        self.node_counter = 0
        self.inner_node_counter = 0
        self.leaf_counter = 0

        self.nodes = []
        self.leaves = []


        # create root
        root = TreeNode("root")

        # recursive call
        self.__create_subtree(root, num_levels-1)
        self.__create_subtree(root, num_levels-1)

        # return root node
        return root



    def __create_subtree(self, parent_node, level):
        """
        recursive call to generate binary tree
        
        @param parent_node: parent node
        @type parent_node: TreeNode
        @param level: current level
        @type level: int
        """

        current_node = TreeNode("no name")
        parent_node.add_child(current_node)

        if level == 0:
            current_node.name = "toy_" + str(self.leaf_counter)
            self.leaf_counter += 1

        else:
            current_node.name = "inner_" + str(self.inner_node_counter)
            self.inner_node_counter += 1
            
            self.__create_subtree(current_node, level-1)
            self.__create_subtree(current_node, level-1)


        self.nodes.append(current_node)
        self.node_counter += 1




def create_hard_hierarchy_deep():
    """
    creates hard-coded hierarchy with eight tasks
   
    @param instance_sets: train data
    @type instance_sets: dict<task_id, list<Instance> >
    @return: root node of simple tree
    @rtype: TreeNode
    """
    
    
    root = TreeNode("root")
    
    inner1 = TreeNode("inner1")
    root.add_child(inner1, 1.0)
    
    
    inner1_a = TreeNode("inner1_a")
    inner1.add_child(inner1_a, 1.0)
    
    toy_0 = TreeNode("toy_0")
    inner1_a.add_child(toy_0, 1.0)
    
    toy_1 = TreeNode("toy_1")
    inner1_a.add_child(toy_1, 1.0)
    
    
    inner1_b = TreeNode("inner1_b")
    inner1.add_child(inner1_b, 1.0)
   
    toy_2 = TreeNode("toy_2")
    inner1_b.add_child(toy_2, 1.0)

    toy_3 = TreeNode("toy_3")
    inner1_b.add_child(toy_3, 1.0)
    
    
    inner2 = TreeNode("inner2")
    root.add_child(inner2, 1.0)
    
    inner2_a = TreeNode("inner2_a")
    inner2.add_child(inner2_a, 1.0)
   
    toy_4 = TreeNode("toy_4")
    inner2_a.add_child(toy_4, 1.0)
    
    toy_5 = TreeNode("toy_5")
    inner2_a.add_child(toy_5, 1.0)


    inner2_b = TreeNode("inner2_b")
    inner2.add_child(inner2_b, 1.0)
   
    toy_6 = TreeNode("toy_6")
    inner2_b.add_child(toy_6, 1.0)
    
    toy_7 = TreeNode("toy_7")
    inner2_b.add_child(toy_7, 1.0)
        
    
    return root



def create_broad_splicing():
    """
   
    @return: root node of deep splicing dataset
    @rtype: TreeNode
    """
    
    
    root = TreeNode()

    nidulans = TreeNode("A.nidulans")
    root.add_child(nidulans)

    plantae = TreeNode("plantae")
    root.add_child(plantae)

    animalia = TreeNode("animalia")
    root.add_child(animalia)

    trichocarpa = TreeNode("P.trichocarpa")
    plantae.add_child(trichocarpa)

    angiosperms = TreeNode("angiosperms")
    plantae.add_child(angiosperms)

    thaliana = TreeNode("A.thaliana")
    angiosperms.add_child(thaliana)

    sativa = TreeNode("O.sativa")
    angiosperms.add_child(sativa)

    chordata = TreeNode("chordata")
    animalia.add_child(chordata)

    savignyi = TreeNode("C.savignyi")
    chordata.add_child(savignyi)

    vertebrata = TreeNode("vertebrata")
    chordata.add_child(vertebrata)

    actinopterygii = TreeNode("actinopterygii")
    vertebrata.add_child(actinopterygii)

    rerio = TreeNode("D.rerio")
    actinopterygii.add_child(rerio)

    latipes = TreeNode("O.latipes")
    actinopterygii.add_child(latipes)

    aculeatus = TreeNode("G.aculeatus")
    actinopterygii.add_child(aculeatus)

    nigroviridis = TreeNode("T.nigroviridis")
    actinopterygii.add_child(nigroviridis)

    mammals = TreeNode("mammals")
    vertebrata.add_child(mammals)

    taurus = TreeNode("B.taurus")
    mammals.add_child(taurus)

    sapiens = TreeNode("H.sapiens")
    mammals.add_child(sapiens)

    musculus = TreeNode("M.musculus")
    mammals.add_child(musculus)

    protostomia = TreeNode("protostomia")
    animalia.add_child(protostomia)

    #nematoda = TreeNode("nematoda")
    elegans = TreeNode("C.elegans")
    protostomia.add_child(elegans)

    diptera = TreeNode("diptera")
    protostomia.add_child(diptera)

    gambiae = TreeNode("A.gambiae")
    diptera.add_child(gambiae)

    melanogaster = TreeNode("D.melanogaster")
    diptera.add_child(melanogaster)


    return root


def create_hard_splicing():
    """
   
    @param instance_sets: train data
    @type instance_sets: dict<task_id, list<Instance> >
    @return: root node of simple tree
    @rtype: TreeNode
    """
    
    
    root = TreeNode()

            
    thaliana = TreeNode("thaliana")
    root.add_child(thaliana) 

    child1 = TreeNode()
    root.add_child(child1)

    drosophila = TreeNode("drosophila")
    child1.add_child(drosophila)

    child2 = TreeNode()
    child1.add_child(child2)
    
    pacificus = TreeNode("pacificus")
    child2.add_child(pacificus)
    
    remanei = TreeNode("remanei")
    child2.add_child(remanei)


    return root
    
    
    

def create_hard_complex_promoter_simplified(train_data):
    """
    create hierarchy

    Gg Gallus gallus (chicken)
    HSV-1 Human herpes simplex virus type 1
    Xl Xenopus laevis (African clawed frog)
    Ps Pisum sativum (pea).
    Mm Mus musculus (mouse)
    Zm Zea mays (maize)
    Hv Hordeum vulgare (barley). (Gerste)
    Ce Caenorhabditis elegans.
    Bt Bos taurus (cattle)
    Ss Sus scrofa (pig).
    Sp Strongylocentrotus purpuratus. (Art Seestern)
    EBV Human Epstein-Barr virus
    Ath A.thaliana
    HCMV Human Cytomegalovirus (HCMV)
    Rn Rattus norvegicus (rat)
    Nt Nicotiana tabacum (common tobacco).
    Gm Glycine max (soybean).
    Ta Triticum aestivum (wheat).

    @param instance_sets: list of datasets
    @type instance_sets: list<list<Instance>>
    @return: root node of simple tree
    @rtype: TreeNode
    """
    
    root = TreeNode()
    

    # plants
    plants = TreeNode()
    root.add_child(plants)

    tmp = TreeNode()
    plants.add_child(tmp)

    tabacco = TreeNode()
    tabacco.data = train_data["Nt"]
    tmp.add_child(tabacco)


    thaliana = TreeNode()
    thaliana.data = train_data["Ath"]
    plants.add_child(thaliana)

    barley = TreeNode()
    barley.data = train_data["Hv"]
    plants.add_child(barley)

    wheat = TreeNode()
    wheat.data = train_data["Ta"]
    plants.add_child(wheat)

    maize = TreeNode()
    maize.data = train_data["Zm"]
    plants.add_child(maize)

    pea = TreeNode()
    pea.data = train_data["Ps"]
    plants.add_child(pea)

    soy_bean = TreeNode()
    soy_bean.data = train_data["Gm"]
    plants.add_child(soy_bean)

    # animals
    animals = TreeNode()
    root.add_child(animals)

    elegans = TreeNode()
    elegans.data = train_data["Ce"]
    animals.add_child(elegans)

    purpuratus = TreeNode()
    purpuratus.data = train_data["Sp"]
    animals.add_child(purpuratus)

    frog = TreeNode()
    frog.data = train_data["Xl"]
    #TODO experimental
    root.add_child(frog)
    
    chicken = TreeNode()
    chicken.data = train_data["Gg"]
    animals.add_child(chicken)

    cow = TreeNode()
    cow.data = train_data["Bt"]
    animals.add_child(cow)

    pig = TreeNode()
    pig.data = train_data["Ss"]
    animals.add_child(pig)

    mouse = TreeNode()
    mouse.data = train_data["Mm"]
    animals.add_child(mouse)

    rat = TreeNode()
    rat.data = train_data["Rn"]
    animals.add_child(rat)

    # viruses
    human_viruses = TreeNode()
    root.add_child(human_viruses)

    herpes = TreeNode()
    herpes.data = train_data["HSV-1"]
    human_viruses.add_child(herpes)

    hcmv = TreeNode()
    hcmv.data = train_data["HCMV"]
    human_viruses.add_child(hcmv)

    viruses = TreeNode()
    root.add_child(viruses)

    bah = TreeNode()
    viruses.add_child(bah)
    
    epv = TreeNode()
    epv.data = train_data["EBV"]
    bah.add_child(epv)

    return root


def create_hard_complex_promoter(train_data):
    """
    create hierarchy

    Gg Gallus gallus (chicken)
    HSV-1 Human herpes simplex virus type 1
    Xl Xenopus laevis (African clawed frog)
    Ps Pisum sativum (pea).
    Mm Mus musculus (mouse)
    Zm Zea mays (maize)
    Hv Hordeum vulgare (barley). (Gerste)
    Ce Caenorhabditis elegans.
    Bt Bos taurus (cattle)
    Ss Sus scrofa (pig).
    Sp Strongylocentrotus purpuratus. (Art Seestern)
    EBV Human Epstein-Barr virus
    Ath A.thaliana
    HCMV Human Cytomegalovirus (HCMV)
    Rn Rattus norvegicus (rat)
    Nt Nicotiana tabacum (common tobacco).
    Gm Glycine max (soybean).
    Ta Triticum aestivum (wheat).

    @param instance_sets: list of datasets
    @type instance_sets: list<list<Instance>>
    @return: root node of simple tree
    @rtype: TreeNode
    """
    
    root = TreeNode()
    

    # plants
    plants = TreeNode()
    root.add_child(plants)

    tabacco = TreeNode()
    tabacco.data = train_data["Nt"]
    plants.add_child(tabacco)

    plants_1 = TreeNode()
    plants.add_child(plants_1)

    thaliana = TreeNode()
    thaliana.data = train_data["Ath"]
    plants_1.add_child(thaliana)

    plants_2 = TreeNode()
    plants_1.add_child(plants_2)

    grain = TreeNode()
    plants_2.add_child(grain)

    legumes = TreeNode()
    plants_2.add_child(legumes)

    barley = TreeNode()
    barley.data = train_data["Hv"]
    grain.add_child(barley)

    wheat = TreeNode()
    wheat.data = train_data["Ta"]
    grain.add_child(wheat)

    maize = TreeNode()
    maize.data = train_data["Zm"]
    grain.add_child(maize)

    pea = TreeNode()
    pea.data = train_data["Ps"]
    legumes.add_child(pea)

    soy_bean = TreeNode()
    soy_bean.data = train_data["Gm"]
    legumes.add_child(soy_bean)

    # animals
    animals = TreeNode()
    root.add_child(animals)

    invertebrates = TreeNode()
    animals.add_child(invertebrates)

    elegans = TreeNode()
    elegans.data = train_data["Ce"]
    invertebrates.add_child(elegans)

    purpuratus = TreeNode()
    purpuratus.data = train_data["Sp"]
    invertebrates.add_child(purpuratus)

    vertebrates = TreeNode()
    animals.add_child(vertebrates)

    frog = TreeNode()
    frog.data = train_data["Xl"]
    vertebrates.add_child(frog)
    
    endotherm = TreeNode()
    vertebrates.add_child(endotherm)

    chicken = TreeNode()
    chicken.data = train_data["Gg"]
    endotherm.add_child(chicken)

    mamals = TreeNode()
    endotherm.add_child(mamals)
   
    artiodactyla = TreeNode()
    mamals.add_child(artiodactyla)

    cow = TreeNode()
    cow.data = train_data["Bt"]
    artiodactyla.add_child(cow)

    pig = TreeNode()
    pig.data = train_data["Ss"]
    artiodactyla.add_child(pig)

    supraprimates = TreeNode()
    mamals.add_child(supraprimates)

    rodentia = TreeNode()
    supraprimates.add_child(rodentia)

    mouse = TreeNode()
    mouse.data = train_data["Mm"]
    rodentia.add_child(mouse)

    rat = TreeNode()
    rat.data = train_data["Rn"]
    rodentia.add_child(rat)

    # viruses
    human_viruses = TreeNode()
    # one could argue that viruses have similar promoters to their hosts
    supraprimates.add_child(human_viruses)
    #root.add_child(human_viruses)

    herpes = TreeNode()
    herpes.data = train_data["HSV-1"]
    human_viruses.add_child(herpes)

    hcmv = TreeNode()
    hcmv.data = train_data["HCMV"]
    human_viruses.add_child(hcmv)
    
    epv = TreeNode()
    epv.data = train_data["EBV"]
    human_viruses.add_child(epv)

    return root


def create_hard_promoter():
    """
    create hierarchy
           root
            /\
           /  \
          /\   \
         /  \   \
        /\   \   \
       /  \   \   \
      /   /\   \   \
     /   /  \   \   \
    Hs  Mm  Rn  Gg  Dm
    
    @param instance_sets: list of datasets
    @type instance_sets: list<list<Instance>>
    @return: root node of simple tree
    @rtype: TreeNode
    
    task_distance = {}
    task_distance[("Hs", "Hs")] = 0.0
    task_distance[("Hs", "Mm")] = 100.0
    task_distance[("Hs", "Rn")] = 100.0
    task_distance[("Hs", "Gg")] = 300.0
    task_distance[("Hs", "Dm")] = 900.0

    task_distance[("Mm", "Mm")] = 0.0
    task_distance[("Mm", "Rn")] = 30.0
    task_distance[("Mm", "Gg")] = 300.0
    task_distance[("Mm", "Dm")] = 900.0

    task_distance[("Rn", "Rn")] = 0.0
    task_distance[("Rn", "Gg")] = 300.0
    task_distance[("Rn", "Dm")] = 900.0

    task_distance[("Gg", "Gg")] = 0.0
    task_distance[("Gg", "Dm")] = 900.0

    task_distance[("Dm", "Dm")] = 0.0
    
    root.add_child(child1, 600.0/1000.0)
    root.add_child(fly, 900.0/1000.0)
    
    chicken = TreeNode()
    chicken.data = train_data["Gg"]
    child1.add_child(chicken, 300.0/1000.0)
    
    child2 = TreeNode()
    child1.add_child(child2, 200.0/1000.0)
    
    human = TreeNode()
    human.data = train_data["Hs"]
    child2.add_child(human, 1000.0/1000.0)
    
    child3 = TreeNode()
    #child2.add_child(child3, 170.0/1000.0)
    child2.add_child(child3, 600.0/1000.0)
    
    
    mouse = TreeNode()
    mouse.data = train_data["Mm"]
    child3.add_child(mouse, 300.0/1000.0)
    
    rat = TreeNode()
    rat.data = train_data["Rn"]
    child3.add_child(rat, 300.0/1000.0)
    """
    
    root = TreeNode()
    
    child1 = TreeNode()
    fly = TreeNode("Dm")
    
    root.add_child(child1)
    root.add_child(fly)
    
    chicken = TreeNode("Gg")
    child1.add_child(chicken)
    
    child2 = TreeNode()
    child1.add_child(child2)
    
    human = TreeNode("Hs")
    child2.add_child(human)
    
    child3 = TreeNode()
    child2.add_child(child3)
    
    
    mouse = TreeNode("Mm")
    child3.add_child(mouse)
    
    rat = TreeNode("Rn")
    child3.add_child(rat)
    
    
    return root



class TaskMap(object):

    def __init__(self, task_map):

        self.task_map = {}
       
        # create unique keys
        for (key, value) in task_map.items():
            new_key = list(key)
            new_key.sort()
            new_key = tuple(new_key)

            self.task_map[new_key] = value

        # keep task ids
        task_ids = []
        for ids in task_map.keys():
            task_ids.extend(list(ids))
        self.task_ids = list(set(task_ids))


    def get_similarity(self, task_a, task_b):

        key = [task_a, task_b]
        key.sort()
        key = tuple(key)

        return self.task_map[key]


    def taskmap2matrix(self, task_ids=None):
        """
        construct task matrix from task map
        """

        if task_ids == None:
            task_ids = self.task_ids

        n_tasks = len(task_ids)

        task_matrix = numpy.zeros((n_tasks, n_tasks))

        for (i, task_a) in enumerate(task_ids):
            for (j, task_b) in enumerate(task_ids):
                
                task_matrix[i][j] = self.get_similarity(task_a, task_b)
                
        return task_matrix



def fetch_gammas(square_root, base, dataset_name):
    """
    for now hard-coded way of computing the gammas
    """

    ####################################################
    # Promoter
    ####################################################

    task_similarity = {}

    if dataset_name == "promoter" or dataset_name == "tiny promoter":
        """
        task_similarity = numpy.array([[0, 300, 100, 30, 900],
                                     [300, 0, 300, 300, 900],
                                     [100, 300, 0, 100, 900],
                                     [30, 300, 100, 0, 900],
                                     [900, 900, 900, 900, 0]])

                /\
               /  \
              /\   \
             /  \   \
            /\   \   \
           /  \   \   \
          /   /\   \   \
         /   /  \   \   \
        Hs  Mm  Rn  Gg  Dm
        """

        assert(base > 1.0)

        task_similarity = {}
        task_similarity[("Hs", "Hs")] = 0.0
        task_similarity[("Hs", "Mm")] = 100.0
        task_similarity[("Hs", "Rn")] = 100.0
        task_similarity[("Hs", "Gg")] = 300.0
        task_similarity[("Hs", "Dm")] = 900.0

        task_similarity[("Mm", "Mm")] = 0.0
        task_similarity[("Mm", "Rn")] = 30.0
        task_similarity[("Mm", "Gg")] = 300.0
        task_similarity[("Mm", "Dm")] = 900.0

        task_similarity[("Rn", "Rn")] = 0.0
        task_similarity[("Rn", "Gg")] = 300.0
        task_similarity[("Rn", "Dm")] = 900.0

        task_similarity[("Gg", "Gg")] = 0.0
        task_similarity[("Gg", "Dm")] = 900.0

        task_similarity[("Dm", "Dm")] = 0.0




    ####################################################
    # Complex Promoter
    ####################################################

    if dataset_name == "complex promoter":
        """

        Gg Gallus gallus (chicken)
        HSV-1 Human herpes simplex virus type 1
        Xl Xenopus laevis (African clawed frog)
        Ps Pisum sativum (pea).
        Mm Mus musculus (mouse)
        Zm Zea mays (maize)
        Hv Hordeum vulgare (barley). (Gerste)
        Ce Caenorhabditis elegans.
        Bt Bos taurus (cattle)
        Ss Sus scrofa (pig).
        Sp Strongylocentrotus purpuratus. (Art Seestern)
        EBV Human Epstein-Barr virus
        Ath A.thaliana
        HCMV Human Cytomegalovirus (HCMV)
        Rn Rattus norvegicus (rat)
        Nt Nicotiana tabacum (common tobacco).
        Gm Glycine max (soybean).
        Ta Triticum aestivum (wheat).

        """

        task_similarity = {}
        task_similarity[("Gg", "Gg")] = 0.0                                                                                                
        task_similarity[("Gg", "HSV-1")] = 2.0                                                                                             
        task_similarity[("Gg", "Xl")] = 1.0                                                                                                
        task_similarity[("Gg", "Ps")] = 2.0                                                                                                
        task_similarity[("Gg", "Mm")] = 1.0                                                                                                
        task_similarity[("Gg", "Zm")] = 2.0                                                                                                
        task_similarity[("Gg", "Hv")] = 2.0                                                                                                
        task_similarity[("Gg", "Ce")] = 1.0                                                                                                
        task_similarity[("Gg", "Bt")] = 1.0                                                                                                
        task_similarity[("Gg", "Ss")] = 1.0                                                                                                
        task_similarity[("Gg", "Sp")] = 1.0                                                                                                
        task_similarity[("Gg", "EBV")] = 2.0                                                                                               
        task_similarity[("Gg", "Ath")] = 2.0                                                                                               
        task_similarity[("Gg", "HCMV")] = 2.0                                                                                              
        task_similarity[("Gg", "Rn")] = 1.0                                                                                                
        task_similarity[("Gg", "Nt")] = 2.0                                                                                                
        task_similarity[("Gg", "Gm")] = 2.0                                                                                                
        task_similarity[("Gg", "Ta")] = 2.0                                                                                                
        task_similarity[("HSV-1", "HSV-1")] = 0.0                                                                                          
        task_similarity[("HSV-1", "Xl")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Ps")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Mm")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Zm")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Hv")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Ce")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Bt")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Ss")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Sp")] = 2.0                                                                                             
        task_similarity[("HSV-1", "EBV")] = 1.0                                                                                            
        task_similarity[("HSV-1", "Ath")] = 2.0                                                                                            
        task_similarity[("HSV-1", "HCMV")] = 1.0                                                                                           
        task_similarity[("HSV-1", "Rn")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Nt")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Gm")] = 2.0                                                                                             
        task_similarity[("HSV-1", "Ta")] = 2.0                                                                                             
        task_similarity[("Xl", "Xl")] = 0.0                                                                                                
        task_similarity[("Xl", "Ps")] = 2.0                                                                                                
        task_similarity[("Xl", "Mm")] = 1.0                                                                                                
        task_similarity[("Xl", "Zm")] = 2.0                                                                                                
        task_similarity[("Xl", "Hv")] = 2.0                                                                                                
        task_similarity[("Xl", "Ce")] = 1.0                                                                                                
        task_similarity[("Xl", "Bt")] = 1.0                                                                                                
        task_similarity[("Xl", "Ss")] = 1.0                                                                                                
        task_similarity[("Xl", "Sp")] = 1.0                                                                                                
        task_similarity[("Xl", "EBV")] = 2.0                                                                                               
        task_similarity[("Xl", "Ath")] = 2.0                                                                                               
        task_similarity[("Xl", "HCMV")] = 2.0                                                                                              
        task_similarity[("Xl", "Rn")] = 1.0                                                                                                
        task_similarity[("Xl", "Nt")] = 2.0                                                                                                
        task_similarity[("Xl", "Gm")] = 2.0                                                                                                
        task_similarity[("Xl", "Ta")] = 2.0                                                                                                
        task_similarity[("Ps", "Ps")] = 0.0                                                                                                
        task_similarity[("Ps", "Mm")] = 2.0                                                                                                
        task_similarity[("Ps", "Zm")] = 1.0                                                                                                
        task_similarity[("Ps", "Hv")] = 1.0                                                                                                
        task_similarity[("Ps", "Ce")] = 2.0                                                                                                
        task_similarity[("Ps", "Bt")] = 2.0                                                                                                
        task_similarity[("Ps", "Ss")] = 2.0                                                                                                
        task_similarity[("Ps", "Sp")] = 2.0                                                                                                
        task_similarity[("Ps", "EBV")] = 2.0                                                                                               
        task_similarity[("Ps", "Ath")] = 1.0                                                                                               
        task_similarity[("Ps", "HCMV")] = 2.0                                                                                              
        task_similarity[("Ps", "Rn")] = 2.0                                                                                                
        task_similarity[("Ps", "Nt")] = 1.0                                                                                                
        task_similarity[("Ps", "Gm")] = 1.0                                                                                                
        task_similarity[("Ps", "Ta")] = 1.0                                                                                                
        task_similarity[("Mm", "Mm")] = 0.0                                                                                                
        task_similarity[("Mm", "Zm")] = 2.0                                                                                                
        task_similarity[("Mm", "Hv")] = 2.0                                                                                                
        task_similarity[("Mm", "Ce")] = 1.0                                                                                                
        task_similarity[("Mm", "Bt")] = 1.0                                                                                                
        task_similarity[("Mm", "Ss")] = 1.0                                                                                                
        task_similarity[("Mm", "Sp")] = 1.0                                                                                                
        task_similarity[("Mm", "EBV")] = 2.0                                                                                               
        task_similarity[("Mm", "Ath")] = 2.0                                                                                               
        task_similarity[("Mm", "HCMV")] = 2.0                                                                                              
        task_similarity[("Mm", "Rn")] = 1.0                                                                                                
        task_similarity[("Mm", "Nt")] = 2.0                                                                                                
        task_similarity[("Mm", "Gm")] = 2.0                                                                                                
        task_similarity[("Mm", "Ta")] = 2.0                                                                                                
        task_similarity[("Zm", "Zm")] = 0.0                                                                                                
        task_similarity[("Zm", "Hv")] = 1.0                                                                                                
        task_similarity[("Zm", "Ce")] = 2.0                                                                                                
        task_similarity[("Zm", "Bt")] = 2.0                                                                                                
        task_similarity[("Zm", "Ss")] = 2.0                                                                                                
        task_similarity[("Zm", "Sp")] = 2.0                                                                                                
        task_similarity[("Zm", "EBV")] = 2.0                                                                                               
        task_similarity[("Zm", "Ath")] = 1.0                                                                                               
        task_similarity[("Zm", "HCMV")] = 2.0                                                                                              
        task_similarity[("Zm", "Rn")] = 2.0                                                                                                
        task_similarity[("Zm", "Nt")] = 1.0                                                                                                
        task_similarity[("Zm", "Gm")] = 1.0                                                                                                
        task_similarity[("Zm", "Ta")] = 1.0                                                                                                
        task_similarity[("Hv", "Hv")] = 0.0                                                                                                
        task_similarity[("Hv", "Ce")] = 2.0                                                                                                
        task_similarity[("Hv", "Bt")] = 2.0                                                                                                
        task_similarity[("Hv", "Ss")] = 2.0                                                                                                
        task_similarity[("Hv", "Sp")] = 2.0                                                                                                
        task_similarity[("Hv", "EBV")] = 2.0                                                                                               
        task_similarity[("Hv", "Ath")] = 1.0                                                                                               
        task_similarity[("Hv", "HCMV")] = 2.0                                                                                              
        task_similarity[("Hv", "Rn")] = 2.0                                                                                                
        task_similarity[("Hv", "Nt")] = 1.0                                                                                                
        task_similarity[("Hv", "Gm")] = 1.0                                                                                                
        task_similarity[("Hv", "Ta")] = 1.0                                                                                                
        task_similarity[("Ce", "Ce")] = 0.0                                                                                                
        task_similarity[("Ce", "Bt")] = 1.0                                                                                                
        task_similarity[("Ce", "Ss")] = 1.0                                                                                                
        task_similarity[("Ce", "Sp")] = 1.0                                                                                                
        task_similarity[("Ce", "EBV")] = 2.0                                                                                               
        task_similarity[("Ce", "Ath")] = 2.0                                                                                               
        task_similarity[("Ce", "HCMV")] = 2.0                                                                                              
        task_similarity[("Ce", "Rn")] = 1.0                                                                                                
        task_similarity[("Ce", "Nt")] = 2.0                                                                                                
        task_similarity[("Ce", "Gm")] = 2.0                                                                                                
        task_similarity[("Ce", "Ta")] = 2.0                                                                                                
        task_similarity[("Bt", "Bt")] = 0.0                                                                                                
        task_similarity[("Bt", "Ss")] = 1.0                                                                                                
        task_similarity[("Bt", "Sp")] = 1.0                                                                                                
        task_similarity[("Bt", "EBV")] = 2.0                                                                                               
        task_similarity[("Bt", "Ath")] = 2.0                                                                                               
        task_similarity[("Bt", "HCMV")] = 2.0                                                                                              
        task_similarity[("Bt", "Rn")] = 1.0                                                                                                
        task_similarity[("Bt", "Nt")] = 2.0                                                                                                
        task_similarity[("Bt", "Gm")] = 2.0                                                                                                
        task_similarity[("Bt", "Ta")] = 2.0                                                                                                
        task_similarity[("Ss", "Ss")] = 0.0                                                                                                
        task_similarity[("Ss", "Sp")] = 1.0                                                                                                
        task_similarity[("Ss", "EBV")] = 2.0                                                                                               
        task_similarity[("Ss", "Ath")] = 2.0                                                                                               
        task_similarity[("Ss", "HCMV")] = 2.0                                                                                              
        task_similarity[("Ss", "Rn")] = 1.0                                                                                                
        task_similarity[("Ss", "Nt")] = 2.0                                                                                                
        task_similarity[("Ss", "Gm")] = 2.0                                                                                                
        task_similarity[("Ss", "Ta")] = 2.0                                                                                                
        task_similarity[("Sp", "Sp")] = 0.0                                                                                                
        task_similarity[("Sp", "EBV")] = 2.0                                                                                               
        task_similarity[("Sp", "Ath")] = 2.0                                                                                               
        task_similarity[("Sp", "HCMV")] = 2.0                                                                                              
        task_similarity[("Sp", "Rn")] = 1.0                                                                                                
        task_similarity[("Sp", "Nt")] = 2.0                                                                                                
        task_similarity[("Sp", "Gm")] = 2.0                                                                                                
        task_similarity[("Sp", "Ta")] = 2.0                                                                                                
        task_similarity[("EBV", "EBV")] = 0.0                                                                                              
        task_similarity[("EBV", "Ath")] = 2.0                                                                                              
        task_similarity[("EBV", "HCMV")] = 1.0                                                                                             
        task_similarity[("EBV", "Rn")] = 2.0                                                                                               
        task_similarity[("EBV", "Nt")] = 2.0                                                                                               
        task_similarity[("EBV", "Gm")] = 2.0                                                                                               
        task_similarity[("EBV", "Ta")] = 2.0                                                                                               
        task_similarity[("Ath", "Ath")] = 0.0                                                                                              
        task_similarity[("Ath", "HCMV")] = 2.0                                                                                             
        task_similarity[("Ath", "Rn")] = 2.0                                                                                               
        task_similarity[("Ath", "Nt")] = 1.0                                                                                               
        task_similarity[("Ath", "Gm")] = 1.0                                                                                               
        task_similarity[("Ath", "Ta")] = 1.0                                                                                               
        task_similarity[("HCMV", "HCMV")] = 0.0                                                                                            
        task_similarity[("HCMV", "Rn")] = 2.0                                                                                              
        task_similarity[("HCMV", "Nt")] = 2.0                                                                                              
        task_similarity[("HCMV", "Gm")] = 2.0                                                                                              
        task_similarity[("HCMV", "Ta")] = 2.0                                                                                              
        task_similarity[("Rn", "Rn")] = 0.0                                                                                                
        task_similarity[("Rn", "Nt")] = 2.0                                                                                                
        task_similarity[("Rn", "Gm")] = 2.0                                                                                                
        task_similarity[("Rn", "Ta")] = 2.0                                                                                                
        task_similarity[("Nt", "Nt")] = 0.0                                                                                                
        task_similarity[("Nt", "Gm")] = 1.0                                                                                                
        task_similarity[("Nt", "Ta")] = 1.0                                                                                                
        task_similarity[("Gm", "Gm")] = 0.0                                                                                                
        task_similarity[("Gm", "Ta")] = 1.0                                                                                                
        task_similarity[("Ta", "Ta")] = 0.0



    ####################################################
    # Motif
    ####################################################


    elif dataset_name.find("motif") != -1:
        task_similarity = numpy.array([[0, 1, 2, 2], [1, 0, 2, 3], [2, 2, 0, 2], [2, 3, 2, 0]])


    elif dataset_name.find("toy") != -1:

        task_similarity = {}
        task_similarity[("toy_0", "toy_0")] = 0.0
        task_similarity[("toy_0", "toy_1")] = 1.0
        task_similarity[("toy_0", "toy_2")] = 2.0
        task_similarity[("toy_0", "toy_3")] = 2.0

        task_similarity[("toy_1", "toy_1")] = 0.0
        task_similarity[("toy_1", "toy_2")] = 2.0
        task_similarity[("toy_1", "toy_3")] = 2.0

        task_similarity[("toy_2", "toy_2")] = 0.0
        task_similarity[("toy_2", "toy_3")] = 1.0

        task_similarity[("toy_3", "toy_3")] = 0.0


    elif dataset_name.find("deep") != -1:

        # we assume number of tasks is encoded in dataset name
        num_levels = int(numpy.log2(int(re.findall("deep(\w+)_", dataset_name)[0])))
        btf = BinaryTreeFactory()
        root = btf.create_binary_tree(num_levels)
    
        task_similarity = create_distance_structure_from_tree(root)


    elif dataset_name.find("broad_splicing") != -1:

        root = create_broad_splicing()    
        task_similarity = create_distance_structure_from_tree(root)


    ####################################################
    # NIPS multi source
    ####################################################


    elif dataset_name == "nips":
        task_similarity = numpy.array([[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]])


    ####################################################
    # Splicing
    ####################################################


    elif dataset_name.find("splicing") != -1:

        #task_similarity = numpy.array([[0, 1, 2, 3], [1, 0, 2, 3], [2, 2, 0, 2], [3, 3, 2, 0]]) 

        task_similarity = {}
        task_similarity[("pacificus", "pacificus")] = 0.0
        task_similarity[("pacificus", "remanei")] = 1.0
        task_similarity[("pacificus", "drosophila")] = 2.0
        task_similarity[("pacificus", "thaliana")] = 3.0

        task_similarity[("remanei", "remanei")] = 0.0
        task_similarity[("remanei", "drosophila")] = 2.0
        task_similarity[("remanei", "thaliana")] = 3.0

        task_similarity[("drosophila", "drosophila")] = 0.0
        task_similarity[("drosophila", "thaliana")] = 3.0

        task_similarity[("thaliana", "thaliana")] = 0.0


    return distance_to_similarity(task_similarity, square_root, base)



def distance_to_similarity(task_distances, square_root, base):
    """
    function to convert distance matrix to similarity matrix
    
    @param task_distances: matrix of distances
    @param square_root: non-linear transformation parameter
    @param base: base similarity
    @return similarity matrix
    """
    

    largest_distance = max(task_distances.values())
    # non-linear transformation of distances
    for (key, value) in task_distances.items():
        task_distances[key] = numpy.power(value, 1.0 / square_root)

        # normalize distances
        task_distances[key] = task_distances[key]/largest_distance

        # convert to similarity
        task_distances[key] = (base - task_distances[key])/base


    return task_distances
    

