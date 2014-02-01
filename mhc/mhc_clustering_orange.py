import numpy



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