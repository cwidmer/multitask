import numpy
import expenv
import helper

import shogun_factory_new as shogun_factory
from shogun.Kernel import MultitaskKernelPlifNormalizer, MultitaskKernelNormalizer, CTaxonomy, CNode
from shogun.Classifier import SVMLight
from helper import Options
from base_method import PreparedMultitaskData



def test_trivial():

    # come up with support
    support = numpy.linspace(0, 4, 4)
    betas = [1,1,1,1]
    
    # come up with distance matrix for binary tree of depth 2
    distances = [[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 1], [2, 2, 1, 0]]
    
    # task vector
    task_vector = [0,1,2,3]
    
    # create tree normalizer 
    norm = MultitaskKernelPlifNormalizer(support, task_vector)


    num_unique = norm.get_num_unique_tasks([0,0,1,1,0,0,2,1,1])
    
    
    print num_unique
    assert(num_unique == 3)

    # set distances
    for i in range(num_unique):
        for j in range(num_unique):
            norm.set_task_distance(i,j, distances[i][j])

    # set betas
    print "simple betas", betas
    for (i, beta) in enumerate(betas):
        norm.set_beta(i, beta)


    # get similarity
    for i in range(num_unique):
        for j in range(num_unique):
            print "(%i,%i) = d %f = s %f" % (i,j, norm.get_task_distance(i,j), norm.get_task_similarity(i,j))
    

    ###########################
    
    betas = numpy.linspace(0, 1, len(betas)).tolist()    
    betas.reverse()
    
    # set betas
    print "linear betas", betas
    for (i, beta) in enumerate(betas):
        norm.set_beta(i, beta)


    # get similarity
    for i in range(num_unique):
        for j in range(num_unique):
            print "(%i,%i) = d %f = s %f" % (i,j, norm.get_task_distance(i,j), norm.get_task_similarity(i,j))
    

    print "0-0:", norm.compute_task_similarity(0,0)

    print "num betas: %i" % (norm.get_num_betas())



def test_data():
    
    ##################################################################
    # select MSS
    ##################################################################
    
    mss = expenv.MultiSplitSet.get(379)
    
    
    
    ##################################################################
    # data
    ##################################################################
    
    # fetch data
    instance_set = mss.get_train_data(-1)
    
    # prepare data
    data = PreparedMultitaskData(instance_set, shuffle=True)
    
    # set parameters
    param = Options()
    param.kernel = "WeightedDegreeStringKernel"
    param.wdk_degree = 4
    param.cost = 1.0
    param.transform = 1.0
    param.id = 666
    param.freeze()
    
    
    
    
    ##################################################################
    # taxonomy
    ##################################################################
    
    
    taxonomy = shogun_factory.create_taxonomy(mss.taxonomy.data)
    
    
    support = numpy.linspace(0, 100, 4)
    
    
    distances = [[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 1], [2, 2, 1, 0]]
    
    # create tree normalizer 
    tree_normalizer = MultitaskKernelPlifNormalizer(support, data.task_vector_names)
    
    
    
    
    task_names = data.get_task_names()
    
    
    FACTOR = 1.0
    
    
    # init gamma matrix
    gammas = numpy.zeros((data.get_num_tasks(), data.get_num_tasks()))
    
    for t1_name in task_names:
        for t2_name in task_names:
            
            similarity = taxonomy.compute_node_similarity(taxonomy.get_id(t1_name), taxonomy.get_id(t2_name))        
            gammas[data.name_to_id(t1_name), data.name_to_id(t2_name)] = similarity
    
    helper.save("/tmp/gammas", gammas)
    
    
    gammas = gammas * FACTOR
    
    cost = param.cost * numpy.sqrt(FACTOR) 
    
    print gammas
    
    
    ##########
    # regular normalizer
    
    normalizer = MultitaskKernelNormalizer(data.task_vector_nums)
    
    for t1_name in task_names:
        for t2_name in task_names:
                    
            similarity = gammas[data.name_to_id(t1_name), data.name_to_id(t2_name)]
            normalizer.set_task_similarity(data.name_to_id(t1_name), data.name_to_id(t2_name), similarity)
    
                
    ##################################################################
    # Train SVMs
    ##################################################################
    
    # create shogun objects
    wdk_tree = shogun_factory.create_kernel(data.examples, param)
    lab = shogun_factory.create_labels(data.labels)
    
    wdk_tree.set_normalizer(tree_normalizer)
    wdk_tree.init_normalizer()
    
    print "--->",wdk_tree.get_normalizer().get_name()
    
    svm_tree = SVMLight(cost, wdk_tree, lab)
    svm_tree.set_linadd_enabled(False)
    svm_tree.set_batch_computation_enabled(False)
    
    svm_tree.train()
    
    del wdk_tree
    del tree_normalizer
    
    print "finished training tree-norm SVM:", svm_tree.get_objective()
    
    
    wdk = shogun_factory.create_kernel(data.examples, param)
    wdk.set_normalizer(normalizer)
    wdk.init_normalizer()
    
    print "--->",wdk.get_normalizer().get_name()
    
    svm = SVMLight(cost, wdk, lab)
    svm.set_linadd_enabled(False)
    svm.set_batch_computation_enabled(False)
    
    svm.train()
    
    print "finished training manually set SVM:", svm.get_objective()
    
    
    alphas_tree = svm_tree.get_alphas()
    alphas = svm.get_alphas()
    
    assert(len(alphas_tree)==len(alphas))
    
    for i in xrange(len(alphas)):
        assert(abs(alphas_tree[i] - alphas[i]) < 0.0001)
        
    print "success: all alphas are the same"
    
