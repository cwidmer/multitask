import numpy
import expenv
import helper

import shogun_factory_new as shogun_factory
from shogun.Kernel import MultitaskKernelTreeNormalizer, MultitaskKernelNormalizer, CTaxonomy, CNode
from shogun.Classifier import SVMLight
from helper import Options
from base_method import PreparedMultitaskData



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


# create tree normalizer 
tree_normalizer = MultitaskKernelTreeNormalizer(data.task_vector_names, data.task_vector_names, taxonomy)


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

