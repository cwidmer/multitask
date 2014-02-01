from shogun.Kernel import WeightedDegreeStringKernel
from shogun.Classifier import LibSVM, SVMLight, DomainAdaptationSVM
from shogun.Features import Labels, StringCharFeatures, DNA
from shogun.Library import Math_init_random

from openopt import QP

import expenv
import numpy


run = expenv.Run.get(1010)
#run = expenv.Run.get(974)
dat = run.get_train_data()

print dat.keys()
d = dat["thaliana"]
subset_size = 200


#############################################
#    compute pre-svm
#############################################

examples_presvm = [i.example for i in d[0:subset_size]]
labels_presvm = [i.label for i in d[0:subset_size]]

labels_presvm[2] = 1
labels_presvm[12] = 1
labels_presvm[15] = 1
labels_presvm[8] = 1
labels_presvm[19] = 1


feat_presvm = StringCharFeatures(DNA)
feat_presvm.set_features(examples_presvm)
wdk_presvm = WeightedDegreeStringKernel(feat_presvm, feat_presvm, 1)
lab_presvm = Labels(numpy.array(labels_presvm))


presvm = SVMLight(1, wdk_presvm, lab_presvm)
presvm.train()

presvm2 = LibSVM(1, wdk_presvm, lab_presvm)
presvm2.train()

print "svmlight", presvm.get_objective()
print "libsvm", presvm2.get_objective()

assert(abs(presvm.get_objective() - presvm2.get_objective())<= 0.001)

print "simple svm", presvm.get_objective()

print "len(examples_presvm)", len(examples_presvm)

print "##############"


#############################################
#    compute linear term manually
#############################################

examples = [i.example for i in d[subset_size:subset_size*2]]
labels = [i.label for i in d[subset_size:subset_size*2]]

labels[2] = 1
labels[12] = 1
labels[15] = 1
labels[8] = 1
labels[19] = 1

feat = StringCharFeatures(DNA)
feat.set_features(examples)
wdk = WeightedDegreeStringKernel(feat, feat, 1)
lab = Labels(numpy.array(labels))
tmp_lab = numpy.double(labels)




#svm = LibSVM(1.0, wdk, lab)


#svm.set_shrinking_enabled(False)


y = numpy.array(labels)

N = len(y)

#f = -numpy.array(range(subset_size))

print "python N:", N


 
B = 2.0
old_svm = presvm

p = numpy.zeros(len(examples))

#compute cross-kernel                
kv = old_svm.get_kernel()
left = old_svm.get_kernel().get_lhs()                    
kv.init(left, feat)


inner = []

for idx in xrange(len(examples)):

    inner_sum = 0.0

    for j in xrange(old_svm.get_num_support_vectors()):

        sv_id = int(old_svm.get_support_vectors()[j])
        alpha = old_svm.get_alpha(j)

        inner_sum = inner_sum + alpha * kv.kernel(sv_id, idx)
        
    inner.append(inner_sum)


    #general case
    p[idx] = B * tmp_lab[idx] * inner_sum - 1.0


################
#checking inner term
presvm.set_bias(0.0)
tmp_out = presvm.classify(feat).get_labels()

for i in xrange(len(examples)):
    
    #print inner[i], tmp_out[i]
    assert(abs(inner[i]-tmp_out[i])<= 0.001)


svm = SVMLight(1.0, wdk, lab)
svm.set_linear_term(p)
Math_init_random(1)
svm.train()

###############
#compare to LibSVM


svm2 = LibSVM(1.0, wdk, lab)
svm2.set_linear_term(p)
Math_init_random(1)
svm2.train()


svm3 = LibSVM(1.0, wdk, lab)
Math_init_random(1)
svm3.train()

print "SVMLight linear:", svm.get_objective()
print "LibSVM linear:", svm2.get_objective()
print "LibSVM:", svm3.get_objective()


print svm.get_objective(), svm2.get_objective()
assert(abs(svm.get_objective()-svm2.get_objective())<= 0.001)


sv_idx = svm.get_support_vectors()
alphas = svm.get_alphas()
alphas_full = numpy.zeros(N)
alphas_full[sv_idx] = alphas




lin = svm.get_linear_term()
print "AAAAAA", lin, type(lin)





objective =  svm.get_objective()

print "svmlight alphas:", numpy.array(alphas[0:5])


#############################################
#    compute DA-SVMs in shogun
#############################################



dasvm = DomainAdaptationSVM(1.0, wdk, lab, presvm, B)
#dasvm = SVMLight(1.0, wdk, lab)
Math_init_random(1)

dasvm.train()


#dasvm = SVMLight(1.0, wdk, lab)
#dasvm.set_linear_term(numpy.double(p))
#dasvm.train()

 

lin_da = dasvm.get_linear_term()
daobj = dasvm.get_objective()

sv_idx_da = dasvm.get_support_vectors()
alphas_da = dasvm.get_alphas()

alphas_full_da = numpy.zeros(N)
alphas_full_da[sv_idx_da] = alphas_da



################
#checking linear term
presvm.set_bias(0.0)
tmp_out = -B*presvm.classify(feat).get_labels()*tmp_lab - 1

for i in xrange(len(examples)):
    
    print lin_da[i], tmp_out[i]
    assert(abs(lin_da[i]-tmp_out[i])<= 0.001)


for i in xrange(len(lin)):
    
    a1 = lin[i]
    a2 = lin_da[i]
    
    print a1, a2
    assert(abs(a1-a2)<= 0.001)
        
    
print "all lin terms agree:"
#print "-------------------------------------"


print "SVM objective:", objective
print "DA_SVM objective:", daobj
assert(abs(objective-daobj)<= 0.001)
print "objectives agree:"

#print "-------------------------------------"


sv_idx_da.sort()
sv_idx.sort()


for i in xrange(len(sv_idx_da)):
    
    a1 = sv_idx[i]
    a2 = sv_idx_da[i]
    
    #print a1, a2
    assert(a1 == a2)
    
    
print "all sv_idx terms agree:"
#print "-------------------------------------"
for i in xrange(len(alphas_full)):
    
    a1 = alphas_full[i]
    a2 = alphas_full_da[i]
    
    #print a1, a2
    assert(abs(a1-a2)<= 0.001)
#print "-------------------------------------"    

print "all alphas agree"
    
    
    
    
    
#############################################
#    classify
#############################################

dat = run.get_eval_data()

dtest = dat["thaliana"]
subset_size = 5


test_examples = [i.example for i in dtest[0:subset_size]]
test_labels = [i.label for i in dtest[0:subset_size]]

feat_test = StringCharFeatures(DNA)
feat_test.set_features(test_examples)

print "classifying with DA_SVM"
out_dasvm = dasvm.classify(feat_test).get_labels()

print out_dasvm


svm_out = svm.classify(feat_test).get_labels()
pre_svm_out = presvm.classify(feat_test).get_labels()

total_out = svm_out + B*(pre_svm_out - presvm.get_bias())

print total_out


for i in xrange(len(out_dasvm)):
    
    o1 = total_out[i]
    o2 = out_dasvm[i]
    
    assert(abs(o1-o2)<= 0.001)


print "classification agrees"

