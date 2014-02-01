from shogun.Kernel import WeightedDegreeStringKernel, GaussianKernel
from shogun.Features import StringCharFeatures, DNA, Labels, RealFeatures
from shogun.Classifier import LibSVM, SVMLight, MSG_DEBUG
#from shogun.Shogun import LibSVM, SVMLight, WeightedDegreeStringKernel, GaussianKernel, StringCharFeatures, DNA, Labels, RealFeatures, MSG_DEBUG

import expenv
import numpy
import helper

from numpy import array, float64
import sys

# create dense matrices A,B,C
A=array([[1,2,3],[4,0,0],[0,0,0],[0,5,0],[0,0,6],[9,9,9]], dtype=float64)
B=array([1,1,1,-1,-1,-1], dtype=float64)


# ... of type Real, LongInt and Byte
feats_train = RealFeatures(A.transpose())
kernel = GaussianKernel(feats_train, feats_train, 1.0)
kernel.io.set_loglevel(MSG_DEBUG)

lab = Labels(B)

svm = SVMLight(1, kernel, lab)
svm.train()


helper.save("/tmp/awesome_svm", svm)
svm = helper.load("/tmp/awesome_svm")

svm.train()


#sys.exit(0)


run = expenv.Run.get(1010)
#run = expenv.Run.get(974)
dat = run.get_train_data()

print dat.keys()
d = dat["thaliana"]
subset_size = 20

examples = [i.example for i in d[0:subset_size]]
labels = [i.label for i in d[0:subset_size]]

print "len(examples)", len(examples)
print "string length", len(examples[0])

labels[2] = 1
labels[12] = 1
labels[15] = 1
labels[8] = 1
labels[19] = 1


feat = StringCharFeatures(DNA)
feat.set_features(examples)

helper.save("/tmp/feat", feat)
feat2 = helper.load("/tmp/feat")


wdk = WeightedDegreeStringKernel(feat, feat, 1)

print "PY: saving kernel"
wdk.io.set_loglevel(MSG_DEBUG)
helper.save("/tmp/awesome", wdk)
#print wdk.toString()
#print "PY: kernel saved, loading kernel"
wdk2 = helper.load("/tmp/awesome")
print "PY: kernel loaded"

#wdk2 = WeightedDegreeStringKernel(feat2, feat2, 1)

lab = Labels(numpy.array(labels))

svm = SVMLight(1, wdk2, lab)
#print "saving SVM"
#helper.save("/tmp/awesome_svm2", svm)
print "done saving svm\n\n"
svm.train()


#print svm.toString()

#print wdk2.toString()

#print "WDK degree", wdk.get_degree()

#print feat.to_string()
#svm2 = helper.load("/tmp/awesome_svm2")
#svm2.train()

#svm3 = SVMLight(1, wdk, lab)
#svm3.train()


print "===================================="

#print "simple svm", svm.get_objective(), svm2.get_objective()#, svm3.get_objective()

print "##############"


