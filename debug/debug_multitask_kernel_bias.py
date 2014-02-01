from shogun.Kernel import WeightedDegreeStringKernel, MultitaskKernelNormalizer, CustomKernel, IdentityKernelNormalizer
from shogun.Features import StringCharFeatures, DNA, Labels
from shogun.Classifier import LibSVM, SVMLight

from openopt import QP
from scipy import io
import scipy.optimize

import cvxopt.base as cb
import cvxopt.solvers as cs
import cvxopt.mosek as mosek
import cvxopt

import expenv
import numpy


def relate_tasks(i,j, size=10):

    if i < size and j < size:
        return 4
    elif i >= size and j >= size:
        return 4
    else:
        return 1


##################################################################
# data
##################################################################

run = expenv.Run.get(1010)
#run = expenv.Run.get(974)
dat = run.get_train_data()

print dat.keys()

d = dat["pacificus"]
subset_size = 20

examples = [i.example for i in d[0:subset_size]]
labels = [i.label for i in d[0:subset_size]]

labels[2] = 1
labels[12] = 1
labels[15] = 1
labels[8] = 1
labels[19] = 1


feat = StringCharFeatures(DNA)
feat.set_features(examples)
lab = Labels(numpy.array(labels))

N = subset_size



##################################################################
# internal modification
##################################################################

task_vector = [0]*(N/2)
task_vector.extend([1]*(N/2))

base_wdk = WeightedDegreeStringKernel(feat, feat, 1)


normalizer = MultitaskKernelNormalizer(task_vector)

#wdk.set_task_vector(task_vector) #, task_vector)

for i in xrange(2):
    for j in xrange(2):

        if i==j:
            normalizer.set_task_similarity(i,j, 4.0)
        else:
            normalizer.set_task_similarity(i,j, 1.0)


base_wdk.set_normalizer(normalizer)

print base_wdk.get_kernel_matrix()
print "--->",base_wdk.get_normalizer().get_name()

wdk = WeightedDegreeStringKernel(feat, feat, 1)

normalizer = IdentityKernelNormalizer()
wdk.set_normalizer(normalizer)


##################################################################
# external modification
##################################################################

km = wdk.get_kernel_matrix()

for i in xrange(N):
    for j in xrange(N):
        km[i,j] = km[i,j]*relate_tasks(i,j)
        #km = km*1.0

print km
#precompute kernel matrix using shogun
y = numpy.array(labels)
K = numpy.transpose(y.flatten() * (km*y.flatten()).transpose())
f = -numpy.ones(N)
C = 1.0

# Important!! QP does not accept ndarray as a type, it must be an array
p = QP(K, f, Aeq=y, beq=0, lb=numpy.zeros(N), ub=C*numpy.ones(N))
r = p.solve('cvxopt_qp', iprint = 0)

#print "cvxopt objective:", r.ff
print "externally modified kernel. objective:", r.ff

ck = CustomKernel()
ck.set_full_kernel_matrix_from_full(km)
#
svm = LibSVM(1, ck, lab)
svm.train()

print "externally modified kernel. objective:", svm.get_objective()


