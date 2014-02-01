from shogun.Shogun import WeightedDegreeStringKernel
from shogun.Shogun import StringCharFeatures
from shogun.Shogun import DNA
from shogun.Shogun import LibSVM, SVMLight
from shogun.Shogun import Labels

from openopt import QP
from scipy import io
import scipy.optimize

import cvxopt.base as cb
import cvxopt.solvers as cs
import cvxopt.mosek as mosek
import cvxopt

import expenv
import numpy

run = expenv.Run.get(1010)
#run = expenv.Run.get(974)
dat = run.get_train_data()

print dat.keys()
d = dat["thaliana"]
subset_size = 20

examples = [i.example for i in d[0:subset_size]]
labels = [i.label for i in d[0:subset_size]]

labels[2] = 1
labels[12] = 1
labels[15] = 1
labels[8] = 1
labels[19] = 1


feat = StringCharFeatures(DNA)
feat.set_string_features(examples)
wdk = WeightedDegreeStringKernel(feat, feat, 1, 0)
lab = Labels(numpy.array(labels))


svm = LibSVM(1, wdk, lab)
svm.train()

svm.set_shrinking_enabled(False)

print "simple svm", svm.get_objective()

print "len(examples)", len(examples)

print "##############"


#print "##############"
#print "svm light"

#svm_light = SVMLight(1.0,wdk,lab)

#svm_light.train()


#print "svmlight objective", svm_light.get_objective()


y = numpy.array(labels)

N = len(y)

#f = -numpy.array(range(subset_size))

print "python N:", N

f = -numpy.ones(N)
 

for idx in xrange(10):

    #f = (-numpy.ones(N)-2)*numpy.random.randn()


    print "############################"
    print "############################"
    print ""
    print "f:", f
    print "\n"

    svm.train_reg(f)

    sv_idx = svm.get_support_vectors()
    alphas = svm.get_alphas()

    alphas_full = numpy.zeros(N)
    alphas_full[sv_idx] = alphas

    alphas_full = alphas_full * y

    print "libsvm objective:", svm.get_objective()
    print "libsvm alphas:", numpy.array(alphas_full[0:5])

    external_objective = 0.0

    for j in xrange(N):

        external_objective += alphas_full[j] * f[j]

        for k in xrange(N):

            external_objective += 0.5 * alphas_full[j] * alphas_full[k] * y[j] * y[k] * wdk.kernel(j,k)

    print "libsvm external objective:", external_objective




    #precompute kernel matrix using shogun
    K = wdk.get_kernel_matrix()
    K = numpy.transpose(y.flatten() * (K*y.flatten()).transpose())

    C = 1.0
    # Important!! QP does not accept ndarray as a type, it must be an array
    p = QP(K, f, Aeq=y, beq=0, lb=numpy.zeros(N), ub=C*numpy.ones(N))
    r = p.solve('cvxopt_qp', iprint = 0)

    print "cvxopt objective:", r.ff
    print "cvxopt alphas:", numpy.array(r.xf[0:5])


    external_objective = 0.0

    for j in xrange(N):

        external_objective += r.xf[j] * f[j]

        assert(r.xf[j] >= -0.001)

        for k in xrange(N):
            external_objective += 0.5 * r.xf[j] * r.xf[k] * K[j,k]
  

    print "cvxopt external_objective:", external_objective
    print "cvxopt external_objective2:", 0.5 * numpy.dot(numpy.dot(r.xf.transpose(), K), r.xf) + numpy.dot(r.xf.transpose(), f)

    print "##################"
    #print "mosek"


    f = numpy.random.rand(N) - numpy.ones(N)

    #matrix for bounds
    #G = numpy.zeros((2*N,N))
    #G[0:N,:] = numpy.eye(N)
    #G[N:2*N,:] = -numpy.eye(N)

    #h = numpy.zeros(2*N)
    #h[0:N] = C

    ##cvxopt
    ##sol = solvers.qp(K, f, G, h, y, 0)

    #print "y:", y.shape, "f:", f.shape
    #y = y.reshape(1,len(y))
    #f = f.reshape(len(f),1)

    #K = wdk.get_kernel_matrix()
    #K = numpy.transpose(y.flatten() * (K*y.flatten()).transpose())

    ##set up QP
    #Q = cvxopt.matrix(K)
    #p = cvxopt.matrix(f)
    #G = cvxopt.matrix(G)
    #h = cvxopt.matrix(h)
    #A = cvxopt.matrix(y)
    #b = cvxopt.matrix(0.0)

    #print Q

    #(solsta, x, z, y) = mosek.qp(Q, p, G, h, A, b)

    #print "x:", len(x)
    #print "y:", len(y)
    #print "z:", len(z)

    #alphas = numpy.array(x).flatten()

    #print alphas
    #print x,z,y

    ##print z
    ##print y

    ##print numpy.array(y)
    ##print numpy.array(y).shape

    #F = y[0,0]


