from shogun.Kernel import LinearKernel
from shogun.Classifier import LibSVM, SVMLight, DomainAdaptationSVM, DomainAdaptationSVMLinear, LibLinear, MSG_DEBUG, L2R_L1LOSS_SVC_DUAL

from shogun.Features import Labels, RealFeatures
from shogun.Library import Math_init_random

import numpy
import helper



#############################################
#    debugging data
#############################################

def get_presvm(B=2.0):

    examples_presvm = [numpy.array([ 2.1788894 ,  3.89163458,  5.55086917,  6.4022742 ,  3.14964751, -0.4622959 ,  5.38538904,  5.9962938 ,  6.29690849]),
     numpy.array([ 2.1788894 ,  3.89163458,  5.55086917,  6.4022742 ,  3.14964751,  -0.4622959 ,  5.38538904,  5.9962938 ,  6.29690849]),
     numpy.array([ 0.93099452,  0.38871617,  1.57968949,  1.25672527, -0.8123137 ,   0.20786586,  1.378121  ,  1.15598866,  0.80265343]),
     numpy.array([ 0.68705535,  0.15144113, -0.81306157, -0.7664577 ,  1.16452945,  -0.2712956 ,  0.483094  , -0.16302007, -0.39094812]),
     numpy.array([-0.71374437, -0.16851719,  1.43826895,  0.95961166, -0.2360497 ,  -0.30425755,  1.63157052,  1.15990427,  0.63801465]),
     numpy.array([ 0.68705535,  0.15144113, -0.81306157, -0.7664577 ,  1.16452945, -0.2712956 ,  0.483094  , -0.16302007, -0.39094812]),
     numpy.array([-0.71374437, -0.16851719,  1.43826895,  0.95961166, -0.2360497 , -0.30425755,  1.63157052,  1.15990427,  0.63801465]),
     numpy.array([-0.98028302, -0.23974489,  2.1687206 ,  1.99338824, -0.67070205, -0.33167281,  1.3500379 ,  1.34915685,  1.13747975]),
     numpy.array([ 0.67109612,  0.12662017, -0.48254886, -0.49091898,  1.31522237, -0.34108933,  0.57832179, -0.01992828, -0.26581628]),
     numpy.array([ 0.3193611 ,  0.44903416,  3.62187778,  4.1490827 ,  1.58832961,  1.95583397,  1.36836023,  1.92521945,  2.41114998])]
    labels_presvm = [-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0]

    examples = [numpy.array([-0.49144487, -0.19932263, -0.00408188, -0.21262012,  0.14621013, -0.50415481,  0.32317317, -0.00317602, -0.21422637]), 
     numpy.array([ 0.0511817 , -0.04226666, -0.30454651, -0.38759116,  0.31639514,  0.32558471,  0.49364473,  0.04515591, -0.06963456]),
     numpy.array([-0.30324369, -0.11909251, -0.03210278, -0.2779561 ,  1.31488853, -0.33165365,  0.60176018, -0.00384946, -0.15603975]),
     numpy.array([ 0.59282756, -0.0039991 , -0.26028983, -0.26722552,  1.63314995, -0.51199338,  0.33340685, -0.0170519 , -0.19211039]),
     numpy.array([-0.18338766, -0.07783465,  0.42019824,  0.201753  ,  2.01160098,  0.33326111,  0.75591909,  0.36631525,  0.1761829 ]),
     numpy.array([ 0.10273793, -0.02189574,  0.91092358,  0.74827973,  0.51882902, -0.1286531 ,  0.64463658,  0.67468349,  0.55587266]),
     numpy.array([-0.09727099, -0.13413522,  0.18771062,  0.19411594,  1.48547364, -0.43169608,  0.55064534,  0.24331473,  0.10878847]),
     numpy.array([-0.22494375, -0.15492964,  0.28017737,  0.29794467,  0.96403895,  0.43880289,  0.08053425,  0.07456818,  0.12102371]),
     numpy.array([-0.18161417, -0.17692039,  0.19554942, -0.00785625,  1.38315115, -0.05923183, -0.05723568, -0.15463646, -0.24249483]),
     numpy.array([-0.36538359, -0.20040061, -0.38384388, -0.40206556, -0.25040256,  0.94205875,  0.40162798,  0.00327328, -0.24107393])]

    labels = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0]

    examples_test = [numpy.array([-0.45159799, -0.11401394,  1.28574573,  1.09144306,  0.92253119,  -0.47230164,  0.77032486,  0.83047366,  0.74768906]),
     numpy.array([ 0.42613105,  0.0092778 , -0.78640296, -0.71632445,  0.41154244,   0.88380309,  0.19475759, -0.14195876, -0.30479425]),
     numpy.array([-0.09727099, -0.13413522,  0.18771062,  0.19411594,  1.48547364,  -0.43169608,  0.55064534,  0.24331473,  0.10878847]),
     numpy.array([ 0.11558796, -0.08867647, -0.26432074, -0.30924546, -1.08243017,  -0.1339607 , -0.1956124 , -0.2428358 , -0.25761213]),
     numpy.array([ 1.23679696,  0.18753081, -0.25593329, -0.12051991,  0.64976989,  -0.17184101,  0.14951337,  0.01988587, -0.0356698 ]),
     numpy.array([ 1.03355002,  0.05316195, -0.97905368, -0.75482121,  0.28673776,   2.27142733,  0.02654739, -0.31109851, -0.44555277]),
     numpy.array([-0.53662325, -0.21434756, -0.12105795, -0.27531257,  0.66947047,   0.05474302, -0.00717455, -0.17700575, -0.22253444]),
     numpy.array([ 0.11272632, -0.12674826, -0.49736457, -0.51445609,  0.88518932,  -0.51558669, -0.12000557, -0.32973613, -0.38488736]),
     numpy.array([ 0.8372111 ,  0.06972199, -1.00454229, -0.79869642,  1.19376333,  -0.40160273, -0.25122157, -0.46417918, -0.50234858]),
     numpy.array([-0.36325018, -0.12206184,  0.10525247, -0.15663416,  1.03616948,  -0.51699463,  0.59566286,  0.35363369,  0.10545559])]


    #############################################
    #    compute pre-svm
    #############################################


    # create real-valued features as first step
    examples_presvm = numpy.array(examples_presvm, dtype=numpy.float64)
    examples_presvm = numpy.transpose(examples_presvm)

    feat_presvm = RealFeatures(examples_presvm)
    lab_presvm = Labels(numpy.array(labels_presvm))
    wdk_presvm = LinearKernel(feat_presvm, feat_presvm)



    presvm_liblinear = LibLinear(1, feat_presvm, lab_presvm)
    presvm_liblinear.set_max_iterations(10000)
    presvm_liblinear.set_bias_enabled(False)
    presvm_liblinear.train()


    #return presvm_liblinear


    #def get_da_svm(presvm_liblinear):


    #############################################
    #    compute linear term manually
    #############################################

    examples = numpy.array(examples, dtype=numpy.float64)
    examples = numpy.transpose(examples)

    feat = RealFeatures(examples)
    lab = Labels(numpy.array(labels))

    dasvm_liblinear = DomainAdaptationSVMLinear(1.0, feat, lab, presvm_liblinear, B)
    dasvm_liblinear.set_bias_enabled(False)
    dasvm_liblinear.train()

    helper.save("/tmp/svm", presvm_liblinear)
    presvm_pickle = helper.load("/tmp/svm")

    dasvm_pickle = DomainAdaptationSVMLinear(1.0, feat, lab, presvm_pickle, B)
    dasvm_pickle.set_bias_enabled(False)
    dasvm_pickle.train()

    helper.save("/tmp/dasvm", dasvm_liblinear)
    dasvm_pickle2 = helper.load("/tmp/dasvm")

    #############################################
    #    load test data
    #############################################

    examples_test = numpy.array(examples_test, dtype=numpy.float64)
    examples_test = numpy.transpose(examples_test)
    feat_test = RealFeatures(examples_test)

    # check if pickled and unpickled classifiers behave the same
    out1 = dasvm_liblinear.classify(feat_test).get_labels()
    out2 = dasvm_pickle.classify(feat_test).get_labels()

    # compare outputs
    for i in xrange(len(out1)):    
        
        try:
            assert(abs(out1[i]-out2[i])<= 0.001)
        except:
            print "(%.5f, %.5f)" % (out1[i], out2[i])

            
    print "classification agrees."


if __name__ == "__main__":
    
    get_presvm()
