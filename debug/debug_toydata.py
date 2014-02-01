import sys
import getopt

import numpy

from shogun.Shogun import WeightedDegreeStringKernel
from shogun.Shogun import StringCharFeatures
from shogun.Shogun import DNA
from shogun.Shogun import SVMLight
from shogun.Shogun import Labels

import helper




def read_files(file_name, shuffle=False):


    examples = []
    labels = []

    for line in file(file_name):

        if line.startswith(">"):
            
            if line.find(">1") != -1:
                labels.append(1.0)
            if line.find(">-1") != -1:
                labels.append(-1.0)
                
        if not line.startswith(">") and line != "":
            examples.append(line.upper().strip())
                    

    if shuffle:
        idx = numpy.random.permutation(range(len(examples)))        
        examples = numpy.array(examples)[idx].tolist()
        labels = numpy.array(labels)[idx].tolist()


    return (examples, labels)
    


def train_test(train_examples_fn, test_examples_fn):


    (train_examples, train_labels) = read_files(train_examples_fn)

    

    feat_train = StringCharFeatures(DNA)
    feat_train.set_string_features(train_examples)
    wdk = WeightedDegreeStringKernel(feat_train, feat_train, 4, 0)
    lab = Labels(numpy.array(train_labels))
    
    
    svm = SVMLight(1.0, wdk, lab)
    svm.train()

    print "svm objective", svm.get_objective()
    

    # testing part
    (test_examples, test_labels) = read_files(test_examples_fn)
    
    feat_test = StringCharFeatures(DNA)
    feat_test.set_string_features(test_examples)
    
    
    svm.get_kernel().init(feat_train, feat_test)
    

    svm_out = svm.classify().get_labels()


    performance = helper.calcprc(svm_out, test_labels)[0]

    print ""
    print ""
    print "========================"
    
    print "auPRC", performance

    return performance

    

def main():
    """
    delegates work
    """

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])

    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)

    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)


    if len(args) != 2:
        print "usage: debug_toydata.py train_examples test_examples"

    else:
        train_test(args[0], args[1])


    
if __name__ == "__main__":

    main()
    
