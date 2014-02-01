'''
Created on Sep 14, 2009

@author: cwidmer
'''

import numpy
import expenv
import pylab
import expenv_runner
import analysis

from pythongrid import KybJob, process_jobs 


    
    
def plot_analysis(idx, target="test", scale="log", task_id="toy_1"):
    
    x = [] 
    y = []
    
    
    print "target:", target
    print "task_id:", task_id
    
    title = "task_id: " + task_id + " [" + str(min(idx)) + "," + str(max(idx)) + "]"
    
    for j in idx:
        print j
        if target=="roc":
            y.append([a.auROC for a in expenv.Experiment.get(j).test_run.assessment.assessments if a.task_id==task_id][0])
        elif target=="test":
            y.append([a.auPRC for a in expenv.Experiment.get(j).test_run.assessment.assessments if a.task_id==task_id][0])
        else:
            y.append(expenv.Experiment.get(j).best_mean_performance)
            
        x.append(float(expenv.Experiment.get(j).description.split(" ")[-1]))
    
    
    print "len x", len(x)
    print "len y", len(y)
    
    myidx = numpy.argsort(x)
    x = numpy.array(x)[myidx]
    y = numpy.array(y)[myidx]
    
    pylab.title(title)
    pylab.gca().get_yaxis().grid(True)
    pylab.gca().get_xaxis().grid(True)
    if scale=="linear":
        pylab.plot(x, y, "-o")
    else:
        pylab.semilogx(x, y, "-o")
    pylab.show()
    
    print "argmax y:", numpy.argmax(y)
    print "max x:", x[numpy.argmax(y)]
    print "max y:", y[numpy.argmax(y)]
    
    
    
def run_job(myarg, comment):
    
    tmp = comment + " " + str(myarg)
    
    exp_id = expenv_runner.run_multi_example(317, "method_hierarchy_svm", tmp, myarg)
    
    return exp_id



if __name__ == '__main__':
    
    #Bs = numpy.double(range(0,41))/4
    #Bs = numpy.double(range(0,2))
    #Bs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 3.0, 10.0, 100.0, 1000.0]
    Bs = [pow(10, i) for i in (numpy.double(range(0, 41))/10 - 4)] #[pow(10, j) for j in (numpy.double(range(0, 21))/10 - 2)]
    
    jobs = []

    mycomment = "global C=1.0, toy_0"

    for B in Bs:
        job = KybJob(analysis.run_job, [B, mycomment])
        jobs.append(job)

    
    idx = [job.ret for job in process_jobs(jobs, True, 4)]
    
    print "using idx", idx
    
    plot_analysis(idx)

    