#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Written (W) 2009 Christian Widmer
# Copyright (C) 2009 Max-Planck-Society

"""
created on 08.04.2009
@author: Christian Widmer
@summary: simple viewer for experiments written in QT4

additionally, this is supposed to help generate plots easily


"""

import re
import os
import sys 
from PyQt4.QtCore import * 
from PyQt4.QtGui import *

from PyQt4 import QtCore, QtGui


 
import numpy
import pylab
from matplotlib.font_manager import fontManager, FontProperties
import numpy.numarray as na
from collections import defaultdict

import expenv
import helper



def plot_convex_combination(exp_id):
    
    
    e = expenv.Experiment.get(exp_id)
    p = numpy.double(range(0,101))/100
    a0 = e.test_run.assessment.assessments[0]
    a1 = e.test_run.assessment.assessments[1]
    perf = []
    for i in p:
        a = i*a0.pred + (1-i)*a1.pred
        perf.append(helper.calcprc(a, a0.lab)[0])
    
    pylab.gca().get_yaxis().grid(True)
    pylab.gca().get_xaxis().grid(True)
    pylab.plot(p,perf, "-o")
    
    print "argmax:", numpy.argmax(perf)
    print "max x:", p[numpy.argmax(perf)]
    print "max y:", numpy.max(perf)
    
    
    pylab.show()




def detailed_bar_plot(data, labels, methods, plot_title="", ylabel="auROC"):
    """
    useful for visualizing results from test set

    @param data - data to plot, for each method, performance for each organism
    @type data - list<list<float>>
    @param labels - tic labels for xaxis
    @type labels - list<str>
    @param plot_title - name of plot
    @type plot_title - str


    xlocations = na.array(range(len(data)))+0.5
    """

    pylab.figure()

    mod_methods = []

    for l in methods:

        #if plot_title=="":
        #    plot_title = re.findall(".+\(", l)[0][0:-1]
    

        mod_methods.append(l)

    used_colors = ["green", "blue", "red", "yellow", "purple", "steelblue", "brown", "orange", "pink"]*10
    #used_colors = colors[0:len(data)]

    width = 0.20
    separator = 0.15
   

    offset = 0
   
    num_methods = len(methods)


    xlocations = []


    print "len(data):", len(data)

    for org_perf in data:
        
        offset += separator

        rects = []
    
        print "len(org_perf):", len(org_perf)

        xlocations.append(offset + num_methods*width/2)

        for (i, method_perf) in enumerate(org_perf):
            
            rects.append(pylab.bar(offset, method_perf, width, color=used_colors[i]))

            offset += width

        offset += separator


    # determine extrema
    ymax = max([max(row) for row in data])*1.1
    ymin = min([min(row) for row in data])*0.9


    # set ticks
    tick_step = 0.05
    ticks = [tick_step*i for i in xrange(round(ymax/tick_step)+1)]

    pylab.yticks(ticks)

    print "xlocations:", xlocations
    print "labels:", labels

    pylab.xticks(xlocations, labels)

    fontsize=17
    ax = pylab.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    pylab.xlim(0, offset)
    pylab.ylim(ymin, ymax)

    pylab.title(plot_title)
    pylab.gca().get_xaxis().tick_bottom()
    pylab.gca().get_yaxis().tick_left()

    pylab.gca().get_yaxis().grid(True)
    pylab.gca().get_xaxis().grid(False)

    pylab.legend( tuple(rects), tuple(mod_methods), loc="upper right")
    
    
    pylab.ylabel(ylabel, fontsize = 20)
    
    
    pylab.show()




def bar_plot(data, labels, plot_title=""):
    """
    useful for visualizing results from test set
    """

    import numpy.numarray as na
    from pylab import *
    import re

    pylab.figure()

    mod_labels = []

    for l in labels:

        if plot_title=="":
            plot_title = re.findall(".+\(", l)[0][0:-1]
    
        lab = re.findall("\(\D+\)", l)[0]
        if lab:
            mod_labels.append(lab)
        else:
            mod_labels.append(l)


    xlocations = na.array(range(len(data)))+0.5
    width = 0.5
    #bar(xlocations, data, yerr=error, width=width)
    bar(xlocations, data, width=width)

    ymax = max(data)*1.2
    #ymin = max(data)*1.2

    ticks = [0.1*i for i in xrange(round(ymax/0.1)+1)]

    yticks(ticks)
    xticks(xlocations+ width/2, mod_labels)
    xlim(0, xlocations[-1]+width*2)

    #spacing = myrange*0.1
    #pylab.ylim( (min_y-spacing, max_y+spacing) )
    ylim(0, ymax)
    title(plot_title)
    gca().get_xaxis().tick_bottom()
    gca().get_yaxis().tick_left()

    gca().get_yaxis().grid(True)
    gca().get_xaxis().grid(False)
    
    show()


def main(): 
    app = QApplication(sys.argv) 
    w = MyWindow() 
    w.show() 
    sys.exit(app.exec_()) 


def get_table_data():
    
    experiments = list(expenv.Experiment.select().orderBy(["id"]))
    
    tabledata= []
    row_to_id = []
    
    for ex in experiments:
        
        row = []

        row_to_id.append(ex.id)

        row.append(ex.id)
        row.append(str(ex.timestamp))
        row.append("%.4f + (%.4f)" % (ex.best_mean_performance, ex.best_std_performance) )
        
        try:
            row.append(ex.test_run.assessment.auROC)
        except Exception, detail:
            #print detail
            row.append("")
        
        try:
            row.append(ex.test_run.assessment.auPRC)
        except Exception, detail:
            #print detail
            row.append("")
            
        row.append(ex.description)
        
        tabledata.append(row)

    return (tabledata, row_to_id)



class MyWindow(QWidget): 


    def __init__(self, *args): 
        QWidget.__init__(self, *args) 

        # create table and load data
        self.createTable()
        self.data_table = DataTableView()

        tab_widget = QTabWidget() 
        tab1 = QWidget() 
        tab2 = QWidget() 
         
        p1_vertical = QVBoxLayout(tab1) 
        p2_vertical = QVBoxLayout(tab2)
         
        tab_widget.addTab(tab1, "Data") 
        tab_widget.addTab(tab2, "Experiments") 
        
        tree_widget = DataTreeView()
        p1_vertical.addWidget(tree_widget)
        p2_vertical.addWidget(self.table)
        
        # settins 
        #self.settings = HyperparameterSettingsFrame()
        self.settings = BottomWidget()
        p2_vertical.addWidget(self.settings)
        
        
         
        # layout
        layout = QVBoxLayout()
        layout.addWidget(tab_widget)
        
        self.setLayout(layout) 
         
       
        # connect signals and slots
        self.connect(self.table, SIGNAL("clicked(QModelIndex)"), self.display_info)
        self.connect(self.table, SIGNAL("refresh()"), self.reload_view)
        self.connect(self.table, SIGNAL("cleanup()"), self.clean_up)
        self.connect(self.table, SIGNAL("plot()"), self.plot)
        self.connect(self.table, SIGNAL("detailed_plot()"), self.detailed_plot)
        self.connect(self.table, SIGNAL("plot_roc_curve()"), self.plot_roc_curve)
        self.connect(self.table, SIGNAL("detailed_training_plot()"), self.detailed_training_plot)
        self.connect(self.table, SIGNAL("hyperparameter_plot()"), self.hyperparameter_plot)
        self.connect(self.table, SIGNAL("hyperparameter_plot_test()"), self.hyperparameter_plot_test)
        self.connect(self.table, SIGNAL("show_graph()"), self.show_graph)
        self.connect(self.table, SIGNAL("create_tex_table()"), self.create_tex_table)
        
        # deep connections
        self.connect(self.settings.hyper.hyper_plot_button, SIGNAL("clicked()"), self.hyperparameter_plot)
        self.connect(self.settings.hyper.task_plot_button, SIGNAL("clicked()"), self.detailed_plot)
        
        
        
        self.setMinimumWidth(800)


    def plot(self, target="auPRC"):
        """
        display main results using a bar plot from matplotlib
        """


        rm_idx = set([self.row_to_id[idx.row()] for idx in self.table.selectedIndexes()])

        data = []
        labels = []

        for id in rm_idx:

            try:
                e = expenv.Experiment.get(id)
                data.append(getattr(e.test_run.assessment, target))
                labels.append(e.description)

            except Exception, detail:
                print detail

        bar_plot(data, labels)



    def create_tex_table(self, target="auROC"):
        """
        display detailed results using a bar plot from matplotlib
        """


        rm_idx = set([self.row_to_id[idx.row()] for idx in self.table.selectedIndexes()])

        
        plot_title = "" #medium mutation rate"
        
        #sort methods by performance
        experiments = [expenv.Experiment.get(id) for id in rm_idx]
        

        
        data = defaultdict(dict)
        
        columns = set()
        methods = set()
        
        for experiment in experiments:
            
            method_name = re.findall("\(method_(\w+)_\w+\)", experiment.description)[0]
            method_name = method_name.replace("augmented", "multikernel").replace("hierarchy", "top-down")

            #print method_name
            
            dataset_name = re.findall("^(deep\d+)", experiment.description)[0]
        
            columns.add(dataset_name)
            methods.add(method_name)
            
            performance = experiment.test_run.assessment.auROC
            
            data[method_name][dataset_name] = performance
            

        #sort column names
        columns = list(columns)
        columns.sort()
        
        columns = ["deep8", "deep16", "deep32", "deep64"]
        
        x = [8, 16, 32, 64]
        
        xlabels = [str(i) + " tasks" for i in x]
        
        print data
        
        methods = ["plain", "union", "pairwise", "multikernel", "top-down"]
        
        for method in methods:
            
            y = []
            
            for column in columns:
                
                if data[method].has_key(column):
                    y.append(data[method][column])
                else:
                    y.append(0.0)
        
            print y
            #pylab.plot(x, y, "-o")
            pylab.semilogx(x, y, "-o")
        
        
        #pylab.yticks(ticks)
    
        #print "xlocations:", xlocations
        #print "labels:", labels
    
        pylab.xticks(x, xlabels)
        pylab.xlim(7, 70)
        #pylab.ylim(0.60, 0.85)
        pylab.ylim(0.62, 0.76)
    
        pylab.title(plot_title)
        pylab.gca().get_xaxis().tick_bottom()
        pylab.gca().get_yaxis().tick_left()
    
        pylab.gca().get_yaxis().grid(True)
        pylab.gca().get_xaxis().grid(True)
    
        #pylab.legend( tuple(rects), tuple(methods), loc="upper left")
        #pylab.legend(tuple(methods), loc="lower left")
        #pylab.legend(tuple(methods), loc="upper left")
        #pylab.legend(tuple(methods), loc="upper right")
        #pylab.legend(tuple(methods), loc="best")
        
        pylab.ylabel(target, fontsize = "large")
        pylab.show()
        
 
       
    def plot_roc_curve(self):
        """
        display detailed results using a ROC plot
        """


        target = str(self.settings.hyper.target_measure_combo.currentText())

        rm_idx = set([self.row_to_id[idx.row()] for idx in self.table.selectedIndexes()])

        data = []
        labels = 0
        methods = []
        
        plot_title = "ROC-curve"#splicing data"
        
        #sort methods by performance
        experiments = [expenv.Experiment.get(id) for id in rm_idx]
        performances = [getattr(e.test_run.assessment, target) for e in experiments]
        sorted_idx = numpy.argsort(performances)
        
        
        exp_idx = [experiments[id].id for id in sorted_idx]

        
        data = defaultdict(list)        
        

        for id in exp_idx:

            try:

                e = expenv.Experiment.get(id)
        
                #plot_title = re.findall(".+\(", e.description)[0][0:-1] + " (testing)"
                
                labels = numpy.sort([a.task_id for a in e.test_run.assessment.assessments]).tolist()

                for a in e.test_run.assessment.assessments:
                    entry = a.load_output_and_labels()
                    print "loaded", len(entry["out"]), "outputs"
                    data[a.task_id].append(entry)
                

                # parse method name from description string
                method_name = e.description
                tmp_name = re.findall("\(.+\)", method_name)[0]
                
                if tmp_name:
                    #methods.append("exp " + str(e.id) + ": " + tmp_name)
                    methods.append(tmp_name.replace("(","").replace(")","").replace("method_","").replace("_svm","").replace("_multitask","").replace("augmented", "multikernel").replace("hierarchy","top-down"))
                else:
                    methods.append("exp " + str(e.id) + ": " + method_name)
                    

                if labels!=0:
                    print labels
                    assert(labels == [a.task_id for a in e.test_run.assessment.assessments])


            except Exception, detail:
                print detail


        
        for key, value in data.items():
       


            # Plot ROC curve
            pylab.figure()
            pylab.clf()
            pylab.plot([0, 1], [0, 1], 'k--')
            pylab.xlim([0.0,1.0])
            pylab.ylim([0.0,1.0])
            pylab.xlabel('False Positive Rate')
            pylab.ylabel('True Positive Rate')
            pylab.title(key)

            plots = []

            for item in value:
                out = item["out"]
                lab = item["lab"]

                roc_auc, tpr, fpr = helper.calcroc(out, lab)
                print "Area under the ROC curve : %f" % roc_auc
                plots.append(pylab.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc))


            #print labels, plots
            pylab.legend(tuple(plots), tuple(methods), loc="lower right")
            pylab.show()
            
       

    def detailed_plot(self, plot_average=True):
        """
        display detailed results using a bar plot from matplotlib
        """


        target = str(self.settings.hyper.target_measure_combo.currentText())

        rm_idx = set([self.row_to_id[idx.row()] for idx in self.table.selectedIndexes()])

        data = []
        labels = 0
        methods = []
        
        plot_title = ""#splicing data"
        
        #sort methods by performance
        experiments = [expenv.Experiment.get(id) for id in rm_idx]
        performances = [getattr(e.test_run.assessment, target) for e in experiments]
        sorted_idx = numpy.argsort(performances)
        
        
        rm_idx = [experiments[id].id for id in sorted_idx]
        
        

        for id in rm_idx:

            try:
                e = expenv.Experiment.get(id)
        
                #plot_title = re.findall(".+\(", e.description)[0][0:-1] + " (testing)"
                
                labels = numpy.sort([a.task_id for a in e.test_run.assessment.assessments]).tolist()
                #if labels[0]=='':
                #    labels = [instances for instances in expenv.Experiment.get(275).test_run.get_eval_data()]
                idx = numpy.argsort([a.task_id for a in e.test_run.assessment.assessments])
                sub_assessments = [getattr(a, target) for a in e.test_run.assessment.assessments]
                sub_assessments = numpy.array(sub_assessments)[idx].tolist()
                
                #TODO remove this is for hard-coded results only!!
                #orgs = ['A.nidulans', 'P.trichocarpa', 'A.thaliana', 'O.sativa', 'C.savignyi', 'D.rerio', 'O.latipes', 'G.aculeatus', 'T.nigroviridis', 'B.taurus', 'H.sapiens', 'M.musculus', 'C.elegans', 'A.gambiae', 'D.melanogaster']
                #labels = orgs
                #sub_assessments = []
                #for org in orgs:
                
                #    tmp = [getattr(a, target) for a in e.test_run.assessment.assessments if a.task_id==org][0]
                #    sub_assessments.append(tmp) 

                # parse method name from description string
                method_name = e.description
                tmp_name = re.findall("\(.+\)", method_name)[0]
                
                if tmp_name:
                    #methods.append("exp " + str(e.id) + ": " + tmp_name)
                    methods.append(tmp_name.replace("(","").replace(")","").replace("method_","").replace("_svm","").replace("_multitask","").replace("augmented", "multikernel").replace("hierarchy","top-down"))
                else:
                    methods.append("exp " + str(e.id) + ": " + method_name)

                
                if plot_average:
                    sub_assessments.append(numpy.mean(sub_assessments))
                    
                data.append(sub_assessments)

                if labels!=0:
                    print labels
                    assert(labels == [a.task_id for a in e.test_run.assessment.assessments])


            except Exception, detail:
                print detail

        if plot_average:
            labels.append("mean")
       
        # transpose data
        data_T = []


        print "__data", len(data)
        print "__labels", len(labels)

        print "labels:", labels

        for i in xrange(len(labels)):

            data_row = []

            for j in xrange(len(data)):
                
                data_row.append(data[j][i])

            data_T.append(data_row)

        
        #plot_title = "performance on test set"

        detailed_bar_plot(data_T, labels, methods, plot_title, target)



    def show_graph(self):
        """
        display graph if one exists
        """


        idx = set([self.row_to_id[idx.row()] for idx in self.table.selectedIndexes()])

        for id in idx:

            e = expenv.Experiment.get(id)
            
            if e.test_run.method.module_name.find("hierarchy") != -1:
                
                param_id = e.test_run.method.param.id
                
                mypath = "/fml/ag-raetsch/share/projects/multitask/graphs/"
                filename = mypath + "graph_" + str(param_id) + ".png"
                
                os.system("evince " + filename + " &")
                
                    
    def hyperparameter_plot_test(self, use_max=True):
        """
        simple wrapper
        """
        
        self.hyperparameter_plot(use_max=use_max, plot_test=True)
        
        
    
    def hyperparameter_plot(self, use_max=True, plot_test=False):
        """
        display a performance plot for every hyper-parameter including error bar

        auto-detect over which parameter we optimize
        """

        #target_param = str(self.settings.target_param_combo.itemData(self.settings.target_param_combo.currentIndex()).toString())
        #target_measure = str(self.settings.target_measure_combo.itemData(self.settings.target_measure_combo.currentIndex()).toString())
        #semilogx = str(self.settings.semi_logx_combo.itemData(self.settings.semi_logx_combo.currentIndex()).toString())
        
        target_param = str(self.settings.hyper.target_param_combo.currentText())
        target_measure = str(self.settings.hyper.target_measure_combo.currentText())
        semilogx = str(self.settings.hyper.semi_logx_combo.currentText())
        
        
        log_x = False
        if semilogx == "True":
            log_x = True
        

        pylab.figure()

        exp_ids = set([self.row_to_id[idx.row()] for idx in self.table.selectedIndexes()])

        exp_names = []
        curves = []

        plot_title = ""

        for id in exp_ids:

            try:
                e = expenv.Experiment.get(id)
                exp_names.append(str(e.id) + " " + e.description)

                plot_title = re.findall(".+\(", e.description)[0][0:-1]

                runs = [run for run in e.eval_runs]

                #num_tasks = 0

                # in order to avrg over splits, we need to find unique x-val values
                unique_params = list(set([getattr(run.method.param, target_param) for run in runs]))
                unique_params.sort()
     
     
                split_idx = list(set([int(run.x_val_idx) for run in runs]))
                split_idx.sort()
                
                y_values = []
                y_values_test = []

                std_values = []
                std_values_test = []

                run_map = defaultdict(list)

                # map runs to unique parameters
                for run in runs:
                    param_value = getattr(run.method.param, target_param)
                    run_map[param_value].append(run)

                # extract performances
                for unique_param in unique_params:
                    
                    assessment_param = []
                    assessment_test_param = []
                    
                    if use_max:
                        
                        print "taking max value for each point"
                        
                        for split_id in split_idx:
                            tmp_assessment = [getattr(run.assessment, target_measure) for run in run_map[unique_param] if hasattr(run.assessment, target_measure) and run.x_val_idx==split_id]
                            
                            if plot_test:
                                tmp_assessment_test = [getattr(run.assessment_test, target_measure) for run in run_map[unique_param] if hasattr(run.assessment_test, target_measure) and run.x_val_idx==split_id]
                            
                            
                            if len(tmp_assessment) > 0:            
                                max_id = numpy.argmax(tmp_assessment)
                                max_value = tmp_assessment[max_id] 
                                assessment_param.append(max_value)
                                
                                if plot_test:
                                    test_value = tmp_assessment_test[max_id]
                                    assessment_test_param.append(test_value)
                        
                    else:

                        print "taking average value for each point"
                        assessment_param = [getattr(run.assessment, target_measure) for run in run_map[unique_param] if hasattr(run.assessment, target_measure)]



                    if plot_test:
                        y_values_test.append(numpy.mean(assessment_test_param))
                        std_values_test.append(numpy.std(assessment_test_param))
                    else:
                        y_values.append(numpy.mean(assessment_param))
                        std_values.append(numpy.std(assessment_param))

                    
                    
                #curve = pylab.errorbar(unique_params, y_values, yerr=std_values)
                if log_x:
                    
                    if plot_test:
                        curve_test = pylab.semilogx(unique_params, y_values_test, "-o")
                    else:
                        curve = pylab.semilogx(unique_params, y_values, "-o")
                        
                else:
                    
                    if plot_test:
                        curve_test = pylab.plot(unique_params, y_values_test, "-o")
                    else:
                        curve = pylab.plot(unique_params, y_values, "-o")
                    
                
                
                if plot_test:
                    curves.append(curve_test)
                else:
                    curves.append(curve)
                

            except Exception, detail:
                import traceback
                traceback.print_exc(file=sys.stdout)
                print detail

            # render plot


            fontsize=14
            ax = pylab.gca()
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)

            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)
            #pylab.xlim(0, offset)
            #pylab.ylim(ymin, ymax)

            pylab.title(plot_title)
            pylab.gca().get_xaxis().tick_bottom()
            pylab.gca().get_yaxis().tick_left()

            pylab.gca().get_yaxis().grid(True)
            pylab.gca().get_xaxis().grid(False)

            new_experiment_names = []
            for name in exp_names:
                
                if plot_test:
                    new_experiment_names.append(name + " (test)")
                else:
                    new_experiment_names.append(name)
            
            #pylab.legend()
            pylab.legend(tuple(curves), tuple(new_experiment_names), loc="best")
            
            pylab.ylabel(target_measure, fontsize = fontsize)
            pylab.xlabel(target_param, fontsize = fontsize)

            pylab.show()



    def hyperparameter_plot_per_task(self, plot_test=False):
        """
        display a performance plot for every hyper-parameter including error bar

        auto-detect over which parameter we optimize
        """

        #target_param = str(self.settings.target_param_combo.itemData(self.settings.target_param_combo.currentIndex()).toString())
        #target_measure = str(self.settings.target_measure_combo.itemData(self.settings.target_measure_combo.currentIndex()).toString())
        #semilogx = str(self.settings.semi_logx_combo.itemData(self.settings.semi_logx_combo.currentIndex()).toString())
        
        target_param = str(self.settings.hyper.target_param_combo.currentText())
        target_measure = str(self.settings.hyper.target_measure_combo.currentText())
        semilogx = str(self.settings.hyper.semi_logx_combo.currentText())
        
        
        log_x = False
        if semilogx == "True":
            log_x = True
        

        pylab.figure()

        exp_ids = set([self.row_to_id[idx.row()] for idx in self.table.selectedIndexes()])

        exp_names = []
        curves = []

        plot_title = ""

        for id in exp_ids:

            try:
                e = expenv.Experiment.get(id)
                exp_names.append(str(e.id) + " " + e.description)

                plot_title = re.findall(".+\(", e.description)[0][0:-1]

                runs = [run for run in e.eval_runs]

                #num_tasks = 0

                # in order to avrg over splits, we need to find unique x-val values
                unique_params = list(set([getattr(run.method.param, target_param) for run in runs]))
                unique_params.sort()
     
     
                split_idx = list(set([int(run.x_val_idx) for run in runs]))
                split_idx.sort()
                
                y_values = []
                y_values_test = []

                std_values = []
                std_values_test = []

                run_map = defaultdict(list)

                # map runs to unique parameters
                for run in runs:
                    param_value = getattr(run.method.param, target_param)
                    run_map[param_value].append(run)

                # extract performances
                for unique_param in unique_params:
                    
                    assessment_param = []
                    assessment_test_param = []
                    
                    if use_max:
                        
                        print "taking max value for each point"
                        
                        for split_id in split_idx:
                            tmp_assessment = [getattr(run.assessment, target_measure) for run in run_map[unique_param] if hasattr(run.assessment, target_measure) and run.x_val_idx==split_id]
                            
                            if plot_test:
                                tmp_assessment_test = [getattr(run.assessment_test, target_measure) for run in run_map[unique_param] if hasattr(run.assessment_test, target_measure) and run.x_val_idx==split_id]
                            
                            
                            if len(tmp_assessment) > 0:            
                                max_id = numpy.argmax(tmp_assessment)
                                max_value = tmp_assessment[max_id] 
                                assessment_param.append(max_value)
                                
                                if plot_test:
                                    test_value = tmp_assessment_test[max_id]
                                    assessment_test_param.append(test_value)
                        
                    else:

                        print "taking average value for each point"
                        assessment_param = [getattr(run.assessment, target_measure) for run in run_map[unique_param] if hasattr(run.assessment, target_measure)]



                    if plot_test:
                        y_values_test.append(numpy.mean(assessment_test_param))
                        std_values_test.append(numpy.std(assessment_test_param))
                    else:
                        y_values.append(numpy.mean(assessment_param))
                        std_values.append(numpy.std(assessment_param))

                    
                    
                #curve = pylab.errorbar(unique_params, y_values, yerr=std_values)
                if log_x:
                    
                    if plot_test:
                        curve_test = pylab.semilogx(unique_params, y_values_test, "-o")
                    else:
                        curve = pylab.semilogx(unique_params, y_values, "-o")
                        
                else:
                    
                    if plot_test:
                        curve_test = pylab.plot(unique_params, y_values_test, "-o")
                    else:
                        curve = pylab.plot(unique_params, y_values, "-o")
                    
                
                
                if plot_test:
                    curves.append(curve_test)
                else:
                    curves.append(curve)
                

            except Exception, detail:
                import traceback
                traceback.print_exc(file=sys.stdout)
                print detail

            # render plot


            fontsize=14
            ax = pylab.gca()
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)

            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)
            #pylab.xlim(0, offset)
            #pylab.ylim(ymin, ymax)

            pylab.title(plot_title)
            pylab.gca().get_xaxis().tick_bottom()
            pylab.gca().get_yaxis().tick_left()

            pylab.gca().get_yaxis().grid(True)
            pylab.gca().get_xaxis().grid(False)

            new_experiment_names = []
            for name in exp_names:
                
                if plot_test:
                    new_experiment_names.append(name + " (test)")
                else:
                    new_experiment_names.append(name)
            
            #pylab.legend()
            pylab.legend(tuple(curves), tuple(new_experiment_names), loc="best")
            
            pylab.ylabel(target_measure, fontsize = fontsize)
            pylab.xlabel(target_param, fontsize = fontsize)

            pylab.show()


    def detailed_training_plot(self, target="auROC"):
        """
        display detailed results using a bar plot from matplotlib
        """


        rm_idx = set([self.row_to_id[idx.row()] for idx in self.table.selectedIndexes()])

        data = []
        data_std = []
        labels = []
        methods = []

        plot_title = ""

        for id in rm_idx:

            try:
                e = expenv.Experiment.get(id)
                methods.append(e.description)

                plot_title = re.findall(".+\(", e.description)[0][0:-1] + " (evaluation)"
                method = e.test_run.method
                candidate_runs = [run for run in e.eval_runs if run.method.id == method.id]

                print candidate_runs

                assessments = []
                num_tasks = 0

                print "number of candidate runs:", len(candidate_runs)

                # extract stuff from evaluation runs
                for run in candidate_runs:
                    sub_assessments = [getattr(a, target) for a in run.assessment.assessments]

                    print sub_assessments
                    
                   
                    num_tasks = len(sub_assessments)
                    labels = numpy.sort([a.task_id for a in run.assessment.assessments]).tolist()
                    idx = numpy.argsort([a.task_id for a in run.assessment.assessments])
                    sub_assessments = numpy.array(sub_assessments)[idx].tolist()
                    
                    if num_tasks!=0:
                        print "new_labels:", [a.task_id for a in run.assessment.assessments]
                        assert(len(sub_assessments)==num_tasks)
                        assert(labels == numpy.sort([a.task_id for a in run.assessment.assessments]).tolist())
                        
                   
                    assessments.append(sub_assessments)
                    
                data_row = []
                
                # summarize data
                for i in xrange(num_tasks):
                    print i, ":", [sub_assessments[i] for sub_assessments in assessments]
                    data_row.append(numpy.mean([sub_assessments[i] for sub_assessments in assessments]))

                data.append(data_row)

            except Exception, detail:
                import traceback
                traceback.print_exc(file=sys.stdout)
                print detail


        print data

        # transpose data
        data_T = []

        print "__data", len(data)
        print "__labels", len(labels)

        print "labels:", labels

        for i in xrange(len(labels)):

            data_row = []

            for j in xrange(len(data)):
                
                data_row.append(data[j][i])

            data_T.append(data_row)


        detailed_bar_plot(data_T, labels, methods, plot_title)



    def display_info(self, stuff):
        
        text = "" #"data: " + stuff.data(0).toString() + "\n"  
        
        
        experiment_id = self.row_to_id[stuff.row()]

        experiment = expenv.Experiment.get(experiment_id)

        text += str(experiment) + "\n"

        unfinished = [run.id for run in experiment.eval_runs if not run.assessment]

        text += "number of unfinished runs: " + str(len(unfinished)) + "\n"
        text += "run ids: " + str(unfinished)

        self.settings.textEdit.setText(text)


    def clean_up(self):
        
        rm_idx = set([self.row_to_id[idx.row()] for idx in self.table.selectedIndexes()])
        print "deleting experiments:", rm_idx

        for id in rm_idx:
            e = expenv.Experiment.get(id)
            e.clean_up()

        self.reload_view()


    def reload_data(self):

        # create table
        tb = get_table_data()
        self.tabledata = tb[0]
        self.row_to_id = tb[1]


    def reload_view(self):

        self.reload_data()

        header = ['experiment id', 'timestamp', 'cv best performance', 'test auROC', 'test auPRC', 'description']

        tm = MyTableModel(self.tabledata, header, self) 
        self.table.setModel(tm)
    

    def createTable(self):
        # create the view
        self.table = MyTableView()

        # set the table model
        self.reload_view()

        # set the minimum size
        # self.setMinimumSize(400, 300)

        # set row height
        nrows = len(self.tabledata)
        for row in xrange(nrows):
            self.table.setRowHeight(row, 18)

        # enable sorting
        # this doesn't work
        #self.table.setSortingEnabled(True)

        return self.table



        
class BaseTableView(QTableView):

    def __init__(self):
        QTableView.__init__(self)
        
        # only select rows
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

        
        # hide grid
        self.setShowGrid(True)
        
        # set the font
        font = QFont("Arial", 11)
        self.setFont(font)

        # hide vertical header
        vh = self.verticalHeader()
        vh.setVisible(False)

        # set horizontal header properties
        hh = self.horizontalHeader()
        hh.setStretchLastSection(True)

        # set column width to fit contents
        self.resizeColumnsToContents()
        
        

class MyTableView(BaseTableView):

    def __init__(self):
        BaseTableView.__init__(self)

    def keyPressEvent(self, e):

        key = e.key()

        if key == Qt.Key_Return:
            print "Awesome."

        elif key == Qt.Key_F5:
            print "reloading model"
            self.emit(SIGNAL("refresh()")) 

        elif key == Qt.Key_Delete:
            print "cleaning up seletion"
            self.emit(SIGNAL("cleanup()")) 

        elif key == Qt.Key_P:
            print "plotting seletion"
            self.emit(SIGNAL("plot()")) 

        elif key == Qt.Key_D:
            print "plotting seletion"
            self.emit(SIGNAL("detailed_plot()")) 

        elif key == Qt.Key_R:
            print "plotting roc curve"
            self.emit(SIGNAL("plot_roc_curve()")) 

        elif key == Qt.Key_G:
            print "displaying graph"
            self.emit(SIGNAL("show_graph()")) 

        elif key == Qt.Key_T:
            print "plotting training performance for seletion"
            self.emit(SIGNAL("detailed_training_plot()"))

        elif key == Qt.Key_F:
            print "plotting hyperparameter selection"
            self.emit(SIGNAL("hyperparameter_plot_test()"))
             
        elif key == Qt.Key_X:
            print "creating tex table"
            self.emit(SIGNAL("create_tex_table()")) 

        else:
            QTableView.keyPressEvent(self,e)





class DataTableView(BaseTableView):

    def __init__(self):
        BaseTableView.__init__(self)
        
        self.reload_view()
        
        
    def keyPressEvent(self, e):

        key = e.key()

        if key == Qt.Key_Return:
            print "Awesome."

        elif key == Qt.Key_F5:
            print "reloading model"
            self.reload_view()

        elif key == Qt.Key_Delete:
            print "cleaning up seletion"
            #self.emit(SIGNAL("cleanup()")) 


    def get_table_data(self):
        
        multi_split_sets = list(expenv.MultiSplitSet.select().orderBy(["id"]))
        
        tabledata= []
        row_to_id = []
        
        for mss in multi_split_sets:
            
            row = []
    
            row_to_id.append(mss.id)
    
            row.append(mss.id)
            row.append(str(mss.description))
            row.append(len(mss.split_sets))
            row.append(str(mss.feature_type))
            
            tabledata.append(row)
    
        return (tabledata, row_to_id)


    def reload_view(self):

        (tabledata, row_to_id) = self.get_table_data()
        

        header = ['id', 'description', 'num_mss', 'feature_type']

        tm = MyTableModel(tabledata, header, self) 
        self.setModel(tm)
        

class DataTreeView(QTreeWidget):

    def __init__(self):
        QTreeView.__init__(self)

        self.setColumnCount(2)
        
        
        self.reload_view()
        
        
    def keyPressEvent(self, e):

        key = e.key()

        if key == Qt.Key_Return:
            print "Awesome."

        elif key == Qt.Key_F5:
            print "reloading model"
            self.reload_view()

        elif key == Qt.Key_Delete:
            print "cleaning up seletion"
            #self.emit(SIGNAL("cleanup()")) 


    def reload_view(self):

        multi_split_sets = list(expenv.MultiSplitSet.select().orderBy(["id"]))
        
        tabledata= []
        row_to_id = []
        
        
        for mss in multi_split_sets:

            try:
                mss_widget = QTreeWidgetItem(self)
                mss_widget.setText(0, mss.description)

                mss_id_widget = QTreeWidgetItem(mss_widget)
                mss_id_widget.setText(0, "mss id")
                mss_id_widget.setText(1, str(mss.id))
                
                mss_description_widget = QTreeWidgetItem(mss_widget)
                mss_description_widget.setText(0, "description")
                mss_description_widget.setText(1, str(mss.description))
                            
                mss_splitsets_widget = QTreeWidgetItem(mss_widget)
                mss_splitsets_widget.setText(0, "split sets")
                mss_splitsets_widget.setText(1, str(len(mss.split_sets)))
                
                for ss in mss.split_sets:
                    
                    mss_splitset_widget = QTreeWidgetItem(mss_splitsets_widget)
                    mss_splitset_widget.setText(0, str(ss.dataset.organism))
                    
                
                    
                    mss_dataset_widget = QTreeWidgetItem(mss_splitset_widget)
                    mss_dataset_widget.setText(0, "split set id")
                    mss_dataset_widget.setText(1, str(ss.id))
                 
                 
                    mss_num_instances_widget = QTreeWidgetItem(mss_splitset_widget)
                    mss_num_instances_widget.setText(0, "num instances")
                    mss_num_instances_widget.setText(1, str(ss.num_instances))
                 
                 
                    mss_splits_widget = QTreeWidgetItem(mss_splitset_widget)
                    mss_splits_widget.setText(0, "splits")
                    mss_splits_widget.setText(1, str(len(ss.splits)))
                 
                       

                    for split in ss.splits:
                        
                        mss_split_widget = QTreeWidgetItem(mss_splits_widget)
                        mss_split_widget.setText(0, str(split.num))
                        
                        if split.is_test_set==True:
                            mss_split_widget.setText(1, "Test Set")

                        mss_split_id_widget = QTreeWidgetItem(mss_split_widget)
                        mss_split_id_widget.setText(0, "id")
                        mss_split_id_widget.setText(1, str(split.id))
                        
                        mss_num_inst_widget = QTreeWidgetItem(mss_split_widget)
                        mss_num_inst_widget.setText(0, "num instances")
                        mss_num_inst_widget.setText(1, str(split.num_instances))
                    
            except Exception, detail:
                print detail
    

class MyTableModel(QAbstractTableModel): 
    def __init__(self, datain, headerdata, parent=None, *args): 
        QAbstractTableModel.__init__(self, parent, *args) 
        self.arraydata = datain
        self.headerdata = headerdata
 
    def rowCount(self, parent): 
        return len(self.arraydata) 
 
    def columnCount(self, parent): 
        return len(self.arraydata[0]) 
 
    def data(self, index, role): 
        if not index.isValid(): 
            return QVariant() 
        elif role != Qt.DisplayRole: 
            return QVariant() 
        return QVariant(self.arraydata[index.row()][index.column()]) 

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headerdata[col])
        return QVariant()



class HyperparameterSettingsUI(object):

    def setupUi(self, Frame):
        Frame.setObjectName("Frame")
        Frame.resize(296, 183)
        Frame.setFrameShape(QtGui.QFrame.StyledPanel)
        Frame.setFrameShadow(QtGui.QFrame.Raised)
        self.gridLayoutWidget = QtGui.QWidget(Frame)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 20, 241, 141))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtGui.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.target_param_combo = QtGui.QComboBox(self.gridLayoutWidget)
        self.target_param_combo.setObjectName("target_param_combo")
        self.target_param_combo.addItem("")
        self.target_param_combo.addItem("")
        self.target_param_combo.addItem("")
        self.gridLayout.addWidget(self.target_param_combo, 0, 1, 1, 1)
        self.label_2 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.target_measure_combo = QtGui.QComboBox(self.gridLayoutWidget)
        self.target_measure_combo.setObjectName("target_measure_combo")
        self.target_measure_combo.addItem("")
        self.target_measure_combo.addItem("")
        self.gridLayout.addWidget(self.target_measure_combo, 1, 1, 1, 1)
        self.label_3 = QtGui.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.semi_logx_combo = QtGui.QComboBox(self.gridLayoutWidget)
        self.semi_logx_combo.setObjectName("semi_logx_combo")
        self.semi_logx_combo.addItem("")
        self.semi_logx_combo.addItem("")
        self.gridLayout.addWidget(self.semi_logx_combo, 2, 1, 1, 1)
        self.hyper_plot_button = QtGui.QPushButton(self.gridLayoutWidget)
        self.hyper_plot_button.setObjectName("hyper_plot_button")
        self.gridLayout.addWidget(self.hyper_plot_button, 3, 0, 1, 1)
        self.task_plot_button = QtGui.QPushButton(self.gridLayoutWidget)
        self.task_plot_button.setObjectName("task_plot_button")
        self.gridLayout.addWidget(self.task_plot_button, 3, 1, 1, 1)

        self.retranslateUi(Frame)
        QtCore.QMetaObject.connectSlotsByName(Frame)

    def retranslateUi(self, Frame):
        Frame.setWindowTitle(QtGui.QApplication.translate("Frame", "Frame", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Frame", "Target Parameter", None, QtGui.QApplication.UnicodeUTF8))
        self.target_param_combo.setItemText(0, QtGui.QApplication.translate("Frame", "cost", None, QtGui.QApplication.UnicodeUTF8))
        self.target_param_combo.setItemText(1, QtGui.QApplication.translate("Frame", "transform", None, QtGui.QApplication.UnicodeUTF8))
        self.target_param_combo.setItemText(2, QtGui.QApplication.translate("Frame", "base_similarity", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Frame", "Target Measure", None, QtGui.QApplication.UnicodeUTF8))
        self.target_measure_combo.setItemText(0, QtGui.QApplication.translate("Frame", "auROC", None, QtGui.QApplication.UnicodeUTF8))
        self.target_measure_combo.setItemText(1, QtGui.QApplication.translate("Frame", "auPRC", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Frame", "Semilog x", None, QtGui.QApplication.UnicodeUTF8))
        self.semi_logx_combo.setItemText(0, QtGui.QApplication.translate("Frame", "True", None, QtGui.QApplication.UnicodeUTF8))
        self.semi_logx_combo.setItemText(1, QtGui.QApplication.translate("Frame", "False", None, QtGui.QApplication.UnicodeUTF8))
        self.hyper_plot_button.setText(QtGui.QApplication.translate("Frame", "Model Plot", None, QtGui.QApplication.UnicodeUTF8))
        self.task_plot_button.setText(QtGui.QApplication.translate("Frame", "Task Plot", None, QtGui.QApplication.UnicodeUTF8))

        
        
class HyperparameterSettingsFrame(QtGui.QFrame, HyperparameterSettingsUI):
    
    def __init__(self): 
        QtGui.QWidget.__init__(self) 
        self.setupUi(self) 

        self.setMinimumWidth(360)



class BottomUi(object):
    
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(719, 299)
    

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.checkBox.setText(QtGui.QApplication.translate("Form", "CheckBox", None, QtGui.QApplication.UnicodeUTF8))


class BottomWidget(QFrame): 

    def __init__(self, *args): 
        QFrame.__init__(self, *args) 

        self.setMinimumHeight(180)
        self.setMaximumHeight(300)

        horizontalLayout = QtGui.QHBoxLayout()
        horizontalLayout.setObjectName("horizontalLayout")
        
        self.hyper = HyperparameterSettingsFrame()
        self.hyper.setObjectName("hyper")
        horizontalLayout.addWidget(self.hyper)
        self.textEdit = QtGui.QTextEdit(self)
        self.textEdit.setObjectName("textEdit")
        horizontalLayout.addWidget(self.textEdit)
        
        self.setLayout(horizontalLayout) 


if __name__ == "__main__": 
    main()

