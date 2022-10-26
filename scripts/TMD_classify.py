# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:50:07 2022

@author: yanyanming77
"""

"""This script is to (train classifiers) use pre-trained classifier to classify junk SWCs and true neuron SWCs"""
"""Classifiers explored: LDA, Decision Tree, Random Forest"""

import importlib
import numpy as np
import tmd
from pathlib import Path
import os
import shutil
from datetime import datetime
from tmd.Population import Population
import collections
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, roc_curve, roc_auc_score
import pickle
from argparse import ArgumentParser
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def generate_data(junk_path, true_path, neurite_type = "basal_dendrite"):
    """
    This function aims to generate the training data (pvecs), train_x, train_y, and labels for model training
    input:
        - junk_path: path to the junk SWCs
        - true_path: path to the true neuron SWCs
        - neurite_type: type of neurite, default is "basal_dendrite for MSNs"
    returns:
        - train_dataset: train_x
        - labels: train_y
        - xlims: x limits that will be used when generating pvecs during prediction
        - ylims: y limits that will be used when generating pvecs during prediction
    """

    # ------------------------ Training dataset --------------------------------
    # load SWCs in junk folder
    pop1 = Population.Population()
    pop1_success = []
    pop1_failed = []
    for f in os.listdir(junk_path):
        # swc = [str(directory1/f)]
        swc = str(junk_path/f)
        try:
            n = tmd.io.load_neuron(swc)
            pop1.append_neuron(n)
            pop1_success.append(f)
        except:
            pop1_failed.append(f)
    print(f"For SWC set1: {len(pop1_success)} files loaded successfully, {len(pop1_failed)} files loaded failed")

    # load SWCs in true neuron folder
    pop2 = Population.Population()
    pop2_success = []
    pop2_failed = []
    for f in os.listdir(true_path):
        swc = str(true_path/f)
        try:
            n = tmd.io.load_neuron(swc)
            pop2.append_neuron(n)
            pop2_success.append(f)
        except:
            pop2_failed.append(f)
    print(f"For SWC set2: {len(pop2_success)} files loaded successfully, {len(pop2_failed)} files loaded failed")

    groups = [pop1, pop2]

    # neurite_type='basal_dendrite'

    # Generate a persistence diagram per neuron
    # need to consider the SWCs that need to be excluded due to failer in getting persistence diagram
    pers_diagrams = []
    indx_dict = {0:[], 1:[]}
    indx_dict_0 = {0:[], 1:[]}
    for i,m in enumerate(groups):
        for j,n in enumerate(m.neurons):
            try:
                p = tmd.methods.get_ph_neuron(n, neurite_type = neurite_type)
                if len(p) > 0:
                    pers_diagrams.append(p)
                    indx_dict[i].append(j)
                elif len(p) == 0:
                    indx_dict_0[i].append(j)
            except:
                pass


    # refine the group object by only keeping those have pvecs successfully retrieved and non-zero pvecs
    group1 = [m for i,m in enumerate(groups[0].neurons) if i in indx_dict[0]]
    group2 = [n for j,n in enumerate(groups[1].neurons) if j in indx_dict[1]]
    groups_new = [group1, group2]


    # Define labels in neuron level, 1 is junk, 2 is true in this case
    labels = [i + 1 for i, k in enumerate(groups_new) for j in k]
    # validate this is correct
    # fq_check = collections.Counter(labels)
    # print(dict(fq_check))

    # Define x-ylimits
    xlims, ylims = tmd.analysis.get_limits(pers_diagrams)
    # Generate a persistence image for each diagram
    # also need to error control
    pers_images_indx = []
    pers_images = []
    for i,p in enumerate(pers_diagrams):
        try:
            p_i = tmd.analysis.get_persistence_image_data(p, xlims=xlims, ylims=ylims)
            # need to handle those NANs
            if np.isnan(p_i).any():
                pass
                print("NANs catched, the index is: ",i)
            else:
                pers_images_indx.append(i)
                pers_images.append(p_i)
        except:
            pass

    # pers_images[0]
    labels = [l for i,l in enumerate(labels) if i in pers_images_indx]

    success1 = dict(collections.Counter(labels)).get(1)
    success2 = dict(collections.Counter(labels)).get(2)

    # print out the summary
    print("Summary after completion of getting persistence image data:")
    print(f"SWC set1 has {success1} success and non-zeros, {len(os.listdir(junk_path))-success1} discarded")
    print(f"SWC set2 has {success2} success and non-zeros, {len(os.listdir(true_path))-success2} discarded")

    # pers_images = [
    #     tmd.analysis.get_persistence_image_data(p, xlims=xlims, ylims=ylims) for p in pers_diagrams
    # ]

    # Create the train dataset from the flatten images
    train_dataset = [i.flatten() for i in pers_images]

    return train_dataset, labels, xlims, ylims


def model_eval(clf, x_test, y_test, clf_name):

    """
    Generates all model evaluation metrics of the given classifier
    and save confusion matrix, roc curve to pdf files
    input:
        - clf: classifier
        - x_test:
        - y_test:
        - clf_name: name of the clf (str)
    output:
        - confusion matrix visualization
        - classification report
        - ROC, AUC
    """
    # accuracy
    # print(f"Accuracy: {clf.score(x_test, y_test)}")

    # confusion matrix
    plot_confusion_matrix(clf, x_test, y_test).figure_
    plt.xticks(ticks=([0,1]), labels=(['junk','true']))
    plt.yticks(ticks=([0,1]), labels=(['junk','true']))
    plt.title("Confusion matrix of " + clf_name)
    plt.savefig(result_path/f"Confusion_matrix_{clf_name}.pdf", format="pdf")
    plt.show()

    # ROC AUC
    y_pred_proba = clf.predict_proba(x_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba, pos_label=2)
    auc = roc_auc_score(y_test, y_pred_proba)
    #create ROC curve
    plt.figure()
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.title("ROC curve for " + clf_name)

    plt.savefig(result_path/f"ROC_curve_{clf_name}.pdf", format="pdf")
    plt.show()

    # classification report
    print(classification_report(y_test, clf.predict(x_test), target_names=['junk','true'], digits=2))

def make_prediction(input, clf, clf_name, xlims=[1,1320], ylims=[0,905], num_iter = 20, neurite_type = "basal_dendrite"):

    """
    this function aims to predict the class of the SWC(s), input could be a single file or a directory of SWCs
    input:
        - input: a SWC file or a directory
        - clf: the classifier to use
        - clf_name: name of the clf (str)
        - num_iter: times of iteration, only use this when input is a single SWC file
    returns:
        - true_neuron_count: number of neurons classified as true positives by the model
    """

    true_neuron_count = 0
    # input = Path("C:\\Users\\yanyanming77\\Desktop\\precision_recall\\TMD\\swc2")
    # clf = pickle.load(open(Path("C:\\Users\\yanyanming77\\Desktop\\precision_recall\\TMD\\model_base_rf_12_52_10.sav"), 'rb'))

    # if input is a single neuron
    if os.path.isfile(input):
        print("input is a file")
        try:
            neuron2test = tmd.io.load_neuron(str(input))
        except:
            print("Error when loading this neuron, please try another one")
        # get pvecs
        pers2test = tmd.methods.get_ph_neuron(neuron2test, neurite_type = neurite_type)
        # get persistence image
        pers_image2test = tmd.analysis.get_persistence_image_data(pers2test, xlims = xlims, ylims = ylims)
        # flatten
        test_dataset = pers_image2test.flatten()

        predict_labels = []

        # Train classifier with training images for selected number_of_trials
        for i in range(num_iter):
            predict_labels.append(clf.predict([test_dataset])[0])

        predict_cnt = dict(collections.Counter(predict_labels))
        predict_cnt_max = max(predict_cnt, key= lambda x: predict_cnt[x])

        if predict_cnt_max == 1:
            print(f"The predicted class is JUNK")
            true_neuron_count = 0
        elif predict_cnt_max == 2:
            print(f"The predicted class is TRUE NEURON")
            true_neuron_count = 1

    # if input is a file path
    elif os.path.isdir(input):
        pop = Population.Population()
        pop_success = []
        pop_failed = []
        for f in os.listdir(input):
            # swc = [str(directory1/f)]
            swc = str(input/f)
            try:
                n = tmd.io.load_neuron(swc)
                pop.append_neuron(n)
            except:
                pass
        # print(f"For classification set: {len(pop.neurons)} loaded successfully")

        predict_junk_list = []
        predict_true_neuron_list = []
        failed_list = []
        # for each neuron in the population, generate pvecs and classify using the designated classifier
        for i,n in enumerate(tqdm(pop.neurons)):
            neuron_name = os.path.split(Path(n.name))[-1] + '.swc'
            try:
                pers2test = tmd.methods.get_ph_neuron(n, neurite_type = 'basal_dendrite')
                pers_image2test = tmd.analysis.get_persistence_image_data(pers2test, xlims = xlims, ylims = ylims)
                test_dataset = pers_image2test.flatten()
                predict_labels = []

                # Train classifier with training images for selected number_of_trials
                for i in range(num_iter):
                    predict_labels.append(clf.predict([test_dataset])[0])

                predict_cnt = dict(collections.Counter(predict_labels))
                predict_cnt_max = max(predict_cnt, key= lambda x: predict_cnt[x])

                if predict_cnt_max == 1:
                    predict_junk_list.append(neuron_name)
                elif predict_cnt_max == 2:
                    predict_true_neuron_list.append(neuron_name)
            except:
                pass

        ### save the classified SWCs to separate folders
        path_result = result_path/f"{clf_name}_{current_time}"
        path_result.mkdir(exist_ok=False)
        path_junk = path_result/'predicted_junk'
        path_junk.mkdir(exist_ok=True)
        path_true = path_result/'predicted_true'
        path_true.mkdir(exist_ok=True)
        path_failed = path_result/'failed'
        path_failed.mkdir(exist_ok=True)

        # list of failed files
        for f in os.listdir(input):
            # junk
            if f in predict_junk_list:
                shutil.copy(input/f, path_junk/f)
            # true
            elif f in predict_true_neuron_list:
                shutil.copy(input/f, path_true/f)
            elif f not in predict_junk_list and f not in predict_true_neuron_list:
                shutil.copy(input/f, path_failed/f)
                failed_list.append(f)

        true_neuron_count = len(predict_true_neuron_list)
        print("Summary of Classification:")
        print(f"# of junk: {len(predict_junk_list)}, # of true neuron: {true_neuron_count}, # failed: {len(failed_list)}")

    return true_neuron_count

def filter_dir_by_model(input, model, xlimits=None, ylimits=None):
    """
    wrapper function to make_prediction() to make classification simpler from outside scripts
    input:
        - input: a SWC file or a directory
        - model: classifier path to use
    returns:
        - true_neuron_count: number of neurons classified as true positives by the model
    """

    m =  pickle.load(open(model, 'rb'))
    # print(f"the model is loaded as {m}")
    m_name = os.path.split(model)[1].split('.')[0]
    # print(f"model name is: {m_name}")
    return make_prediction(input, m, m_name, xlimits, ylimits)

def main():
    time_now = datetime.now()
    current_time = time_now.strftime("%H:%M:%S").replace(':', '_')

    parser = ArgumentParser(
        description = "TMD filtering, model training and classification")

    # if needs to train model first
    parser.add_argument("--junk", "-j", type=str, required=False, help="path to junk folder")
    parser.add_argument("--true", "-t", type=str, required=False, help="path to true neuron folder")
    parser.add_argument("--model", "-m", type=str, required=False, help="path and name of the saved model")
    parser.add_argument("--filter", "-f", type=str, required=True, help="path or file to classifiy")
    parser.add_argument("--xlims", "-xlims", nargs='+', type=int, required=False, help="lower limit, upper limit of X separated by space, based on the training set")
    parser.add_argument("--ylims", "-ylims", nargs='+', type=int, required=False, help="lower limit, upper limit of Y separated by space, based on the training set")
    args = parser.parse_args()

    junk_path = None
    true_path = None
    model = None
    xlimits = None
    ylimits = None

    if args.junk:
        junk_path = Path(args.junk)
    if args.true:
        true_path = Path(args.true)
    if args.model:
        model = Path(args.model)
    input = Path(args.filter)
    if args.xlims:
        xlimits = args.xlims
    if args.ylims:
        ylimits = args.ylims
    print(f"xlimits {xlimits}")
    print(f"ylimits {ylimits}")

    # results will be save in the parent directory of the file(s) to be filtered
    result_path = input.parent

    ### evaluate inputs
    # if need to train model first
    if junk_path and true_path and input:
        ### TRAIN MODEL, TRY LDA, DT, RF
        TRAIN_X, TRAIN_Y, xlims, ylims = generate_data(junk_path, true_path, neurite_type = "basal_dendrite")
        print("Take note of x and y limits:")
        print(f"X limits: {xlims}, Y limits: {ylims}")

        # split data to train and val
        x_train, x_val, y_train, y_val = train_test_split(TRAIN_X, TRAIN_Y, train_size=0.8, random_state=42)
        print(f"train x: {len(x_train)}, train y: {len(y_train)}")
        print(f"val x: {len(x_val)}, val y: {len(y_val)}")

        # # 1) LDA baseline
        # lda_clf_base = LinearDiscriminantAnalysis()
        # lda_clf_base.fit(x_train, y_train)
        # lda_clf_base.score(x_val, y_val) # accuracy: 0.97

        # # 2) Decision Tree
        # dt_clf_base = DecisionTreeClassifier(random_state=42)
        # dt_clf_base.fit(x_train, y_train)
        # dt_clf_base.score(x_val, y_val) # accuracy: 0.998

        # 3) Random Forest
        rf_clf_base = RandomForestClassifier(random_state=42)
        rf_clf_base.fit(x_train, y_train)
        print(f"Baseline RF accuracy: {rf_clf_base.score(x_val, y_val)}") # 0.98

        # evaluate baseline model
        # model_eval(lda_clf_base, x_val, y_val, "baseline LDA")
        # model_eval(dt_clf_base, x_val, y_val, "baseline Decision Tree")
        model_eval(rf_clf_base, x_val, y_val, "baseline Random Forest")

        # save model
        model_name = result_path/f"model_base_rf_{current_time}.sav"
        pickle.dump(rf_clf_base, open(model_name, 'wb'))

        # make prediction
        make_prediction(input, rf_clf_base, "rf_clf_base", xlims=xlims, ylims=ylims)

    # if have pre-trained model
    elif model and input:
        filter_dir_by_model(input, model, xlimits, ylimits)

if __name__ == '__main__':
    main()

# # for testing purpose
# junk_path = Path("C:\\Users\\yanyanming77\\Desktop\\precision_recall\\TMD\\junk_swc")
# true_path = Path("C:\\Users\\yanyanming77\\Desktop\\precision_recall\\TMD\\proofread_swc")

# # result path is the path where script is located
# script_path = Path( __file__ ).absolute()

# base_dir = Path("C:\\Users\\yanyanming77\\Desktop\\precision_recall\\TMD\\swc1")
# input = base_dir/"001_TME10-1_30x_Str_02A_x3891_y12982_z405.swc"

