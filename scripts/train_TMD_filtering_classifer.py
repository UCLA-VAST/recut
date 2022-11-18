# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:04:05 2022

@author: yanyanming77

To train a new random forest classifer using new input data
Save the model and get the x, y limits that are used in calculating pvecs
"""

import numpy as np
import tmd
import pickle
import os
from os.path import dirname, abspath
import shutil
import collections
import matplotlib.pyplot as plt
import warnings
import logging as log
from datetime import datetime
from tmd.Population import Population
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report, roc_curve, roc_auc_score
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

warnings.filterwarnings("ignore")


def generate_data(junk_path, true_path, neurite_type="all"):
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
    for f in tqdm(os.listdir(junk_path), desc="Reading files in junk path: "):
        # swc = [str(directory1/f)]
        swc = str(junk_path / f)
        try:
            n = tmd.io.load_neuron(swc)
            pop1.append_neuron(n)
            pop1_success.append(f)
        except:
            pop1_failed.append(f)
    print(
        f"For files in junk path: {len(pop1_success)} files loaded successfully, {len(pop1_failed)} files loaded failed")

    # load SWCs in true neuron folder
    pop2 = Population.Population()
    pop2_success = []
    pop2_failed = []
    for f in tqdm(os.listdir(true_path), desc="Reading files in true path: "):
        swc = str(true_path / f)
        try:
            n = tmd.io.load_neuron(swc)
            pop2.append_neuron(n)
            pop2_success.append(f)
        except:
            pop2_failed.append(f)
    print(
        f"For files in true path: {len(pop2_success)} files loaded successfully, {len(pop2_failed)} files loaded failed")

    groups = [pop1, pop2]

    # neurite_type='basal_dendrite'

    # Generate a persistence diagram per neuron
    # need to consider the SWCs that need to be excluded due to failer in getting persistence diagram
    pers_diagrams = []
    indx_dict = {0: [], 1: []}
    indx_dict_0 = {0: [], 1: []}
    for i, m in enumerate(groups):
        for j, n in enumerate(tqdm(m.neurons, desc=f"Getting P-diagrams for group {i + 1}")):
            try:
                p = tmd.methods.get_ph_neuron(n, neurite_type=neurite_type)
                if len(p) > 0:
                    pers_diagrams.append(p)
                    indx_dict[i].append(j)
                elif len(p) == 0:
                    indx_dict_0[i].append(j)
            except:
                pass

    # refine the group object by only keeping those have pvecs successfully retrieved and non-zero pvecs
    group1 = [m for i, m in enumerate(groups[0].neurons) if i in indx_dict[0]]
    group2 = [n for j, n in enumerate(groups[1].neurons) if j in indx_dict[1]]
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
    for i, p in enumerate(tqdm(pers_diagrams, desc="Getting Pvecs: ")):
        try:
            p_i = tmd.analysis.get_persistence_image_data(p, xlims=xlims, ylims=ylims)
            # need to handle those NANs
            if np.isnan(p_i).any():
                pass
                print("NANs catched, the index is: ", i)
            else:
                pers_images_indx.append(i)
                pers_images.append(p_i)
        except:
            pass

    # pers_images[0]
    labels = [l for i, l in enumerate(labels) if i in pers_images_indx]

    success1 = dict(collections.Counter(labels)).get(1)
    success2 = dict(collections.Counter(labels)).get(2)

    # print out the summary
    print("Summary after completion of getting persistence image data:")
    print(f"SWC set1 has {success1} success and non-zeros, {len(os.listdir(junk_path)) - success1} discarded")
    print(f"SWC set2 has {success2} success and non-zeros, {len(os.listdir(true_path)) - success2} discarded")
    

    # Create the train dataset from the flatten images
    train_dataset = [i.flatten() for i in pers_images]

    return train_dataset, labels, xlims, ylims

def model_eval(clf, x_test, y_test, clf_name, result_path):
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
    plt.xticks(ticks=([0, 1]), labels=(['junk', 'true']))
    plt.yticks(ticks=([0, 1]), labels=(['junk', 'true']))
    plt.title("Confusion matrix of " + clf_name)
    plt.savefig(os.path.join(result_path, f"Confusion_matrix_{clf_name}.pdf"), format="pdf")
    plt.show()

    # ROC AUC
    y_pred_proba = clf.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=2)
    auc = roc_auc_score(y_test, y_pred_proba)
    # create ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.title("ROC curve for " + clf_name)

    plt.savefig(os.path.join(result_path, f"ROC_curve_{clf_name}.pdf"), format="pdf")
    plt.show()

    # classification report
    print(classification_report(y_test, clf.predict(x_test), target_names=['junk', 'true'], digits=2))
    
    
def train_classifier():
    """
    This function is to train a random forest classifer with weighted classes
    """

    # get generated training data and the xylimits
    train_x, train_y, xlims, ylims = generate_data(junk_path, true_path, neurite_type='all')
    # split data to train and val
    x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, train_size=0.8, random_state=42)
    print(f"size of train x: {len(x_train)}, size of train y: {len(y_train)}")
    print(f"size of validation x: {len(x_val)}, size of validation y: {len(y_val)}")

    # Random Forest
    rf_clf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_clf_base.fit(x_train, y_train)
    print(f"Baseline RF accuracy: {rf_clf_base.score(x_val, y_val)}")  # 0.98

    # evaluate baseline model
    result_path = os.path.join(dirname(abspath(__file__)), f"classifier_{current_time}")
    os.mkdir(result_path)
    model_eval(rf_clf_base, x_val, y_val, "Baseline.Random.Forest", result_path)

    # save model
    model_name = os.path.join(result_path, f"model_base_rf_{current_time}.sav")
    pickle.dump(rf_clf_base, open(model_name, 'wb'))
    
    print(f"x lims: {[round(i,2) for i in xlims]}")
    print(f"y lims: {[round(i,2) for i in ylims]}")
    
    # write the x and ylimits to a text file
    with open(os.path.join(result_path, 'xy_limits.txt'),'w') as limits_file:
        limits_file.write(f'The x and y limits of the training data, should use these limits if you want to make predictions of new SWCs using this classifier\n\n')
        limits_file.write(f'x limits: {[round(i, 2) for i in xlims]}\n')
        limits_file.write(f'y limits: {[round(i, 2) for i in ylims]}\n')
        
    limits_file.close()
    
    num_junk_train = dict(collections.Counter(y_train)).get(1)
    num_true_train = dict(collections.Counter(y_train)).get(2)
    num_junk_val = dict(collections.Counter(y_val)).get(1)
    num_true_val = dict(collections.Counter(y_val)).get(2)
    
    # write log of the training process
    with open(os.path.join(result_path, 'log.txt'),'w') as log_file:
        log_file.write("Log\n")
        log_file.write(f"size of train x: {len(x_train)}, size of train y: {len(y_train)}\n")
        log_file.write(f"size of validation x: {len(x_val)}, size of validation y: {len(y_val)}\n\n")
        log_file.write(f"In training data, {num_junk_train} junk, {num_true_train} true neuron used\n")
        log_file.write(f"In validation data, {num_junk_val} junk, {num_true_val} true neuron used") 
    log_file.close()

if __name__ == '__main__':
    
    current_time = datetime.now().strftime("%H_%M_%S")
    
    parser = ArgumentParser(description="Train a RF classifier using labeled junk and true neuron SWCs for TMD filtering")
    parser.add_argument("--junk", "-j", type=str, required=True, help="path to junk folder")
    parser.add_argument("--true", "-t", type=str, required=True, help="path to true neuron folder")
    args = parser.parse_args()
    
    junk_path = Path(args.junk)
    true_path = Path(args.true)

    train_classifier()
    
    
    

