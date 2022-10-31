# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:50:07 2022

@author: yanyanming77

This script is to (train classifiers) use pre-trained classifier to classify junk SWCs and true neuron SWCs
Classifiers explored: LDA, Decision Tree, Random Forest
"""
import numpy as np
import tmd
import pickle
import os
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

# import importlib
# from sklearn.metrics import confusion_matrix
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
warnings.filterwarnings("ignore")


def generate_data(junk_path, true_path, neurite_type="basal_dendrite"):
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
        swc = str(junk_path / f)
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
        swc = str(true_path / f)
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
    indx_dict = {0: [], 1: []}
    indx_dict_0 = {0: [], 1: []}
    for i, m in enumerate(groups):
        for j, n in enumerate(m.neurons):
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
    for i, p in enumerate(pers_diagrams):
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

    # pers_images = [
    #     tmd.analysis.get_persistence_image_data(p, xlims=xlims, ylims=ylims) for p in pers_diagrams
    # ]

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
    plt.savefig(result_path / f"Confusion_matrix_{clf_name}.pdf", format="pdf")
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

    plt.savefig(result_path / f"ROC_curve_{clf_name}.pdf", format="pdf")
    plt.show()

    # classification report
    print(classification_report(y_test, clf.predict(x_test), target_names=['junk', 'true'], digits=2))


def copy_file(file, destination):
    if "multi-component" in file.parent.name:
        multi_component_dest_dir = destination / file.parent.name
        multi_component_dest_dir.mkdir(exist_ok=True)
        for n_file in file.parent.glob(f"{file.name[:-len(file.suffix)]}.*"):
            shutil.copy(n_file, multi_component_dest_dir)
    else:
        shutil.copytree(src=file.parent, dst=destination/file.parent.name, dirs_exist_ok=True)


def make_prediction(input_path: Path, clf, clf_name: str, result_path: Path, current_time: str,
                    xlims=(1, 1320), ylims=(0, 905), num_iter=20, neurite_type="basal_dendrite"):
    """
    this function aims to predict the class of the SWC(s), input could be a single file or a directory of SWCs

    - input_path: a SWC file or a directory
    - clf: the classifier to use
    - clf_name: name of the clf (str)
    - num_iter: times of iteration, only use this when input is a single SWC file

    returns:
    - true_neuron_count: number of neurons classified as true positives by the model
    """

    true_neuron_count = junk_neuron_count = failed_neuron_count = 0

    path_result = result_path / f"{input_path.name}_{clf_name}_{current_time}"
    path_result.mkdir(exist_ok=False)
    path_junk = path_result / 'predicted_junk'
    path_junk.mkdir(exist_ok=True)
    path_true = path_result / 'predicted_true'
    path_true.mkdir(exist_ok=True)
    path_failed = path_result / 'failed'
    path_failed.mkdir(exist_ok=True)
    log_file = path_result/"prediction.log"
    log.basicConfig(filename=str(log_file), level=log.INFO)
    log.FileHandler(str(log_file), mode="w")  # rewrite the file instead of appending

    for file in tqdm(list(input_path.rglob("*.swc")), desc="classification: "):
        try:
            n = tmd.io.load_neuron(str(file))
        except Exception as e:
            log.error(f"tmd.io.load_neuron failed to load {file}: {e}")
            failed_neuron_count += 1
            copy_file(file, path_failed)
            continue
        try:
            pers2test = tmd.methods.get_ph_neuron(n, neurite_type='basal_dendrite')
            pers_image2test = tmd.analysis.get_persistence_image_data(pers2test, xlims=xlims, ylims=ylims)
            test_dataset = pers_image2test.flatten()
            predict_labels = []

            # Train classifier with training images for selected number_of_trials
            for idx in range(num_iter):
                predict_labels.append(clf.predict([test_dataset])[0])

            predict_cnt = dict(collections.Counter(predict_labels))
            predict_cnt_max = max(predict_cnt, key=lambda x: predict_cnt[x])

            if predict_cnt_max == 1:
                junk_neuron_count += 1
                copy_file(file, path_junk)
            elif predict_cnt_max == 2:
                true_neuron_count += 1
                copy_file(file, path_true)
            else:
                log.info(f"unexpected class {predict_cnt_max} for {file}")
                failed_neuron_count += 1
                copy_file(file, path_failed)
        except Exception as e:
            log.error(f"failure in classification for {file}: {e}")
            failed_neuron_count += 1
            copy_file(file, path_failed)

    print("Summary of Classification:\n"
          f"\t{junk_neuron_count} \t # junk neurons\n"
          f"\t{true_neuron_count} \t # true neurons\n"
          f"\t{failed_neuron_count}\t # failed neurons"
          f"\t{true_neuron_count/(true_neuron_count+failed_neuron_count+junk_neuron_count):.2f} yield %")

    return true_neuron_count, junk_neuron_count


def filter_dir_by_model(input_path: Path, model: Path, result_path: Path, current_time: str,
                        xlimits=None, ylimits=None):
    """
    wrapper function to make_prediction() to make classification simpler from outside scripts
    input:
        - input: a SWC file or a directory
        - model: classifier path to use
    returns:
        - true_neuron_count: number of neurons classified as true positives by the model
    """

    m = pickle.load(open(model, 'rb'))
    # print(f"the model is loaded as {m}")
    m_name: str = os.path.split(model)[1].split('.')[0]
    # print(f"model name is: {m_name}")
    return make_prediction(input_path, m, m_name, result_path, current_time, xlimits, ylimits)


def main():
    time_now = datetime.now()
    current_time = time_now.strftime("%H:%M:%S").replace(':', '_')

    parser = ArgumentParser(description="TMD filtering, model training and classification")

    # if you need to train model first
    parser.add_argument("--junk", "-j", type=str, required=False, help="path to junk folder")
    parser.add_argument("--true", "-t", type=str, required=False, help="path to true neuron folder")
    parser.add_argument("--model", "-m", type=str, required=False, help="path and name of the saved model")
    parser.add_argument("--filter", "-f", type=str, required=True, help="path or file to classify")
    parser.add_argument("--xlims", "-xlims", nargs='+', type=int, required=False,
                        help="lower limit, upper limit of X separated by space, based on the training set")
    parser.add_argument("--ylims", "-ylims", nargs='+', type=int, required=False,
                        help="lower limit, upper limit of Y separated by space, based on the training set")
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
    input_path = Path(args.filter)
    if args.xlims:
        xlimits = args.xlims
    if args.ylims:
        ylimits = args.ylims
    print(f"xlimits {xlimits}")
    print(f"ylimits {ylimits}")

    # results will be saved in the parent directory of the file(s) to be filtered
    result_path = input_path.parent

    # evaluate inputs
    # if you need to train model first
    if junk_path and true_path and input_path:
        # TRAIN MODEL, TRY LDA, DT, RF
        TRAIN_X, TRAIN_Y, xlims, ylims = generate_data(junk_path, true_path, neurite_type="basal_dendrite")
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
        print(f"Baseline RF accuracy: {rf_clf_base.score(x_val, y_val)}")  # 0.98

        # evaluate baseline model
        # model_eval(lda_clf_base, x_val, y_val, "baseline LDA")
        # model_eval(dt_clf_base, x_val, y_val, "baseline Decision Tree")
        model_eval(rf_clf_base, x_val, y_val, "baseline Random Forest", result_path)

        # save model
        model_name = result_path / f"model_base_rf_{current_time}.sav"
        pickle.dump(rf_clf_base, open(model_name, 'wb'))

        # make prediction
        make_prediction(input_path, rf_clf_base, "rf_clf_base", result_path, current_time, xlims=xlims, ylims=ylims)

    # if you have pre-trained model
    elif model and input_path:
        filter_dir_by_model(input_path, model, result_path, current_time, xlimits, ylimits)


if __name__ == '__main__':
    main()
