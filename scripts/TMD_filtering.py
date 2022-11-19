# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:02:37 2022

@author: yanyanming77

To perform SWC filtering of junks/true neurons using pre-defined conditions and pre-trained random forest classifier

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

warnings.filterwarnings("ignore")

def copy_file(file, destination):
    
    """
    Copy files to their corresponding folder based on the filtering result
    """
    
    if "multi-component" in file.parent.name:
        multi_component_dest_dir = destination / file.parent.name
        multi_component_dest_dir.mkdir(exist_ok=True)
        for n_file in file.parent.glob(f"{file.name[:-len(file.suffix)]}.*"):
            shutil.copy(n_file, multi_component_dest_dir)
    else:
        shutil.copy(file, dst=destination / file.name)


def make_prediction(input_path: Path, clf, clf_name: str, result_path: Path, current_time: str,
                    xlims, ylims, num_iter=20, neurite_type="all"):
    """
    this function aims to predict the class of the SWC(s), input could be a single file or a directory of SWCs

    - input_path: a SWC file or a directory to be classified
    - clf: the classifier to use
    - clf_name: name of the clf (str)
    - num_iter: times of iteration, only use this when input is a single SWC file

    returns:
    - true_neuron_count: number of neurons classified as true positives by the model
    """

    true_neuron_count = junk_neuron_count = failed_neuron_count = 0
    junk_neuron_short_length_count = junk_neuron_singular_matrix_count = junk_neuron_classified_count = 0
    
    path_result = result_path / f"{input_path.name}_{clf_name}_{current_time}"
    path_result.mkdir(exist_ok=False)
    path_junk = path_result / 'predicted_junk'
    path_junk.mkdir(exist_ok=True)
    path_true = path_result / 'predicted_true'
    path_true.mkdir(exist_ok=True)
    path_failed = path_result / 'failed'
    path_failed.mkdir(exist_ok=True)
    log_file = path_result / "prediction.log"
    log.basicConfig(filename=str(log_file), level=log.INFO)
    log.FileHandler(log_file.__str__(), mode="w")  # rewrite the file instead of appending
    
    ### validate xy limits
    print(f"Current x limits: {xlims}, Current y limits: {ylims}")


    for file in tqdm(list(input_path.rglob("*.swc")), desc="classification: "):
        try:
            ### try morphio as well
            n = tmd.io.load_neuron(file.__str__())
            # n = tmd.io.load_neuron_from_morphio(file.__str__())
        except Exception as e:
            log.error(f"tmd.io.load_neuron failed to load {file}: {e}")
            failed_neuron_count += 1
            copy_file(file, path_failed)
            continue
        try:
            pers2test = tmd.methods.get_ph_neuron(n, neurite_type=neurite_type)
            # Check for junk definition1: length of P-diagram is 0 or 1
            if len(pers2test) in (0, 1):
                junk_neuron_count += 1
                junk_neuron_short_length_count += 1
                copy_file(file, path_junk)
            elif len(pers2test) > 1:
                pers_image2test = np.empty(1, dtype='float64')  # initialize
                # Check fro junk definition2: singular matrix error when getting P-vecs
                try:
                    pers_image2test = tmd.analysis.get_persistence_image_data(pers2test, xlims=xlims, ylims=ylims)
                except np.linalg.LinAlgError:
                    junk_neuron_count += 1
                    junk_neuron_singular_matrix_count += 1
                    copy_file(file, path_junk)
                except Exception as e:
                    log.info(f"other error when extracting pvecs for {file}: {e}")
                    failed_neuron_count += 1
                    copy_file(file, path_failed)

                if len(pers_image2test) > 1:

                    test_dataset = pers_image2test.flatten()
                    predict_labels = []

                    # Train classifier with training images for selected number_of_trials
                    for idx in range(num_iter):
                        predict_labels.append(clf.predict([test_dataset])[0])

                    predict_cnt = dict(collections.Counter(predict_labels))
                    predict_cnt_max = max(predict_cnt, key=lambda x: predict_cnt[x])
                    # print("pvecs: ", test_dataset)
                    # print(f'predict_result of {file.name}', predict_cnt)
                    # print(f"x limit: {xlims}, y limit: {ylims}, niter: {num_iter}")

                    if predict_cnt_max == 1:
                        junk_neuron_count += 1
                        junk_neuron_classified_count += 1
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
          f"\tjunk neuron break down:\n"
          f"\t\t{junk_neuron_short_length_count} identified by 0- or 1-length P-diagrams\n"
          f"\t\t{junk_neuron_singular_matrix_count} identified by singular matrix\n"
          f"\t\t{junk_neuron_classified_count} identified by classifier\n"
          f"\t{true_neuron_count} \t # true neurons identified by classifier\n"
          f"\t{failed_neuron_count}\t # failed neurons\n"
          f"\t{true_neuron_count / (true_neuron_count + failed_neuron_count + junk_neuron_count) * 100:.2f}%\t yield")
    
    # write the result to a text file
    with open(path_result/'summary.txt','w') as summary_file:
        summary_file.write("Summary of TMD filtering: \n")
        summary_file.write(f"Junk neuron count: {junk_neuron_count}, break down: \n")
        summary_file.write(f"\t\t{junk_neuron_short_length_count} identified by 0- or 1-length P-diagrams\n")
        summary_file.write(f"\t\t{junk_neuron_singular_matrix_count} identified by singular matrix\n")
        summary_file.write(f"\t\t{junk_neuron_classified_count} identified by classifier\n")
        summary_file.write(f"{true_neuron_count} true neurons identified by classifier\n")
        summary_file.write(f"{failed_neuron_count} SWC failed, refer to the log file (prediction.log)\n")
        summary_file.write(f"{true_neuron_count / (true_neuron_count + failed_neuron_count + junk_neuron_count) * 100:.2f}%\t yield")
    summary_file.close()

    return true_neuron_count, junk_neuron_count


if __name__ == '__main__':
    current_time = datetime.now().strftime("%H_%M_%S")
    
    parser = ArgumentParser(description="TMD filtering")
    parser.add_argument("--model", "-m", type=str, required=True, help="model")
    parser.add_argument("--filter", "-f", type=str, required=True, help="SWCs to be filtered")
  
    args = parser.parse_args()
    
    pickle_data = []
    with open(args.model, 'rb') as fr:
        try:
            while True:
                pickle_data.append(pickle.load(fr))
        except EOFError:
            pass
    fr.close()
    
    model_name = os.path.split(args.model)[1].split('.')[0]
    model = pickle_data[0]
    
    xlims = pickle_data[1]
    ylims= pickle_data[2]
    
    input_path = Path(args.filter)
    result_path = input_path.parent
    
    make_prediction(input_path, model, model_name, result_path, current_time, xlims, ylims)
    
