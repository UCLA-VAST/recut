# -*- coding: utf-8 -*-
"""

@author: yanyanming77
Last update: 02-01-2023

To perform SWC filtering of junks/true neurons using pre-defined conditions and pre-trained random forest classifier

input arguments: 
    --model/-m: the pre-trained model to use, default most up-to-date model is model_base_rf_14_10_17.sav
    --filter/-f: the directory to be filtered 
    --delete/-d: whether to delete the original recut output to save space
                do not =specify if don't want to delete
    --bbox_threshold/-bb_thresh: integer, an extra filtering based on the bounding box volume, default 30%
                if specified, SWCs that have bounding box sizes below the percentage of this value will be defined as junk

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

def assign_number_to_folder(path_to_rename: Path):
    total_num = len(os.listdir(path_to_rename))
    num_range = [str(i) for i in range(1,total_num + 1)]
    for folder, num in zip(os.listdir(path_to_rename), num_range):
        new_name = num + '_' + folder
        os.rename(path_to_rename/folder, path_to_rename/new_name)
        
def delete_original_outputs(path_to_delete: Path):
    shutil.rmtree(path_to_delete)

def copy_file(file, destination, copy_type):
    
    """
    Copy files to their corresponding folder based on the filtering result
    """
    
    if "multi-component" in file.parent.name:
        multi_component_dest_dir = destination / file.parent.name
        multi_component_dest_dir.mkdir(exist_ok=True)
        
        for n_file in file.parent.glob(f"{file.name[:-len(file.suffix)]}.*"):
            shutil.copy(n_file, multi_component_dest_dir)
        # save one copy of <seeds>, <log.csv>, <tif>  in each multi-component folder
        if not os.path.exists(multi_component_dest_dir/'seeds'):
            shutil.copytree(file.parent/'seeds', multi_component_dest_dir/'seeds')
        if not os.path.exists(multi_component_dest_dir/'*.csv'):
            for csv_file in file.parent.rglob(r'*.csv'):
                shutil.copy(file.parent/csv_file.name, multi_component_dest_dir/csv_file.name)
        
        if copy_type == 'true': # only copy tif if classified as true neuron
            if len(list(file.parent.rglob(r'*.tif'))) > 0 and len(list(multi_component_dest_dir.rglob(r'*.tif'))) == 0:
                for tif_file in file.parent.rglob(r'*.tif'):
                    shutil.copy(file.parent/tif_file.name, multi_component_dest_dir/tif_file.name)
                
    else:
        shutil.copytree(file.parent, dst=destination / file.parent.name)

    # save one copy of overall <seeds> folder and <log.csv> in the result folder
    if not os.path.exists(destination.parent/'seeds'):
        shutil.copytree(input_path/'seeds', destination.parent/'seeds')
    if not os.path.exists(destination.parent/'log.csv'):
        shutil.copy(input_path/'log.csv', destination.parent/'log.csv')


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
    
    path_result = result_path / f"{input_path.name}_filtered_{current_time}"
    path_result.mkdir(exist_ok=True)
    path_junk = path_result / Path(f"{input_path.name}-discarded-components")
    path_junk.mkdir(exist_ok=True)
    path_true = path_result / Path(f"{input_path.name}-components-for-proofread")
    path_true.mkdir(exist_ok=True)
    path_failed = path_result / Path(f"{input_path.name}-failed")
    path_failed.mkdir(exist_ok=True)
    log_file = path_result / "error_log.log"
    log.basicConfig(filename=str(log_file), level=log.INFO)
    log.FileHandler(log_file.__str__(), mode="w")  # rewrite the file instead of appending
    
    ### validate xy limits
    print(f"Current x limits: {xlims}, Current y limits: {ylims}")
    
    
    bbox_volume = []
    ### find the mean of the volume of each bounding box (if not soma only)
    for file in tqdm(list(input_path.rglob("*.swc")), desc = "Calculating bounding boxes: "):
        try:
            n = tmd.io.load_neuron(file.__str__())
            bbox_n = n.get_bounding_box()
            bbox_length = abs(np.subtract(bbox_n[0], bbox_n[1]))
            bbox_volume.append(bbox_length[0] * bbox_length[1] * bbox_length[2])
        except:
            pass
        
    bbox_thresh_volume = np.percentile(np.array(bbox_volume), bbox_threshold)
    
    for file in tqdm(list(input_path.rglob("*.swc")), desc="classification: "):
        try:
            ### try morphio as well
            n = tmd.io.load_neuron(file.__str__())
            
            try:
                bbox_n = n.get_bounding_box()
                bbox_length = abs(np.subtract(bbox_n[0], bbox_n[1]))
                bbox_volume = bbox_length[0] * bbox_length[1] * bbox_length[2]
            except:
                bbox_volume = 0
                
            # n = tmd.io.load_neuron_from_morphio(file.__str__())
        except Exception as e:
            log.error(f"tmd.io.load_neuron failed to load {file}: {e}")
            failed_neuron_count += 1
            copy_file(file, path_failed, 'failed')
            continue
        try:
            pers2test = tmd.methods.get_ph_neuron(n, neurite_type=neurite_type)
            # Check for junk definition1: length of P-diagram is 0,1,2
            # Add another condition to filter junks:
            # the ratio of len(persistence_diagrams)/number of dendrites, if out of the bound of 1~17.5, then assign junk
            # if len(pers2test) in (0, 1, 2) or len(pers2test)/len(n.dendrites) > 17.5 or len(n.dendrites) < 2 or n.get_bounding_box() < bbox_mean:
            if len(pers2test) in (0, 1, 2) or len(pers2test)/len(n.dendrites) > 17.5 or bbox_volume < bbox_thresh_volume:
                junk_neuron_count += 1
                junk_neuron_short_length_count += 1 # short p_diagrams, too few dendrites, giant component, small bounding box
                copy_file(file, path_junk, 'junk')
 
            elif len(pers2test) > 1:
                pers_image2test = np.empty(1, dtype='float64')  # initialize
                # Check fro junk definition2: singular matrix error when getting P-vecs
                try:
                    pers_image2test = tmd.analysis.get_persistence_image_data(pers2test, xlims=xlims, ylims=ylims)
                except np.linalg.LinAlgError:
                    junk_neuron_count += 1
                    junk_neuron_singular_matrix_count += 1
                    copy_file(file, path_junk, 'junk')
                except Exception as e:
                    log.info(f"other error when extracting pvecs for {file}: {e}")
                    failed_neuron_count += 1
                    copy_file(file, path_failed, 'failed')
                
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
                        copy_file(file, path_junk, 'junk')
                    elif predict_cnt_max == 2:
                        true_neuron_count += 1
                        copy_file(file, path_true, 'true')
                    else:
                        log.info(f"unexpected class {predict_cnt_max} for {file}")
                        failed_neuron_count += 1
                        copy_file(file, path_failed, 'failed')

        except Exception as e:
            log.error(f"failure in classification for {file}: {e}")
            failed_neuron_count += 1
            copy_file(file, path_failed, 'failed')
    
    # assign number to components in the proofread folder, for the convenience of distributing data for proofreading
    assign_number_to_folder(path_true)

    print("Summary of Classification:\n"
          f"\t{junk_neuron_count} \t # junk neurons\n"
          f"\tjunk neuron break down:\n"
          f"\t\t{junk_neuron_short_length_count} identified by short P-diagrams/dendrites, or super small/big component\n"
          f"\t\t{junk_neuron_singular_matrix_count} identified by singular matrix\n"
          f"\t\t{junk_neuron_classified_count} identified by classifier\n"
          f"\t{true_neuron_count} \t # true neurons identified by classifier\n"
          f"\t{failed_neuron_count}\t # failed neurons\n"
          f"\t{true_neuron_count / (true_neuron_count + failed_neuron_count + junk_neuron_count) * 100:.2f}%\t yield")
    
    # write the result to a text file
    with open(path_result/'filter_summary.txt','w') as summary_file:
        summary_file.write("Summary of TMD filtering: \n")
        summary_file.write(f"Junk neuron count: {junk_neuron_count}, break down: \n")
        summary_file.write(f"\t\t{junk_neuron_short_length_count} identified by short P-diagrams/dendrites, or super small/big component\n")
        summary_file.write(f"\t\t{junk_neuron_singular_matrix_count} identified by singular matrix\n")
        summary_file.write(f"\t\t{junk_neuron_classified_count} identified by classifier\n")
        summary_file.write(f"{true_neuron_count} true neurons identified by classifier\n")
        summary_file.write(f"{failed_neuron_count} SWC failed, refer to the log file (prediction.log)\n")
        summary_file.write(f"{true_neuron_count / (true_neuron_count + failed_neuron_count + junk_neuron_count) * 100:.2f}%\t yield\n\n")
        summary_file.write(f"x limit: {xlims}, y limit: {ylims}\n")
        summary_file.write(f"Bounding box filtering threshold: {bbox_threshold}%")
    summary_file.close()

    return true_neuron_count, junk_neuron_count


if __name__ == '__main__':
    current_time = datetime.now().strftime("%H_%M_%S")
    
    parser = ArgumentParser(description="TMD filtering")
    parser.add_argument("--model", "-m", type=str, required=False, help="model")
    parser.add_argument("--filter", "-f", type=str, required=True, help="SWCs to be filtered")
    parser.add_argument("--delete", "-d", action = 'store_true', help="Whether to delete the original recut output")
    parser.add_argument("--bbox_threshold", "-bb_thresh", type=int, required=False, help="Percentile of bounding box below which to be treated as junk")
    
    args = parser.parse_args()
    
    # if the default model: model_base_rf_17_55_57.sav
    if not args.model:
        the_model = Path(__file__).absolute().parent/'model_base_rf_14_10_17.sav'
    elif args.model:
        the_model = args.model
    
    pickle_data = []
    with open(the_model, 'rb') as fr:
        try:
            while True:
                pickle_data.append(pickle.load(fr))
        except EOFError:
            pass
    fr.close()
    
    model_name = os.path.split(the_model)[1].split('.')[0]
    print(f"The classifier used: {model_name}")
    model = pickle_data[0]
    
    xlims = pickle_data[1]
    ylims= pickle_data[2]
    
    input_path = Path(args.filter)
    result_path = input_path.parent

    if not args.bbox_threshold:
        bbox_threshold = 30
    elif args.bbox_threshold:
        bbox_threshold = args.bbox_threshold
    print(f"Threshold of bounding box: {bbox_threshold}% percentile of the bounding box sizes")

    make_prediction(input_path, model, model_name, result_path, current_time, xlims, ylims)

    # whether to delete the original output
    if args.delete:
        delete_original_outputs(input_path)

    
