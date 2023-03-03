# -*- coding: utf-8 -*-
"""

@author: yanyanming77
Last update: 03-03-2023

To perform SWC filtering of junks/true neurons using pre-defined conditions and pre-trained random forest classifier

<03-03-2023: updated, only filter non-multi-components, as reconstructions are based on proofread somas>


input arguments:
    --filter/-f: (required) the directory to be filtered    
    --model/-m: (optional) the pre-trained model to use, default 'msn', to use clf_msn.sav, which was trained on true/junk based on MSNs (scale is in um)
                if specified 'all', then will use clf_all.sav, which was trained on true/junk based on all cell types
    --delete/-d: (optional) whether to delete the original recut output to save space
                do not =specify if don't want to delete
    --scale/-s: (optional) a list of scaling factor from voxel to um, for x y z and radii, deafult is 0.4,0.4,0.4,1
    
    
    ((deprecated) --bbox_threshold/-bb_thresh: (optional) integer, an extra filtering based on the bounding box volume, default not applying this function
                if specified, SWCs that have bounding box sizes below the percentage of this value will be defined as junk)

"""

  
import numpy as np
import pandas as pd
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
    
    components_list = [file for file in os.listdir(path_to_rename) if file not in ['seeds', 'log.csv']]
    total_num = len(components_list)
    num_range = [str(i) for i in range(1,total_num + 1)]
    for folder, num in zip(components_list, num_range):
        new_name = num + '_' + folder
        os.rename(path_to_rename/folder, path_to_rename/new_name)

def delete_original_outputs(path_to_delete: Path):
    shutil.rmtree(path_to_delete)

def copy_file(file, destination):
    
    """
    Copy files to their corresponding folder based on the filtering result
    """
    if file.is_file():
        shutil.copy(file, destination)
    else:
        shutil.copytree(file, dst = destination)
    
def scale_swc_files(root_dir, scale_x, scale_y, scale_z, scale_radii, scale_which):
    
    if scale_which == 'original_recut_swc':
        description = 'Scaling recut swc from voxel to um in input path: '
    elif scale_which == 'scaled_recut_swc_input':
        description = 'Scaling recut swc back to voxel in input path: '
    elif scale_which == 'scaled_recut_swc_result':
        description = 'Scaling recut swc back to voxel in result path: '
    
    for file in tqdm(list(root_dir.rglob('*.swc')), desc = description):
        comment_lines = [l for l in open(file,'r').readlines() if '#' in l]
        num_comment_lines = len(comment_lines)
        # read data
        df = pd.read_csv(file, delimiter=' ', skiprows=num_comment_lines, names = ['node_num1', 'type', 'x', 'y', 'z', 'radii', 'node_num2'])
        # scaling
        df['x'] = df['x'] * scale_x
        df['x'] = df['x'].round(decimals=3)
        df['y'] = df['y'] * scale_y
        df['y'] = df['y'].round(decimals=3)
        df['z'] = df['z'] * scale_z
        df['z'] = df['z'].round(decimals=3)
        df['radii'] = df['radii'] * scale_radii
        df['radii'] = df['radii'].round(decimals=5)
        
        with open(file,'w') as scaled_file:
            for cmt_line in comment_lines:
                scaled_file.write(cmt_line)
            for row in df.itertuples():
                scaled_file.write(f"{row.node_num1} {row.type} {row.x} {row.y} {row.z} {row.radii} {row.node_num2}\n")
        scaled_file.close()


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
    
    scale_swc_files(input_path, scale_x, scale_y, scale_z, scale_radii, 'original_recut_swc')
    

    # create a list of .swcs that only belong to non-multi-components and not the discard- component
    non_multi_list = [file for file in list(input_path.rglob("*.swc")) if 'multi-' not in file.parent.name and 'discard-' not in file.parent.name] 
    # create a list of folders/file in the filtering path that are single components/seeds/log.csv
    multi_and_other_list = [file for file in os.listdir(input_path) if 'multi-' in file or file in ['seeds','log.csv']]
    
    # copy multi-components, seeds folder and log.csv to the -for-proofread folder
    for file in tqdm(multi_and_other_list, desc = 'Copying multi-components: '):
        copy_file(input_path/file, path_true/file)

    # copy discard-component directly to the discarded folder
    discard_list = [file for file in list(os.listdir(input_path)) if 'discard-' in file]  
    for file in discard_list:
        shutil.copytree(input_path/file, path_junk/file)
    
    # TMD filtering on non-multi-components        
    for file in tqdm(non_multi_list, desc = "TMD filtering on non-multi-components: "):
        try:
            n = tmd.io.load_neuron(file.__str__())
  
        except Exception as e:
            log.error(f"tmd.io.load_neuron failed to load {file}: {e}")
            failed_neuron_count += 1
            copy_file(file.parent, path_failed/file.parent.name)
            continue
        try:
            pers2test = tmd.methods.get_ph_neuron(n, neurite_type=neurite_type)
            # Check for junk definition1: length of P-diagram is 0,1,2, ratio of pers_diagrams and num_of_dendrites > 30
            if len(pers2test) in (0, 1, 2) or len(pers2test)/len(n.dendrites) > 30: 
                # or bbox_volume < bbox_thresh_volume_low or bbox_volume > bbox_thresh_volume_high:
                junk_neuron_count += 1
                junk_neuron_short_length_count += 1 # short p_diagrams, too few dendrites, giant component, small bounding box
                copy_file(file.parent, path_junk/file.parent.name)
 
            elif len(pers2test) > 1:
                pers_image2test = np.empty(1, dtype='float64')  # initialize
                # Check fro junk definition2: singular matrix error when getting P-vecs
                try:
                    pers_image2test = tmd.analysis.get_persistence_image_data(pers2test, xlims=xlims, ylims=ylims)
                except np.linalg.LinAlgError:
                    junk_neuron_count += 1
                    junk_neuron_singular_matrix_count += 1
                    copy_file(file.parent, path_junk/file.parent.name)
                except Exception as e:
                    log.info(f"other error when extracting pvecs for {file}: {e}")
                    failed_neuron_count += 1
                    copy_file(file.parent, path_failed/file.parent.name)
                
                if len(pers_image2test) > 1:

                    test_dataset = pers_image2test.flatten()
                    predict_labels = []

                    # Train classifier with training images for selected number_of_trials
                    for idx in range(num_iter):
                        predict_labels.append(clf.predict([test_dataset])[0])

                    predict_cnt = dict(collections.Counter(predict_labels))
                    predict_cnt_max = max(predict_cnt, key=lambda x: predict_cnt[x])

                    if predict_cnt_max == 1:
                        junk_neuron_count += 1
                        junk_neuron_classified_count += 1
                        copy_file(file.parent, path_junk/file.parent.name)
                    elif predict_cnt_max == 2:
                        true_neuron_count += 1
                        copy_file(file.parent, path_true/file.parent.name)
                    else:
                        log.info(f"unexpected class {predict_cnt_max} for {file}")
                        failed_neuron_count += 1
                        copy_file(file.parent, path_failed/file.parent.name)

        except Exception as e:
            log.error(f"failure in classification for {file}: {e}")
            failed_neuron_count += 1
            copy_file(file.parent, path_failed/file.parent.name)
    
    # assign number to components in the proofread folder, for the convenience of distributing data for proofreading
    assign_number_to_folder(path_true)
    
    scale_swc_files(input_path, 1/scale_x, 1/scale_y, 1/scale_z, 1/scale_radii, 'scaled_recut_swc_input')
    scale_swc_files(path_result, 1/scale_x, 1/scale_y, 1/scale_z, 1/scale_radii, 'scaled_recut_swc_result')

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
        summary_file.write(f"The model used: {model_name}")
        # summary_file.write(f"Bounding box filtering threshold: {bbox_threshold_low}%\nBounding box size at {bbox_threshold_low}%th percentle: {bbox_thresh_volume_low} and Bounding box size at {bbox_threshold_high}%th percentle: {bbox_thresh_volume_high} ")
    summary_file.close()

    return true_neuron_count, junk_neuron_count


if __name__ == '__main__':
    current_time = datetime.now().strftime("%H_%M_%S")
    
    parser = ArgumentParser(description="TMD filtering")
    parser.add_argument("--model", "-m", type=str, required=False, default='msn', help="Whether to use the model trained on pure MSNs (msn) or all cell types (all)")
    parser.add_argument("--filter", "-f", type=str, required=True, help="SWCs to be filtered")
    parser.add_argument("--delete", "-d", action = 'store_true', help="Whether to delete the original recut output")
    parser.add_argument("--scale", '-s', required=False, default=[0.4,0.4,0.4,1], \
                        nargs=4, metavar=('scale_x','scale_y','scale_z','scale_radii'), type=float, help="scaling factor to convert to um unit, input in the order of x, y, z, radii")
    
    args = parser.parse_args()
    
    # if the default model: model_base_rf_15_26_00.sav
    if args.model == 'msn':
        the_model = Path(__file__).absolute().parent/'clf_msn.sav'
    elif args.model == 'all':
        the_model = Path(__file__).absolute().parent/'clf_all.sav'
    
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
    
    # scaling factor
    scale_x, scale_y, scale_z, scale_radii = args.scale[0], args.scale[1], args.scale[2], args.scale[3]
    print(f"scaling factor to convert to um: x: {scale_x}, y: {scale_y}, z: {scale_z}, radii: {scale_radii}")

    make_prediction(input_path, model, model_name, result_path, current_time, xlims, ylims)

    # whether to delete the original output
    if args.delete:
        delete_original_outputs(input_path)

    
