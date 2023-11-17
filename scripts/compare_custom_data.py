import argparse
import os
import subprocess
from glob import glob
from pathlib import Path
from collections import namedtuple
from test_precision_recall import gather_markers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import re
from recut_interface import *
import pandas as pd

diadem_path = "/home/kdmarrett/diadem/DiademMetric.jar"

def run_volumetric_accuracy(kwargs, proof_swc, auto_swc, offset):
    timeout_seconds = 4 * 60
    cmd = "/home/kdmarrett/recut/result/bin/recut {} --test {} --voxel-size {} {} {} --disable-swc-scaling --image-offsets {} {} {}".format(proof_swc,auto_swc, kwargs['voxel_size_x'], kwargs['voxel_size_y'], kwargs['voxel_size_z'], *offset)
    print("Run: " + cmd)
    try:
        result = subprocess.run(cmd.split(), capture_output=True, check=False, text=True,
                        timeout=timeout_seconds)
    except:
        print("Failed python call")
        return None
    return result

def run_diadem(proof_swc, auto_swc):
    timeout_seconds = 4 * 60
    cmd = "java -jar {}.jar -G {} -T {}".format(diadem_path, proof_swc, auto_swc)
    print("Run: " + cmd)
    try:
        result = subprocess.run(cmd.split(), capture_output=True, check=False, text=True,
                        timeout=timeout_seconds)
    except:
        print("Failed python call")
        return None
    return result

def extract(string, token):
    contains_token = lambda line: token in line
    # partial
    line = list(filter(contains_token, string.split('\n')))[0]
    return float(line.split()[-1])

def extract_accuracy(result):
    if result:
        if result.stdout:
            print('output: ', result.stdout)
            ex = partial(extract, result.stdout)
            return (ex('recall'), ex('precision'), ex('F1'))
        if result.stderr:
            print('error: ', result.stderr)
        if result.stderr != "" or result.returncode != 0:
          return False
        return True
    else:
        return False

def run_accuracy(kwargs, match):
    returncode = run_volumetric_accuracy(kwargs, *match)
    is_success = extract_accuracy(returncode)
    if is_success:
        print("Success")
        print(is_success)
        print()
        print()
        return is_success
    print("Failed in compare")
    return None

def plot(df):
    des = df.describe()
    cols = des.columns.to_list()
    bar_colors = ['red', 'yellow', 'darkorange']
    r = lambda row: des.loc[row,:].to_list()
    pct = lambda xs: list(map(lambda x: x *100, xs))
    means = pct(r("mean"))
    print(r("mean"))
    print(r("std"))
    # plt.plot(cols, means, label=cols, xerr=r("std"))
    plt.plot(cols, means, label=cols)
    plt.title("Neurite Accuracy N=" + str(r("count")[0]))
    # plt.set_ylabel("%")
    plt.savefig("/home/kdmarrett/fig/fg-.08-neurite-accuracy.png")
    import pdb; pdb.set_trace()
    # fig, ax = plt.subplots()
    # textures = ['///', '...', '', 'OOO']
    # texture = textures[0]
    # import pdb; pdb.set_trace()
    # for name, val in df.mean().items():
      # bars = ax.bar(val, edgecolor='black', hatch=texture, color='White', align='edge', label=name)
    # fig.tight_layout()
    # fig.show()

# explore the breakdown of how recut classified its component outputs only for 
# the final set of markers that proofreaders deemed proofreadable
def describe_recut_component_classifier(raw_matched):
    discards = len(list(filter(lambda pair: 'discard' in pair[1], raw_matched)))
    multis = len(list(filter(lambda pair: 'multi' in pair[1], raw_matched)))
    others = len(list(filter(lambda pair: 'multi' not in pair[1] and 'discard' not in pair[1], raw_matched)))
    print("Automateds discards count: " + str(discards) + '/' + str(len(raw_matched)))
    print("Automateds multis count: " + str(multis) + '/' + str(len(raw_matched)))
    print("Automateds others count: " + str(others) + '/' + str(len(raw_matched)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proofread')
    parser.add_argument('automated')
    parser.add_argument('v1')
    args = parser.parse_args()
    kwargs = {}
    kwargs['voxel_size_x'] = .4
    kwargs['voxel_size_y'] = .4
    kwargs['voxel_size_z'] = .4
    voxel_sizes = np.array([kwargs['voxel_size_x'], kwargs['voxel_size_y'], kwargs['voxel_size_z']], dtype=float)
    distance_threshold = 30

    # for dealing with runs with mixed human proofreads and 
    # automated outputs you can pass this to keep original automateds
    f = lambda n: '_' not in os.path.basename(n)

    # converts all swcs to world-space um
    automateds = gather_swcs(args.automated, voxel_sizes)
    proofreads = gather_swcs(args.proofread, voxel_sizes)
    v1 = gather_swcs(args.v1, voxel_sizes, f)
    print("Proofread count: " + str(len(proofreads)))
    print("Automateds count: " + str(len(automateds)))
    print("Original automateds count: " + str(len(v1)))

    # match all data based off soma distance
    raw_matched = match_coord_keys(proofreads, automateds, distance_threshold)
    # keep only the matches whose automated succeeded (had more than the soma in the swc)
    matched = filter_fails(raw_matched)
    matched_v1 = match_coord_keys(proofreads, v1, distance_threshold)
    v1_offset = list(map(extract_offset, matched_v1))
    matched_dict = dict(matched)
    v1_dict = dict(v1_offset)
    params = [(key, matched_dict[key], v1_dict[key]) for key in set(matched_dict) & set(v1_dict)]
    print("Raw match count: " + str(len(raw_matched)) + '/' + str(len(proofreads)))
    print("Match count: " + str(len(matched)) + '/' + str(len(proofreads)))
    print("Automated sucess rate on final proofreads: " + str(len(matched)) + '/' + str(len(raw_matched)))
    print("Match v1 count: " + str(len(matched_v1)) + '/' + str(len(proofreads)))
    print("Params count: " + str(len(params)) + '/' + str(len(proofreads)))

    print()

    import pdb; pdb.set_trace()
    run = partial(run_accuracy, kwargs)
    accuracies = rm_none(map(run, params))
    acc_dict = {}
    acc_dict['recall'] = list(map(lambda x: x[0], accuracies))
    acc_dict['precision'] = list(map(lambda x: x[1], accuracies))
    acc_dict['F1'] = list(map(lambda x: x[2], accuracies))
    df = pd.DataFrame(data=acc_dict)
    print("Comparison count: " + str(len(accuracies)) + '/' + str(len(proofreads)))
    plot(df)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
