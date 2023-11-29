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
import pandas as pd
from recut_interface import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proofread', help='Folder of swcs required to be in global um space')
    parser.add_argument('automated', help='Recut `run-X` folder, SWCs required to be in global um space and not windowed')
    args = parser.parse_args()
    kwargs = {}
    kwargs['voxel_size_x'] = .4
    kwargs['voxel_size_y'] = .4
    kwargs['voxel_size_z'] = .4
    voxel_sizes = np.array([kwargs['voxel_size_x'], kwargs['voxel_size_y'], kwargs['voxel_size_z']], dtype=float)
    distance_threshold = 30

    automateds = gather_swcs(args.automated, voxel_sizes)
    proofreads = gather_swcs(args.proofread, voxel_sizes)
    p('Proofread', proofreads)
    p('Automateds', automateds)

    # match all data based off soma distance of the file names
    raw_matched = match_coord_keys(proofreads, automateds, distance_threshold)
    # keep only the matches whose automated succeeded (had more than the soma in the swc)
    matched = filter_fails(raw_matched)
    no_offset = None # inputs are required to be global
    params = [(proof, auto, no_offset) for proof, auto in matched]
    p('Match', raw_matched, proofreads)
    p('Automated reconstruction success', matched, raw_matched)

    diadem_scores = rm_none((handle_diadem_output(proof, auto) for proof, auto in matched))
    p('Diadem comparison', diadem_scores, matched)
    if len(diadem_scores):
        stats(np.mean, diadem_scores)
        stats(np.std, diadem_scores)
    exit(1)

    run = partial(run_accuracy, kwargs)
    accuracies = rm_none(map(run, params))
    acc_dict = {}
    acc_dict['recall'] = list(map(lambda x: x[0], accuracies))
    acc_dict['precision'] = list(map(lambda x: x[1], accuracies))
    acc_dict['F1'] = list(map(lambda x: x[2], accuracies))
    df = pd.DataFrame(data=acc_dict)
    p("Comparison", accuracies, proofreads)
    plot(df)

if __name__ == "__main__":
    main()
