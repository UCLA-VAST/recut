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
from recut_interface import rm_none, filter_fails, gather_swcs
import pandas as pd

def compare_2_swcs(kwargs, proof_swc, auto_swc, offset):
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

def euc_dist(l, r):
    return np.sqrt((l[0] - r[0]) ** 2 + (l[1] - r[1]) ** 2 + (l[2] - r[2]) ** 2)

def match_closest(swc_dict, thresh, coord):
    coords = list(swc_dict.keys())
    # for this particular coord..
    dist = partial(euc_dist, coord)
    # distance to all others in swc_dict
    dists = list(map(dist, coords))
    coords_swcs_dists = zip(coords, swc_dict.values(), dists)
    sorted_coord_dists = sorted(coords_swcs_dists, key=lambda p: p[2])
    # print(sorted_coord_dists)
    smallest_tup = sorted_coord_dists[0]
    smallest_dist = smallest_tup[2]
    # filename of smallest dist
    return smallest_tup[1] if (thresh and (smallest_dist <= thresh)) or not thresh else None

# def match_markers(markers, swcs, thresh=None):
    # l = []
    # for coord_um in markers:
        # # FIXME parameterize
        # coord = (int(coord_um[0] / .4), int(coord_um[1] / .4),
                 # int(coord_um[2] / .4))
        # if coord in swcs:
            # swc = swcs[coord]
        # else:
            # closest_coord = match_closest(coord, swcs, thresh)
            # if closest_coord: 
                # swc = swcs[closest_coord];
            # else:
                # continue
        # tup = (coord, swc)
        # l.append(tup)
    # return l

def match_coord(f, p):
    (coord, proof_name) = p
    file = f(coord)
    return (proof_name, file) if file else None

# match each proofread against the larger set of automateds
def match_swcs(proofreads, automateds, thresh=None):
    closest_file = partial(match_closest, automateds, thresh)
    make_pair = partial(match_coord, closest_file)
    matched = map(make_pair, proofreads.items())
    return rm_none(matched)

def run_accuracy(kwargs, match):
    returncode = compare_2_swcs(kwargs, *match)
    is_success = extract_accuracy(returncode)
    if is_success:
        print("Success")
        print(is_success)
        print()
        print()
        return is_success
    print("Failed in compare")
    return None

def extract_offset(pair):
    (proof_fn, v1_fn) = pair
    file = open(v1_fn, 'r')
    first_line = file.readline()
    file.close()

    start = re.findall(r'\[.*?\]', first_line)[0]
    to_int = lambda token: int(re.sub("[^0-9]","",token))
    return (proof_fn, tuple(map(to_int, start.split(','))))

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proofread')
    parser.add_argument('automated')
    parser.add_argument('v1')
    # parser.add_argument('markers')
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
    matched = match_swcs(proofreads, automateds, distance_threshold)
    matched_v1 = match_swcs(proofreads, v1, distance_threshold)
    v1_offset = list(map(extract_offset, matched_v1))
    matched_dict = dict(matched)
    v1_dict = dict(v1_offset)
    params = [(key, matched_dict[key], v1_dict[key]) for key in set(matched_dict) & set(v1_dict)]
    discards = len(list(filter(lambda pair: 'discard' in pair[1], matched)))
    multis = len(list(filter(lambda pair: 'multi' in pair[1], matched)))
    others = len(list(filter(lambda pair: 'multi' not in pair[1] and 'discard' not in pair[1], matched)))
    print("Match count: " + str(len(matched)) + '/' + str(len(proofreads)))
    print("Match v1 count: " + str(len(matched_v1)) + '/' + str(len(proofreads)))
    print("Params count: " + str(len(params)) + '/' + str(len(proofreads)))
    print("Automateds discards count: " + str(discards) + '/' + str(len(matched)))
    print("Automateds multis count: " + str(multis) + '/' + str(len(matched)))
    print("Automateds others count: " + str(others) + '/' + str(len(matched)))
    print()

    # markers are always in world space (um)
    # (markers, _) = gather_markers(args.markers)
    # print("Marker count: " + str(len(markers)))
    # matched_markers = match_markers(markers, proofreads)
    # matched_markers_auto = match_markers(markers, automateds)
    # print("Marker-proof match count: " + str(len(matched_markers)) + '/' + str(len(markers)))
    # print("Marker-auto match count: " + str(len(matched_markers_auto)) + '/' + str(len(markers)))

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
