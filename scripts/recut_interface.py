import subprocess
from glob import glob
import os
from pathlib import Path
from functools import partial
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

matching_distance = 30

# new file format is in world space um units
def fn_to_coord(fn_name):
    start = re.findall(r'\[.*?\]', fn_name)[0][1:-1]
    to_int = lambda token: round(float(token))
    return tuple(map(to_int, start.split(',')))

# translates from pixel space to world-space um
def swc_to_coord(swc, voxel_sizes):
    if '[' in swc:
        return fn_to_coord(swc)
    # if len(Path(swc).stem.split('_') > 1:
    coord_space = tuple(map(int, Path(swc).stem.split('_')[0].split('-')[-3:]))
    return (coord_space[0] * voxel_sizes[0], 
            coord_space[1] * voxel_sizes[1],
            coord_space[2] * voxel_sizes[2])

def swcs_to_dict(swcs, voxel_sizes):
    d = {}
    for swc in swcs:
        t = swc_to_coord(swc, voxel_sizes)
        d[t] = swc
    return d

def valid(voxel_size):
    return type(voxel_size) == list and len(voxel_size) == 3

rm_none = lambda iterable: list(filter(lambda x: x is not None, iterable))

# keep swcs that have more topology than just the soma
def is_substantial(pair):
    (proof_swc, automated_swc) = pair
    with open(automated_swc, 'r') as fp:
        return pair if len(fp.readlines()) > 3 else None

def filter_fails(matched):
    return rm_none(map(is_substantial, matched))

def gather_swc_files(path, f=None):
    pattern = path + "/**/*.swc"
    swcs = glob(pattern, recursive=True)
    if (f):
        swcs = filter(f, swcs)
    return swcs

def gather_swcs(path, voxel_sizes, f=None):
    pattern = path + "/**/*.swc"
    swcs = glob(pattern, recursive=True)
    if (f):
        swcs = filter(f, swcs)
    return swcs_to_dict(swcs, voxel_sizes)

def call_recut(**kwargs):
    if 'inferenced_path' in kwargs:
        run_dir = kwargs['inferenced_path']
    else:
        # whitelist certain arguments to pass directly to recut
        include = ['min_radius', 'max_radius', 'open_steps', 'close_steps', 'fg_percent',
                   'preserve_topology', 'seed_action']
        args = "".join([f"--{k} {v} ".replace('_', '-') for k, v in kwargs.items() if k in include])
        cmd = f"/home/kdmarrett/recut/result/bin/recut {kwargs['image']} --seeds {kwargs['image']}/somas {args} --voxel-size {kwargs['voxel_size_x']} {kwargs['voxel_size_y']} " \
              f"{kwargs['voxel_size_z']}"
        print('  ')
        print(cmd)

        output = subprocess.check_output(cmd.split()).strip().decode().split('\n')
        run_dir = [v.split()[-1] for v in output if "written to:" in v][0]
        kwargs['inferenced_path'] = f"{run_dir}/seeds"
    return run_dir;

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

def match_coord(f, p):
    (coord, proof_name) = p
    file = f(coord)
    return (proof_name, file) if file else None

# match each proofread against the larger set of automateds
def match_coord_keys(proofreads, automateds, thresh=None, rm=True):
    closest_file = partial(match_closest, automateds, thresh)
    make_pair = partial(match_coord, closest_file)
    matched = map(make_pair, proofreads.items())
    return rm_none(matched) if rm else matched

def extract_offset(pair):
    (proof_fn, v1_fn) = pair
    file = open(v1_fn, 'r')
    first_line = file.readline()
    file.close()

    start = re.findall(r'\[.*?\]', first_line)[0]
    to_int = lambda token: int(re.sub("[^0-9]","",token))
    return (proof_fn, tuple(map(to_int, start.split(','))))

def extract_soma_coord(pair):
    (proof_fn, _) = pair
    with open(proof_fn, 'r') as file:
        _ = file.readline()
        return tuple(map(float, file.readline().split()[2:-2]))

def extract_soma_radius(pair):
    (proof_fn, _) = pair
    with open(proof_fn, 'r') as file:
        _ = file.readline()
        return float(file.readline().split()[-2])

def run_surface_accuracy(voxel_sizes, proof_swc, auto_swc, offset, quiet=False):
    timeout_seconds = 4 * 60
    recut_path = "/home/kdmarrett/recut/result/bin/recut"
    if not os.path.exists(recut_path):
        recut_path = "recut"
    cmd = "{} {} --test {} --voxel-size {} {} {}".format(recut_path, proof_swc, auto_swc, *voxel_sizes)
    if offset:
        cmd += "--disable-swc-scaling --image-offsets {} {} {}".format(*offset)
    if not quiet:
        print("Run: " + cmd)
    try:
        result = subprocess.run(cmd.split(), capture_output=True, check=False, text=True,
                        timeout=timeout_seconds)
    except:
        if not quiet:
            print("Failed python call")
        return None
    return result

def run_diadem(proof_swc, auto_swc, threshold, path_threshold, quiet=False):
    timeout_seconds = 4 * 60
    diadem_path = os.path.expanduser("~/diadem/DiademMetric.jar")
    if not os.path.exists(diadem_path):
        print(f'Install the diadem metric such that the jar file is at this exact location: {diadem_path}\nYou also need to have java installed, do so via something like: `sudo apt install default-jre`')
    cmd = "java -jar {} -G {} -T {} --xy-threshold {} --z-threshold {} --excess-nodes false --xyPathThresh {} --zPathThresh {}".format(diadem_path, proof_swc, auto_swc, threshold, threshold, path_threshold, path_threshold)
    if not quiet:
        print("Run: " + cmd)
    try:
        result = subprocess.run(cmd.split(), capture_output=True, check=False, text=True,
                        timeout=timeout_seconds)
    except:
        if not quiet:
            print("Failed python call")
        return None
    return result

def last_float(line):
    return float(line.split()[-1])

def extract(string, token):
    contains_token = lambda line: token in line
    line = list(filter(contains_token, string.split('\n')))[0]
    return last_float(line)

def extract_accuracy(result, diadem=False, quiet=False):
    if result:
        if result.stdout:
            if not quiet:
                print('output: ', result.stdout)
            if diadem:
                return last_float(result.stdout)
            else:
                ex = partial(extract, result.stdout)
                # uncomment to get the surface to surface accuracies
                # return ex('recall'), ex('precision'), ex('F1')
                return ex('recall'), ex('precision'), 0
        if result.stderr:
            if not quiet:
                print('error: ', result.stderr)
        if result.stderr != "" or result.returncode != 0:
          return False
        return True
    else:
        return False

def handle_diadem_output(proof_swc, auto_swc, threshold, path_threshold, quiet=False):
    returncode = run_diadem(proof_swc, auto_swc, threshold, path_threshold, quiet)
    is_success = extract_accuracy(returncode, True, quiet)
    if is_success:
        if not quiet:
            print("Success")
            print(is_success)
            print()
            print()
        return is_success
    if not quiet:
        print("Failed in diadem")
    return None

def handle_surface_output(kwargs, match, quiet=False):
    returncode = run_surface_accuracy(kwargs, *match, quiet)
    is_success = extract_accuracy(returncode, False, quiet)
    if is_success:
        if not quiet:
            print("Success")
            print(is_success)
            print()
            print()
        return is_success
    if not quiet:
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

def stats(f, xs):
    print(f'{f.__name__}: {f(xs)}')

def p(name, l, l2=None):
    print(f'{name} count: {str(len(l))}', end='')
    if l2:
        print('/' + str(len(l2)) + ', ' + str(float(len(l)) / len(l2)), end='')
    print()

# def get_log(run_dir):
    # log = run_dir + "/log.swc"
    # f = pd.read_csv(log, header=None).T
    # f = f.rename(columns=f.iloc[0]).drop(f.index[0])[['VC', 'CC', 'SDF', 'TC+TP', 'Thread count', 'Original voxel count', 'Selected', 'Neuron count']].astype({'Thread count':'int'})
