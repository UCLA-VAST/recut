import subprocess
from glob import glob
from pathlib import Path
from functools import partial
import numpy as np
import re

# new file format is in world space um units
def fn_to_coord(fn_name):
    start = re.findall(r'\[.*?\]', fn_name)[0][1:-1]
    to_int = lambda token: round(float(token))
    return tuple(map(to_int, start.split(',')))

# translates from pixel space to world-space um
def swc_to_coord(swc, voxel_sizes):
    if '[' in swc:
        return fn_to_coord(swc)
    else:
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
def match_coord_keys(proofreads, automateds, thresh=None):
    closest_file = partial(match_closest, automateds, thresh)
    make_pair = partial(match_coord, closest_file)
    matched = map(make_pair, proofreads.items())
    return rm_none(matched)

def extract_offset(pair):
    (proof_fn, v1_fn) = pair
    file = open(v1_fn, 'r')
    first_line = file.readline()
    file.close()

    start = re.findall(r'\[.*?\]', first_line)[0]
    to_int = lambda token: int(re.sub("[^0-9]","",token))
    return (proof_fn, tuple(map(to_int, start.split(','))))

def extract_soma_coord(pair):
    (proof_fn, v1_fn) = pair
    with open(proof_fn, 'r') as file:
        _ = file.readline()
        return tuple(map(float, file.readline().split()[2:-1]))

def extract_soma_radius(pair):
    (proof_fn, v1_fn) = pair
    with open(proof_fn, 'r') as file:
        _ = file.readline()
        return float(file.readline().split()[-2])
