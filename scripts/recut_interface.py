import subprocess
from glob import glob
from pathlib import Path

def swcs_to_dict(swcs, voxel_sizes):
    d = {}
    for swc in swcs:
        t = swc_to_coord(swc, voxel_sizes)
        d[t] = swc
    return d

# translates from pixel space to world-space um
def swc_to_coord(swc, voxel_sizes):
    coord_space = tuple(map(int, Path(swc).stem.split('_')[0].split('-')[-3:]))
    return (coord_space[0] * voxel_sizes[0], 
            coord_space[1] * voxel_sizes[1],
            coord_space[2] * voxel_sizes[2])

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

