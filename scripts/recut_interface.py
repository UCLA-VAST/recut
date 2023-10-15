import subprocess

def valid(voxel_size):
    return type(voxel_size) == list and len(voxel_size) == 3

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

