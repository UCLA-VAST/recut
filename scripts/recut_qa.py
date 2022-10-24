import subprocess
import argparse
import re
from test_precision_recall import precision_recall

def parse_range(string):
    if string.contains('-'):
        m = re.match(r'(\d+)(?:-(\d+))?$', string)
        # ^ (or use .split('-'). anyway you like.)
        if not m:
            raise ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
        start = m.group(1)
        end = m.group(2) or start
        return list(range(int(start,10), int(end,10)+1))
    else:
        return list(int(string))


def call_recut(**kwargs):
    if kwargs['inferenced_path'] is None:
        # whitelist certain arguments to pass directly to recut
        include = ['min_radius', 'max_radius', 'open_denoise', 'open_steps', 'close_steps', 'fg_percent']
        args = "".join([f"--{k} {v} ".replace('_', '-') for k,v in kwargs.items() if k in include if v])
        cmd = f"recut {kwargs['image']} {args} --voxel-size {kwargs['voxel_size_x']} {kwargs['voxel_size_y']} {kwargs['voxel_size_z']}"
        print(cmd)

        output = subprocess.check_output(cmd.split()).strip().decode().split('\n')
        run_dir = [v.split()[-1] for v in output if "written to:" in v][0]
        kwargs['inferenced_path'] = f"{run_dir}/seeds"

    precision_recall(**kwargs)

    # return an object that will be a row of a pandas df for soma_recall, soma_precision, neuron yield, neuron precision, run, git commit, voxel size, for this given run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    # explicitly for precision and recall
    parser.add_argument('labeled_path', help='Path to ground truth somas with which to compare the inferences to')
    parser.add_argument('--inferenced-path', help="Use an existing input path run for inferenced soma location")
    parser.add_argument("--distance-threshold", type=float, default=17,
                        help="a minimum allowed distance between inferences and ground truth in µm "
                             "to consider them a match.")
    # recut arguments
    parser.add_argument('--min-radius', type=float, help="a min allowed radii for inferences for filtering purpose")
    parser.add_argument('--max-radius', type=float, help="a max allowed radii for inferences for filtering purpose")
    parser.add_argument('--open-denoise', type=float)
    parser.add_argument('--open-steps', type=float)
    parser.add_argument('--close-steps', type=float)
    parser.add_argument('--fg-percent', type=float)
    parser.add_argument("--voxel-size-x", type=float, default=1, help="x voxel size in µm.")
    parser.add_argument("--voxel-size-y", type=float, default=1, help="y voxel size in µm.")
    parser.add_argument("--voxel-size-z", type=float, default=1, help="z voxel size in µm.")
    args = parser.parse_args()

    call_recut(**vars(args))

if __name__ == "__main__":
    main()
