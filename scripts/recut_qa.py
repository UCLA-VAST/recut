import subprocess
import argparse
import re
# import pandas as pd
from test_precision_recall import precision_recall
from plots import get_hash
from TMD_classify import filter_dir_by_model


def parse_range(string):
    if string.contains('-'):
        m = re.match(r'(\d+)(?:-(\d+))?$', string)
        # ^ (or use .split('-'). anyway you like.)
        if not m:
            raise argparse.ArgumentTypeError(
                "'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
        start = m.group(1)
        end = m.group(2) or start
        return list(range(int(start, 10), int(end, 10) + 1))
    else:
        return list(int(string))


def call_recut(**kwargs):
    if kwargs['inferenced_path'] is None:
        # whitelist certain arguments to pass directly to recut
        include = ['min_radius', 'max_radius', 'open_steps', 'close_steps', 'fg_percent']
        args = "".join([f"--{k} {v} ".replace('_', '-') for k, v in kwargs.items() if k in include if v])
        cmd = f"recut {kwargs['image']} {args} --voxel-size {kwargs['voxel_size_x']} {kwargs['voxel_size_y']} " \
              f"{kwargs['voxel_size_z']}"
        print(cmd)

        output = subprocess.check_output(cmd.split()).strip().decode().split('\n')
        run_dir = [v.split()[-1] for v in output if "written to:" in v][0]
        kwargs['inferenced_path'] = f"{run_dir}/seeds"
    else:
        run_dir = kwargs['inferenced_path']

    #precision_recall(**kwargs)

    git_hash = get_hash()

    true_neuron_count, junk_neuron_count = filter_dir_by_model(run_dir, kwargs['model'])
    neuron_count = true_neuron_count + junk_neuron_count
    fraction_true_neurons = true_neuron_count / neuron_count
    print(f"Neuron counts true: {true_neuron_count} junk: {junk_neuron_count}")
    print(f"True neuron %: {100 * fraction_true_neurons}")

    # TODO return a pandas dataframe with 1 row for this particular run with columns for soma_recall, soma_precision,
    #  neuron yield, neuron precision, run dir, git hash, voxel size, TMD model name, for this given run note that
    #  this function will eventually be called multiple times, each run being placed in a new row by the caller the
    #  run of this whole script will output a single csv from the pd.df, each row corresponding to a recut run
    # df = pd.DataFrame(kwargs)

    # number of neurons that pass as true positives based on the topology descriptor trained in kwargs['model']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    # explicitly for precision and recall
    parser.add_argument('labeled-path', help='Path to ground truth somas with which to compare the inferences to')
    parser.add_argument('--inferenced-path', help="Use an existing input path run for inferenced soma location")
    parser.add_argument("--distance-threshold", type=float, default=17,
                        help="a minimum allowed distance between inferences and ground truth in µm "
                             "to consider them a match.")
    # recut arguments
    parser.add_argument('--min-radius', type=float, help="a min allowed radii for inferences for filtering purpose")
    parser.add_argument('--max-radius', type=float, help="a max allowed radii for inferences for filtering purpose")
    parser.add_argument('--open-steps', type=float)
    parser.add_argument('--close-steps', type=float)
    parser.add_argument('--fg-percent', type=float)
    parser.add_argument("--voxel-size-x", type=float, default=1, help="x voxel size in µm.")
    parser.add_argument("--voxel-size-y", type=float, default=1, help="y voxel size in µm.")
    parser.add_argument("--voxel-size-z", type=float, default=1, help="z voxel size in µm.")
    # explicitly for TMD_classify
    default_model = "model_base_rf_13_50_49.sav"
    parser.add_argument("--model", default=f"{default_model}",
                        help=f"path to the TMD classifier model; defaults to the MSN model: {default_model}")
    args = parser.parse_args()

    call_recut(**vars(args))

if __name__ == "__main__":
    main()
