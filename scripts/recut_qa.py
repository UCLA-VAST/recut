import argparse
import re
# import pandas as pd
from test_precision_recall import precision_recall
from plots import get_hash
from TMD_classify import filter_dir_by_model
from recut_interface import call_recut

def parse_range(string):
    if string.contains('-'):
        m = re.match(r'(\d+)(?:-(\d+))?$', string)
        if not m:
            raise argparse.ArgumentTypeError(
                "'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
        start = m.group(1)
        end = m.group(2) or start
        return list(range(int(start, 10), int(end, 10) + 1))
    else:
        return list(int(string))


def persistence_precision_recall(run_dir):
    #precision_recall(**kwargs)

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

    run_dir = call_recut(**vars(args))
    persistence_precision_recall(run_dir)

if __name__ == "__main__":
    main()
