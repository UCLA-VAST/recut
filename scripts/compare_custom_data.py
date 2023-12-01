import argparse
from recut_interface import *

def main(args):
    voxel_sizes = tuple(args.voxel_sizes)

    proofreads = gather_swcs(args.proofread, voxel_sizes)
    p('Proofread', proofreads)

    automateds = gather_swcs(args.automated, voxel_sizes)
    p('Automateds', automateds)

    # match all data based off soma distance of the file names
    raw_matched = match_coord_keys(proofreads, automateds, distance_threshold)
    # keep only the matches whose automated succeeded (had more than the soma in the swc)
    matched = filter_fails(raw_matched)
    no_offset = None # inputs are required to be global
    params = [(proof, auto, no_offset) for proof, auto in matched]
    p('Match', raw_matched, proofreads)
    p('Automated reconstruction success', matched, raw_matched)

    if not args.disable_diadem:
        diadem_scores = rm_none((handle_diadem_output(proof, auto, args.threshold, args.quiet) for proof, auto in matched))
        p('Diadem comparison', diadem_scores, matched)
        if len(diadem_scores):
            stats(np.mean, diadem_scores)
            stats(np.std, diadem_scores)

    if not args.disable_surface:
        accuracies = rm_none((handle_surface_output(voxel_sizes, param, args.threshold, args.quiet) for param in params))
        acc_dict = {}
        acc_dict['recall'] = list(map(lambda x: x[0], accuracies))
        acc_dict['precision'] = list(map(lambda x: x[1], accuracies))
        acc_dict['F1'] = list(map(lambda x: x[2], accuracies))
        df = pd.DataFrame(data=acc_dict)
        p("Comparison", accuracies, proofreads)
        plot(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('proofread', help='Folder of swcs required to be in global um space')
    parser.add_argument('automated', help='Recut `run-X` folder, SWCs required to be in global um space and not windowed')
    parser.add_argument('voxel_sizes', nargs=3, type=float, help='Specify the width in um of each dimension of the voxel e.g. .4 .4 .4 for 15x')
    parser.add_argument('threshold', type=float)
    parser.add_argument('-s', '--disable-surface', action='store_true')
    parser.add_argument('-d', '--disable-diadem', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()
    main(args)
