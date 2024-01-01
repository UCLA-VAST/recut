import argparse
from recut_interface import *

DIADEM_XYZ_THRESH = 8
anisotropic_scale = {.4 : .22, 1 : .16}

snd = lambda xs: xs[1]

def main(args):
    voxel_sizes = tuple(args.voxel_sizes)

    proofreads = gather_swcs(args.proofread, voxel_sizes)
    p('Proofread', proofreads)

    automateds = gather_swcs(args.automated, voxel_sizes)
    p('Automateds', automateds)

    # match all data based off soma distance of the file names
    raw_matched = match_coord_keys(proofreads, automateds, matching_distance)
    # keep only the matches whose automated succeeded (had more than the soma in the swc)
    matched = filter_fails(raw_matched)
    no_offset = None # inputs are required to be global
    params = [(proof, auto, no_offset) for proof, auto in matched]
    p('Match', raw_matched, proofreads)
    p('Automated reconstruction success', matched, raw_matched)

    if not args.surface_only:
        # diadem_scores = rm_none((auto, handle_diadem_output(proof, auto, args.threshold, args.path_threshold, args.quiet) for proof, auto in matched))
        results = {auto : handle_diadem_output(proof, auto, args.threshold, args.path_threshold, not args.verbose) for proof, auto in matched}
        diadem_scores = [*results.values()]
        diadem_scores = rm_none(diadem_scores)
        p('Diadem comparison', diadem_scores, matched)
        if len(diadem_scores):
            stats(np.mean, diadem_scores)
            stats(np.std, diadem_scores)

    if not args.diadem_only:
        accuracies = [handle_surface_output(voxel_sizes, param, not args.verbose) for param in params]
        p("Comparison", accuracies, proofreads)
        df = pd.concat(accuracies, axis=0)
        import pdb; pdb.set_trace()
        # acc_dict = {}
        # acc_dict['recall'] = list(map(lambda x: x[0], accuracies))
        # acc_dict['precision'] = list(map(lambda x: x[1], accuracies))
        # acc_dict['F1'] = list(map(lambda x: x[2], accuracies))
        # df = pd.DataFrame(data=acc_dict)
        # df = pd.concat([df, df.apply(['mean', 'median'])])
        # df1 = pd.concat([df, df.apply('mean')])
        # df = df.concat(df.agg(['sum', 'mean']))
        print(df.mean())
        # print(df)
        # plot(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('proofread', help='Folder of swcs required to be in global um space')
    parser.add_argument('automated', help='Recut `run-X` folder, SWCs required to be in global um space and not windowed')
    parser.add_argument('-v', '--voxel_sizes', nargs=3, type=float, default=(1., 1., 1.), help='Specify the width in um of each dimension of the voxel e.g. .4 .4 .4 for 15x')
    # default threshold set from 6x data using max bifurc radii of proofread dataset as proscribed by
    # DIADEM metric/competition paper
    parser.add_argument('-t', '--threshold', type=float, help='The DIADEM xy, and z distance threshold, by default calculated based off the diameter of the largest bifurcation branch point in the dataset. Do not change this parameter or probe possible values unless you are specifically testing a reported hypothesis', default=DIADEM_XYZ_THRESH)
    parser.add_argument('-p', '--path-threshold', type=float, help='By default the path distance threshold is calculated based off the DIADEM xy and z distance threshold / anisotropic factor both of which are dataset specific. Do not set this parameter or probe possible values unless you are specifically testing a reported hypothesis')
    parser.add_argument('-s', '--surface-only', action='store_true', help='If passed only the volumetric surface accuracies will be computed')
    parser.add_argument('-d', '--diadem-only', action='store_true', help='If passed only the diadem accuracies will be computed')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.voxel_sizes[0] not in anisotropic_scale:
        raise Exception("Unrecognized dataset only .4 .4 .4 and 1 1 1 objectives allowed")
    if args.path_threshold is None:
        args.path_threshold = args.threshold / min(anisotropic_scale.values())
    main(args)
