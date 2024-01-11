import argparse
from recut_interface import *

DIADEM_XYZ_THRESH = 8.
anisotropic_scale = {.4 : .22, 1 : .16}

snd = lambda xs: xs[1]

def main(args):
    voxel_size = tuple(args.voxel_size)

    proofreads = gather_swcs(args.proofread, voxel_size)
    p('Proofread', proofreads)

    automateds = gather_swcs(args.automated, voxel_size)
    p('Automateds', automateds)

    # match all data based off soma distance of the file names
    raw_matched = match_coord_keys(proofreads, automateds, matching_distance)
    # keep only the matches whose automated succeeded (had more than the soma in the swc)
    matched = filter_fails(raw_matched)
    auto_reconstruct_fail_count = len(raw_matched) - len(matched)

    no_offset = None # inputs are required to be global
    params = [(proof, auto, no_offset) for proof, auto in matched]

    p('Auto with corresponding proofread match', raw_matched, proofreads)
    # report this with those below 2 std
    # p('Automated reconstruction success', matched, raw_matched)

    if not args.surface_only:
        # diadem_scores = rm_none((auto, handle_diadem_output(proof, auto, args.threshold, args.path_threshold, args.quiet) for proof, auto in matched))
        results = {auto : handle_diadem_output(proof, auto, args.threshold, args.path_threshold, not args.verbose) for proof, auto in matched}
        diadem_scores = [*results.values()]
        diadem_scores = rm_none(diadem_scores)
        p('Comparisons that did not crash DIADEM program', diadem_scores, matched)
        m = np.mean(diadem_scores)
        s = np.std(diadem_scores)
        # print('Raw mean: ' + str(m))
        mind = m - 2 * s
        maxd = m + 2 * s
        successes = []
        for d in diadem_scores:
            if (d > mind) and (d < maxd):
                successes.append(d)
        p("Total successful comparisons", successes, proofreads)

        below_2_stds = set(diadem_scores) - set(successes)
        p('Failed reconstruction', range(0, len(below_2_stds) + auto_reconstruct_fail_count), proofreads)
        print('Note: the two scores above do not always add up to the proofread count because of 1) inability to match auto with proofread 2) crash of diadem metric, etc.')

        if len(successes):
            stats(np.mean, successes)
            stats(np.std, successes)
        # import pdb; pdb.set_trace()

    if not args.diadem_only:
        accuracies = [handle_surface_output(voxel_size, param, not args.verbose) for param in params]
        p("Comparison", accuracies, proofreads)
        df = pd.concat(accuracies, axis=0)
        print(df.mean())
        # import pdb; pdb.set_trace()
        # acc_dict = {}
        # acc_dict['recall'] = list(map(lambda x: x[0], accuracies))
        # acc_dict['precision'] = list(map(lambda x: x[1], accuracies))
        # acc_dict['F1'] = list(map(lambda x: x[2], accuracies))
        # df = pd.DataFrame(data=acc_dict)
        # df = pd.concat([df, df.apply(['mean', 'median'])])
        # df1 = pd.concat([df, df.apply('mean')])
        # df = df.concat(df.agg(['sum', 'mean']))
        # print(df)
        # plot(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('proofread', help='Folder of swcs required to be in global um space')
    parser.add_argument('automated', help='Recut `run-X` folder, SWCs required to be in global um space and not windowed')
    parser.add_argument('-v', '--voxel-size', nargs=3, type=float, default=(1., 1., 1.), help='Specify the width in um of each dimension of the voxel e.g. .4 .4 .4 for 15x')
    # default threshold set from 6x data using max bifurc radii of proofread dataset as proscribed by
    # DIADEM metric/competition paper
    parser.add_argument('-t', '--threshold', type=float, help='The DIADEM xy, and z distance threshold, by default calculated based off the diameter of the largest bifurcation branch point in the dataset. Do not change this parameter or probe possible values unless you are specifically testing a reported hypothesis', default=DIADEM_XYZ_THRESH)
    parser.add_argument('-p', '--path-threshold', type=float, help='By default the path distance threshold is calculated based off the DIADEM xy and z distance threshold / anisotropic factor both of which are dataset specific. Do not set this parameter or probe possible values unless you are specifically testing a reported hypothesis')
    parser.add_argument('-s', '--surface-only', action='store_true', help='If passed only the volumetric surface accuracies will be computed')
    parser.add_argument('-d', '--diadem-only', action='store_true', help='If passed only the diadem accuracies will be computed')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.voxel_size[0] not in anisotropic_scale:
        raise Exception("Unrecognized dataset only .4 .4 .4 and 1 1 1 objectives allowed")
    print("Threshold um: " + str(args.threshold))
    if args.path_threshold is None:
        args.path_threshold = args.threshold / min(anisotropic_scale.values())
    main(args)
