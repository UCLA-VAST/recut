import argparse
from recut_interface import *

def main(args):
    voxel_sizes = tuple(args.voxel_sizes)

    # for dealing with runs with mixed human proofreads and 
    # automated outputs you can pass this to keep original automateds
    f = lambda n: '_' not in os.path.basename(n)

    # converts all swcs to world-space um
    automateds = gather_swcs(args.automated, voxel_sizes)
    proofreads = gather_swcs(args.proofread, voxel_sizes)
    v1 = gather_swcs(args.v1, voxel_sizes, f)
    global_proofreads = gather_swcs(args.globals, voxel_sizes) if args.globals else None
    print("Proofread count: " + str(len(proofreads)))
    if global_proofreads:
        print("Global proofread count: " + str(len(global_proofreads)))
    print("Automateds count: " + str(len(automateds)))
    print("Original automateds count: " + str(len(v1)))

    # match all data based off soma distance
    raw_matched = match_coord_keys(proofreads, automateds, distance_threshold)
    # keep only the matches whose automated succeeded (had more than the soma in the swc)
    matched = filter_fails(raw_matched)
    matched_v1 = match_coord_keys(proofreads, v1, distance_threshold)
    matched_globals = match_coord_keys(global_proofreads, automateds, distance_threshold) if args.globals else None
    v1_offset = list(map(extract_offset, matched_v1))
    matched_dict = dict(matched)
    v1_dict = dict(v1_offset)
    params = [(key, matched_dict[key], v1_dict[key]) for key in set(matched_dict) & set(v1_dict)]
    print("Raw match count: " + str(len(raw_matched)) + '/' + str(len(proofreads)))
    print("Match count: " + str(len(matched)) + '/' + str(len(proofreads)))
    print("Automated sucess rate on final proofreads: " + str(len(matched)) + '/' + str(len(raw_matched)))
    print("Match v1 count: " + str(len(matched_v1)) + '/' + str(len(proofreads)))
    print("Params count: " + str(len(params)) + '/' + str(len(proofreads)))

    print()
    if args.globals:
        diadem_scores = rm_none((handle_diadem_output(proof, auto, args.quiet) for proof, auto in matched_globals))
        print("Diadem comparison count: " + str(len(diadem_scores)) + '/' + str(len(matched_globals)))
        if len(diadem_scores):
            stats(np.mean, diadem_scores)
            stats(np.std, diadem_scores)
        exit(1)

    run = partial(handle_surface_output, kwargs)
    accuracies = rm_none(map(run, params))
    acc_dict = {}
    acc_dict['recall'] = list(map(lambda x: x[0], accuracies))
    acc_dict['precision'] = list(map(lambda x: x[1], accuracies))
    acc_dict['F1'] = list(map(lambda x: x[2], accuracies))
    df = pd.DataFrame(data=acc_dict)
    print("Comparison count: " + str(len(accuracies)) + '/' + str(len(proofreads)))
    plot(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('proofread')
    parser.add_argument('automated')
    parser.add_argument('v1')
    parser.add_argument('voxel_sizes', nargs=3, type=float, help='Specify the width in um of each dimension of the voxel e.g. .4 .4 .4 for 15x')
    parser.add_argument('--globals', help='proofreads swcs in world/global um space. If passed diadem will be run')
    args = parser.parse_args()
    main(args)
