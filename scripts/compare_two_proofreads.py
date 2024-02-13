import argparse
from recut_interface import *

distance_threshold = 40

def main(args):
    voxel_sizes = tuple(args.voxel_sizes)

    # for dealing with runs with mixed human proofreads and 
    # automated outputs you can pass this to keep original automateds
    f = lambda n: '_' not in os.path.basename(n)

    # converts all swcs to world-space um
    global_proofreads = gather_swcs(args.globals, voxel_sizes)
    proofreads = gather_swcs(args.proofread, voxel_sizes)
    v1 = gather_swcs(args.v1, voxel_sizes, f)

    print("Global proofread count: " + str(len(global_proofreads)))
    print("Proofread count: " + str(len(proofreads)))
    print("Original automateds count: " + str(len(v1)))

    # match all data based off soma distance
    # create a mapping between (windowed) proofreads with proofreads in global space
    # matched = match_coord_keys(proofreads, global_proofreads, distance_threshold)
    # matched = match_coord_keys(global_proofreads, v1, distance_threshold)
    matched = match_coord_keys(proofreads, global_proofreads, distance_threshold)
    matched_dict = dict(matched)
    # import pdb; pdb.set_trace()

    # create a mapping between (windowed) proofreads with their original automated output
    matched_v1 = match_coord_keys(proofreads, v1, distance_threshold)

    v1_offset = dict(list(map(extract_offset, matched_v1)))
    params = [(key, matched_dict[key], v1_offset[key]) for key in set(matched_dict) & set(v1_offset)]
    print("Match count: " + str(len(matched)) + '/' + str(len(proofreads)))
    print("Match v1 count: " + str(len(matched_v1)) + '/' + str(len(proofreads)))
    print("Params count: " + str(len(params)) + '/' + str(len(proofreads)))
    # import pdb; pdb.set_trace()

    # you can't run diadem unless you translate the SWC files

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
    parser.add_argument('globals')
    parser.add_argument('proofread')
    parser.add_argument('v1')
    parser.add_argument('voxel_sizes', nargs=3, type=float, help='Specify the width in um of each dimension of the voxel e.g. .4 .4 .4 for 15x')
    args = parser.parse_args()
    main(args)
