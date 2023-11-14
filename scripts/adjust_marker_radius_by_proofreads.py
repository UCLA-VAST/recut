import argparse
from shutil import rmtree, copyfile
from os import mkdir
from os import path
from recut_interface import *
import numpy as np
from glob import glob
from more_itertools import partition

# takes a directory path returns list of coord+radius
# markers are always in world space (um units)
def gather_markers(path):
    pattern = path + "/**/marker*"
    return glob(pattern, recursive=True)

def marker_to_coord(file):
    return tuple(map(float, file.split('/')[-1].split('_')[1:-1]))

# takes the directory of the new proofread marker folder and a list
# of all unmatched markers and simply copies them over, unmodified
def copy_unmatched(proofread_marker_dir, unmatched_files):
    dst = [proofread_marker_dir + unmatched_files.split('/')[-1] for unmatched_file in unmatched_files]
    _ = list(map(copyfile, unmatched_files, dst))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proofreads')
    parser.add_argument('markers')
    parser.add_argument('v1')
    args = parser.parse_args()

    # create a fresh directory for proofread markers
    proofread_marker_dir = "proofread-markers"
    if path.isdir(proofread_marker_dir):
        rmtree(proofread_marker_dir)
    mkdir(proofread_marker_dir)

    kwargs = {}
    kwargs['voxel_size_x'] = .4
    kwargs['voxel_size_y'] = .4
    kwargs['voxel_size_z'] = .4
    voxel_sizes = np.array([kwargs['voxel_size_x'], kwargs['voxel_size_y'], kwargs['voxel_size_z']], dtype=float)
    distance_threshold = 30

    # for dealing with runs with mixed human proofreads and 
    # automated outputs you can pass this to keep original automateds
    f = lambda n: '_' not in path.basename(n)

    # get a dict of coord (world space um) to proofread path
    proofread_dict = gather_swcs(args.proofreads, voxel_sizes)

    # get a dict of coord (world space um) to marker path
    markers = gather_markers(args.markers)
    coords = list(map(marker_to_coord, markers))
    marker_dict = dict(zip(coords, markers))

    # you need to get v1 outputs to match to, get correct offset adjustment
    v1 = gather_swcs(args.v1, voxel_sizes, f)
    matched_v1 = match_coord_keys(proofread_dict, v1, distance_threshold)
    print("Match v1 count: " + str(len(matched_v1)) + '/' + str(len(proofread_dict)))

    # extract offset from proofreads
    v1_offset = list(map(extract_offset, matched_v1))

    # extract final (proofread) soma location and radius from proofreads
    proofread_soma = list(map(extract_soma_coord, matched_v1))
    proofread_soma_radius = list(map(extract_soma_coord_radius, matched_v1))

    # offset + soma location 
    add_coord = lambda l, r: l[0] + r[0], l[1] + r[1], l[2] + r[2]
    # TODO make sure the matches are aligned
    soma_coords_pixels = map(add_coord, v1_offset, proofread_soma)

    # scaled to voxel size for location and radius
    scale_coord = lambda coord: coord[0] * voxel_size[0], coord[1] * voxel_size[1], coord[2] * voxel_size[2]
    map(scale_coord, soma_coords_pixels)

    # pairs of proofread file path matched with marker file path
    matches = match_coord_keys(proofread_dict, marker_dict, distance_threshold)
    print("match count: " + str(len(matches)) + '/' + str(len(proofread_dict)))
    import pdb; pdb.set_trace()

    # is_matched = lambda marker_pair: euc_dist(marker_pair 

    # split markers into those that match the final set of proofreads and those that do not
    # unmatched, matched_markers = match_markers(markers, proofreads)
    unmatched, matched = partition(is_matched, markers)
    print("Marker-proof match count: " + str(len(matched_markers)) + '/' + str(len(markers)))

    # for those that do not match, simply copy them over to the new directory
    copy_unmatched(proofread_marker_dir, unmatched)

    # for matches update the radii from the proofread swc file

if __name__ == "__main__":
    main()
