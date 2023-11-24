import argparse
from shutil import rmtree, copyfile
import os
from recut_interface import *
import numpy as np
from glob import glob
from math import pi

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
    dst = [proofread_marker_dir + '/' + unmatched_file.split('/')[-1] for unmatched_file in unmatched_files]
    _ = list(map(copyfile, unmatched_files, dst))

def create_matched(proofread_marker_dir, coords_radii):
    f2str = lambda f : "{:.2f}".format(f)
    for coord, radius in coords_radii:
        str_coord = list(map(f2str, coord))
        suff = '_'.join(str_coord) + '_' + f2str((4. / 3) * pi * (radius ** 3))
        # f name in world space (um)
        with open(proofread_marker_dir + '/' + "marker_" + suff, "w") as f:
            f.write("# x,y,z,radius_um\n")
            # example: 3306,1223,3267,12, everthing in world space (um)
            f.write(','.join(str_coord) + ',' + f2str(radius))

def get_proofread_soma_pos_rad(matched_v1, voxel_sizes):
    # extract offset from proofreads
    # returns a list of tuples where fst is proofread path and snd is offset coord
    v1_offset = list(map(extract_offset, matched_v1))

    # extract final (proofread) soma location and radius from proofreads
    proofread_soma = list(map(extract_soma_coord, matched_v1))
    proofread_soma_radius = map(extract_soma_radius, matched_v1)

    # offset + soma location 
    add_coord = lambda x, y : (x[0] + y[0], x[1] + y[1], x[2] + y[2])
    if (len(v1_offset) != len(proofread_soma)):
        raise Exception("v1 offsets and proofreads do not match")
    # TODO make sure the matches are aligned
    soma_coords_pixels = map(add_coord, map(lambda tup : tup[1], v1_offset),
                             proofread_soma)

    # scaled to voxel size for location and radius
    scale_coord = lambda coord: (coord[0] * voxel_sizes[0], coord[1] * voxel_sizes[1], coord[2] * voxel_sizes[2])
    final_coords = [scale_coord(c) for c in soma_coords_pixels]
    final_radii = [voxel_sizes[0] * r for r in proofread_soma_radius]
    # print(list(final_coords))
    return list(zip(final_coords, final_radii))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proofreads')
    parser.add_argument('markers')
    parser.add_argument('v1')
    args = parser.parse_args()

    # create a fresh directory for proofread markers
    proofread_marker_dir = "proofread-markers"
    if os.path.isdir(proofread_marker_dir):
        rmtree(proofread_marker_dir)
    os.mkdir(proofread_marker_dir)

    kwargs = {}
    kwargs['voxel_size_x'] = .4
    kwargs['voxel_size_y'] = .4
    kwargs['voxel_size_z'] = .4
    voxel_sizes = np.array([kwargs['voxel_size_x'], kwargs['voxel_size_y'], kwargs['voxel_size_z']], dtype=float)
    distance_threshold = 30

    # for dealing with runs with mixed human proofreads and 
    # automated outputs you can pass this to keep original automateds
    f = lambda n: '_' not in os.path.basename(n)

    # get a dict of coord (world space um) to proofread path
    proofread_dict = gather_swcs(args.proofreads, voxel_sizes)

    # get a list of marker paths
    markers = gather_markers(args.markers)
    # extract the coords from the paths
    coords = list(map(marker_to_coord, markers))
    # match those coords to each path so you can match with necessary offset below
    marker_dict = dict(zip(coords, markers))

    # you need to get v1 outputs to match to, get correct offset adjustment
    v1 = gather_swcs(args.v1, voxel_sizes, f)

    # list of (proofread path, v1 path):
    # note once you start matching you need to keep things as a list of tuples
    # so that ordering (which implies matching) is preserved
    # sets and dicts in python can be tricky to reason about in sorted order
    matched_v1 = match_coord_keys(proofread_dict, v1, distance_threshold)
    soma_coords_radii = get_proofread_soma_pos_rad(matched_v1, voxel_sizes)

    # pairs of proofread file path matched with marker file path
    matches = match_coord_keys(proofread_dict, marker_dict, distance_threshold)
    final_matched_proofreads = [proof for proof, _ in matches]

    # soma_coords_radii and matched_v1 originate from the same list so they share indices
    final_soma_coords_radii = [soma_coords_radii[i] for i, tup in enumerate(matched_v1) 
                                   if tup[0] in final_matched_proofreads] 
    final_proofreads = [proof for proof, _ in matched_v1 
                                   if proof in final_matched_proofreads] 
    double_matched_markers = {marker for proof in final_proofreads for proof2, marker in matches if proof2 == proof}
    print("final marker count: " + str(len(double_matched_markers)) + '/' + str(len(proofread_dict)))
    if (len(final_soma_coords_radii) != len(double_matched_markers)):
        raise Exception("if a marker cant be matched you need to at least guarantee reuse of the original")

    # split markers into those that match the final set of proofreads and those that do not
    marker_set = set(marker_dict.values())
    unmatched_markers = marker_set - double_matched_markers

    # for those that do not match, simply copy them over to the new directory
    copy_unmatched(proofread_marker_dir, unmatched_markers)

    # for matches update the radii from the proofread swc file
    create_matched(proofread_marker_dir, final_soma_coords_radii)
    print("Success, exiting...")

if __name__ == "__main__":
    main()
