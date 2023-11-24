from re import compile
from pathlib import Path
import os
from pandas import read_csv
import argparse
import numpy as np
from recut_interface import gather_swcs, match_coord_keys, extract_offset
import os
from shutil import rmtree, copyfile

def translate_and_scale(file_with_offset, voxel_sizes):
    # always create a fresh directory for proofread swcs
    # to prevent confusing state
    global_swc_dir = "global-swcs"
    if os.path.isdir(global_swc_dir):
        print("Overwriting " + global_swc_dir)
        rmtree(global_swc_dir)
    else:
        print("Writing " + global_swc_dir)
    os.mkdir(global_swc_dir)

    SWC_COLUMNS = ["id", "type", "x", "y", "z", "radius", "parent_id"]
    n=0
    regex = compile(r'\d+\.?\d*')

    for file, (x, y, z) in file_with_offset:
        swc_df = read_csv(file, sep=r"\s+", comment="#", names=SWC_COLUMNS)

        # translate from windowed (cropped) pixel space -> global image pixel space
        swc_df.x += x
        swc_df.y += y
        swc_df.z += z

        # scale from global pixel coordinates -> global um units
        swc_df.x *= voxel_sizes[0] 
        swc_df.y *= voxel_sizes[1]
        swc_df.z *= voxel_sizes[2]
        swc_df.radius *= voxel_sizes[0]
        
        if swc_df.loc[0].radius < 5:
            swc_df.at[0, 'radius'] = 5
        row = swc_df.loc[0]
        output_file = Path(global_swc_dir) / f'[{int(row.x):06n},{int(row.y):06n},{int(row.z):06n}]-r={int(row.radius):06n}.swc'
        if swc_df.drop(columns=['id', 'type', 'radius', 'parent_id']).duplicated().sum() > 0:
            n += 1
            print(f"duplicate nodes: {n} -> {output_file}")
        with open(output_file, 'a'):
            output_file.write_text("#")
            swc_df.to_csv(output_file, sep=" ", mode="a", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proofread')
    parser.add_argument('v1')
    args = parser.parse_args()
    kwargs = {}
    kwargs['voxel_size_x'] = .4
    kwargs['voxel_size_y'] = .4
    kwargs['voxel_size_z'] = .4
    voxel_sizes = np.array([kwargs['voxel_size_x'], kwargs['voxel_size_y'], kwargs['voxel_size_z']], dtype=float)
    distance_threshold = 30

    # for dealing with runs with mixed human proofreads and 
    # automated outputs you can pass this to keep original automateds
    f = lambda n: '_' not in os.path.basename(n)

    proofreads = gather_swcs(args.proofread, voxel_sizes)
    # you need to find the original automated outputs corresponding to the final proofreads
    # within the SWC file metadata of the automated outputs is the pixel based offset to the window
    v1 = gather_swcs(args.v1, voxel_sizes, f)
    print("Proofread count: " + str(len(proofreads)))
    print("Original automateds count: " + str(len(v1)))

    # match all data based off soma distance
    # keep only the matches whose automated succeeded (had more than the soma in the swc)
    matched_v1 = match_coord_keys(proofreads, v1, distance_threshold)
    file_with_offset = map(extract_offset, matched_v1)
    print("Match count: " + str(len(matched_v1)) + '/' + str(len(proofreads)))
    print("Only matches can be automatically placed in global (um) space")

    translate_and_scale(file_with_offset, voxel_sizes)

if __name__ == "__main__":
    main()
