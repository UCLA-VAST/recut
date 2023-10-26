import argparse
from recut_interface import gather_swc_files, filter_fails
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')

    args = parser.parse_args()
    kwargs = {}
    kwargs['voxel_size_x'] = .4
    kwargs['voxel_size_y'] = .4
    kwargs['voxel_size_z'] = .4
    voxel_sizes = np.array([kwargs['voxel_size_x'], kwargs['voxel_size_y'], kwargs['voxel_size_z']], dtype=float)

    all_swcs = gather_swc_files(args.run)
    substantials = filter_fails(zip(all_swcs, all_swcs))
    print("Substantial count: " + str(len(substantials)) + '/' + str(len(all_swcs)))

if __name__ == "__main__":
    main()
