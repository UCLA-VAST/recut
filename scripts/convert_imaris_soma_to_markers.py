# convert proofread somas from imaris to recut marker files

from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from math import pi
from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace

# dir = Path(r'S:\Reconstructions_Manual\Other_Shared\For_MY\convert_recut_terafly_imaris\seeds_for_Imaris_proofread.swc')
# dir_IMS_proofread_seeds = dir.parent/'IMS_proofread_seeds'
# dir_IMS_proofread_seeds.mkdir(exist_ok=True)

def main(args: Namespace):

    dir = Path(args.soma)
    dir_IMS_proofread_seeds = dir.parent/str('IMS_proofread_recut_seeds_' + str(dir.name.replace('.swc','')))
    dir_IMS_proofread_seeds.mkdir(exist_ok=True)

    voxel_size_x = args.voxel_size_x
    voxel_size_y = args.voxel_size_y
    voxel_size_z = args.voxel_size_z

    radii = args.radii

    with open(dir, 'r') as conslidated_seeds:
        lines = conslidated_seeds.readlines()
        for l in lines:
            l_split = l.split(' ')
            x = int(float(l_split[2]) * voxel_size_x)
            y = int(float(l_split[3]) * voxel_size_y)
            z = int(float(l_split[4]) * voxel_size_z)
            # radii = float(l_split[5])
            volume = int(4/3*pi*radii**3)
            with open(dir_IMS_proofread_seeds/f"marker_{x}_{y}_{z}_{volume}", 'w') as marker_file:
                marker_file.write("# x,y,z,radius_um\n")
                marker_file.write(f"{x},{y},{z},{radii}")
            marker_file.close()
    conslidated_seeds.close()

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert Imaris proofread somas to recut seeds (marker files) \n\n",
    )
    parser.add_argument("--soma", "-s", type=str, required=True,
                        help="Path to imaris proofread soma file (.SWC)")
    parser.add_argument("--radii", "-r", type=float, required=False, default= 12,
                        help="Force the radii to be () um")
    parser.add_argument("--voxel_size_x", "-vx", type=float, required=False, default=0.4, 
                        help="voxel size x")
    parser.add_argument("--voxel_size_y", "-vy", type=float, required=False, default=0.4, 
                        help="voxel size y")
    parser.add_argument("--voxel_size_z", "-vz", type=float, required=False, default=0.4, 
                        help="voxel size z")

    main(parser.parse_args())




