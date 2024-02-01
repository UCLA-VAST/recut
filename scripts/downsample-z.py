import argparse
from glob import glob
from shutil import copy
from os import makedirs

def downsample(args):
    out = str(args.input)[:-1] + "_downsample" + str(args.factor)
    print(out)
    pattern = args.input + "/*.tif"
    tifs = sorted(glob(pattern, recursive=False))[0::args.factor]
    makedirs(out, exist_ok=True)
    for i, src in enumerate(tifs):
        dst = out + "/img_" + str(i).zfill(6) + ".tif"
        print(src, ' -> ', dst)
        copy(src, dst) 
    print("Remember to multiply the z voxel size by " + str(args.factor) + " when inputting to reconstruction")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Folder of tif files')
    parser.add_argument('-f', '--factor', help='downsample factor', default=2, type=int)
    args = parser.parse_args()
    downsample(args)
