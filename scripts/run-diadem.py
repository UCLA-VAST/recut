import argparse
import os, glob
from recut_interface import call_recut, valid

# 40x lens with 1.5 zoom
fgpct = .5
lab_args = {'OlfactoryProjectFibers' : [1, 1, 1]  }

# extract voxel size per lab
def extract_voxel_size(lab, paths, show=False):
    lab_voxel_size = 'invalid'
    if lab in lab_args: 
        lab_voxel_size = lab_args[lab]
    else:
        txt = list(filter(lambda path: os.path.isfile(path) and ("txt" in path), paths))
        if len(txt) > 0:
            temp = file_to_voxel_size(txt[0])
            if len(temp) == 3:
                lab_voxel_size = temp

    if show:
        if type(lab_voxel_size) == list:
            print('  ', end="") # offset
            print(lab_voxel_size)
        elif type(lab_voxel_size) == dict:
            print(lab_voxel_size)
    return lab_voxel_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    # recut arguments
    parser.add_argument('--fg-percent', type=float)
    args = parser.parse_args()
    show = False
    delete_run_dirs = True
    if delete_run_dirs:
        for d in glob.glob("run-*"):
            shutil.rmtree(d)

    for lab in os.listdir(args.dir):
        lab_path = os.path.join(args.dir, lab)
        if os.path.isfile(lab_path):
            continue;
        print(lab)
        image_path = os.path.join(lab_path, 'ImageStacks')
        subsubdirs = [image for image in os.listdir(image_path)]
        paths = list(map(lambda image: os.path.join(image_path, image), subsubdirs))

        # TODO match the voxel size to each image of the lab
        lab_voxel_size = extract_voxel_size(lab, paths)

        # TODO extract soma from each proofread swc
        # TODO write seed
        # TODO convert file format

        # extract individual images
        image_paths = list(filter(lambda dirp: "run" not in dirp, filter(lambda image: os.path.isdir(image), paths)))
        image_paths.sort()
        for image in image_paths:
            #TODO extract voxel size from lab's dict
            image_voxel_size = 'invalid'
            if type(lab_voxel_size) == dict:
                _, tail = os.path.split(image)
                image_voxel_size = lab_voxel_size[tail]
            elif type(lab_voxel_size) == list:
                image_voxel_size = lab_voxel_size
            else:
                for file in os.listdir(image):
                    if "txt" in file:
                        fpath = os.path.join(image, file)
                        image_voxel_size = file_to_voxel_size(fpath)
            if valid(image_voxel_size):
                if show:
                    print('    ', end="")
                    print(image_voxel_size)
                # build args
                args = { 'image': image, 'voxel_size_x' : image_voxel_size[0],
                        'voxel_size_y' : image_voxel_size[1], 'voxel_size_z' : image_voxel_size[2],
                        'seed_action' : 'find', 'preserve_topology' : '',
                        'fg_percent' : fgpct}
                call_recut(**args)

if __name__ == "__main__":
    main()
