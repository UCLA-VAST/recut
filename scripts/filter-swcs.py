import os
import sys

# python filter-swcs.py proofread_folder recut_folder out_folder

swc_types = ['recut', 'app2', 'proofreads']
out = sys.argv[3]
out_paths = []

for dirn in swc_types:
    full_dir = "%s/%s" % (out, dirn)
    print("Make dir: %s" % full_dir)
    out_paths.append(full_dir)

print(sys.argv[1])
proofreads = {}
for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        name = os.path.join(root, file)
        if '.swc' in file:
            coord = tuple([int(dim[1:]) for dim in file.split('.')[0].split('_')[-3:]])
            proofreads[coord] = name

recuts = {}
app2 = {}
markers = {}
print(sys.argv[2])
# valid_components = (f for f in os.scandir(sys.argv[2]) if f.is_dir() and ('multi' not in f.name))
valid_components = (f for f in os.scandir(sys.argv[2]) if f.is_dir())
for component in valid_components:
    for root, dirs, files in os.walk(component):
        #ignore a multi's since they don't have good app2s
        for file in files:
            name = os.path.join(root, file)
            if 'tree-with' in file:
                x, y, z = file.split('.')[0].split('-')[-3:]
                coord = (int(x), int(y), int(z))
                if coord in proofreads:
                    recuts[coord] = name
                    # for nb_file in os.scandir(root):
                        # if 'app2-' in nb_file:
                            # app2s[coord] = os.path
                # else:
                    # continue

path = '/home/kdmarrett/data/TME07/TME07-1_30x_Str_01A/marker_files'
# marker_entries = (f for f in os.scandir('/home/kdmarrett/data/TME07/TME07-1_30x_Str_01A/marker_files'))
# for marker in marker_entries:
    # print(marker)
for root, dirs, files in os.walk(path):
    for file in files:
        name = os.path.join(root, file)
        if 'marker_' in file:
            x, y, z, radius = file.split('_')[1:]
            coord = (int(x) - 1, int(y) - 1, int(z) - 1)
            markers[coord] = name
            # if coord in proofreads:
                # print(coord)
import pdb; pdb.set_trace()
