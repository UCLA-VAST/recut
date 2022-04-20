import os
import sys
import shutil
import pathlib

# python filter-swcs.py proofread_folder recut_folder out_folder

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
valid_components = (f for f in os.scandir(sys.argv[2]) if f.is_dir() and ('multi' not in f.name))
# valid_components = (f for f in os.scandir(sys.argv[2]) if f.is_dir())
for component in valid_components:
    for root, dirs, files in os.walk(component):
        #ignore a multi's since they don't have good app2s
        for file in files:
            name = os.path.join(root, file)
            if 'tree-with' in file:
                x, y, z = file.split('.')[0].split('-')[-3:]
                coord = (int(x), int(y), int(z))
                if coord in proofreads:
                    if 'app2-' in file:
                        app2[coord] = name
                    else:
                        recuts[coord] = name
                    # for nb_file in os.scandir(root):
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

print('Proofread count: %d' % len(proofreads))
print('Recut count: %d' % len(recuts))
print('APP2 count: %d' % len(app2))

# Out
swc_types = ['proofread', 'recut', 'app2']
out = sys.argv[3]
out_paths = []

for dirn in swc_types:
    full_dir = "%s/%s" % (out, dirn)
    out_paths.append(full_dir)

for out_path, dic in zip(out_paths, [proofreads, recuts, app2]):
    print("Make dir: %s" % out_path)
    # os.makedir(full_dir)
    os.makedirs(out_path, exist_ok=True)
    # path = pathlib.Path(out_path)
    # path.parent.mkdir(parents=True, exist_ok=True)
    for fn in dic.values():
        shutil.copy2(fn, out_path)

# import pdb; pdb.set_trace()
