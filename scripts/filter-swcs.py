import os
import sys
import shutil
import pathlib

def gather_proofreads(proofread_dir):
    # build set of sections from proofread folder
    # sections = [f.name for f in os.scandir(proofread_dir) if f.is_dir()]
    proofreads = {}
    for root, dirs, files in os.walk(proofread_dir):
        for file in files:
            name = os.path.join(root, file)
            if '.swc' in file:
                coord = tuple([int(dim[1:]) for dim in file.split('.')[0].split('_')[-3:]])
                proofreads[coord] = name
    return proofreads


def gather_reconstructions(recut_dir):
    recuts = {}
    app2 = {}
    recut_sections = [f.name for f in os.scandir(recut_dir) if f.is_dir() and ('TME07-1_30x_Str' in f.name)]
    for recut_section in recut_sections:
        section_dir = recut_dir + '/' + recut_section + '/run-1'
        #ignore a multi's since they don't have partitioned app2s
        # valid_components = (f for f in os.scandir(section_dir) if f.is_dir() and ('multi' not in f.name))
        valid_components = (f for f in os.scandir(section_dir) if f.is_dir())
        for component in valid_components:
            for root, dirs, files in os.walk(component):
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
    return recuts, app2

# python filter-swcs.py proofread_folder recut_folder out_folder
proofread_dir = sys.argv[1]
recut_dir = sys.argv[2]
out_dir = sys.argv[3]

proofreads = gather_proofreads(proofread_dir)
print('Proofread count: %d' % len(proofreads))
recuts, app2 = gather_reconstructions(recut_dir)
print('Recut count: %d' % len(recuts))
print('APP2 count: %d' % len(app2))

# Out
swc_types = ['proofread', 'recut', 'app2']
out_paths = []

for dirn in swc_types:
    full_dir = "%s/%s" % (out_dir, dirn)
    out_paths.append(full_dir)

for out_path, dic in zip(out_paths, [proofreads, recuts, app2]):
    print("Make dir: %s" % out_path)
    # os.makedir(full_dir)
    os.makedirs(out_path, exist_ok=True)
    # path = pathlib.Path(out_path)
    # path.parent.mkdir(parents=True, exist_ok=True)
    for fn in dic.values():
        shutil.copy2(fn, out_path)
