import os
import sys

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
            # import pdb; pdb.set_trace()

recuts = {}
app2 = {}
print(sys.argv[2])
valid_components = (f for f in os.scandir(sys.argv[1]) if f.is_dir() and ('multi' not in f.name))
for component in valid_components:
    for root, dirs, files in os.walk(component):
        #ignore a multi's since they don't have good app2s
        for file in files:
            name = os.path.join(root, file)
            if 'tree-with' in file:
                x, y, z = file.split('-')[-3:]
                coord = (x,y,z)
                if coord in proofreads:
                    recuts[coord] = name
                    print(name)
                    print(root)
                    for nb_file in os.scandir(root):
                        if 'app2-' in nb_file:
                            app2s[coord] = os.path
                else:
                    continue


            # if 'marker_' in file:
                # for x, y, z, radius in file.split('_')[1:]:
                    # if tuple(x,y,z) in proofreads:
                        # copy n
               # import pdb; pdb.set_trace()
