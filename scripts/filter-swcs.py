import os
import sys

print(sys.argv[1])
final_soma_coords = set()
for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        name = os.path.join(root, file)
        if '.swc' in file:
            coord = tuple([int(dim[1:]) for dim in file.split('.')[0].split('_')[-3:]])
            set.add(coord)
            # import pdb; pdb.set_trace()

print(sys.argv[2])
for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        name = os.path.join(root, file)
        if 'marker_' in file:
            for x, y, z, radius in file.split('_')[1:]:
                if tuple(x,y,z) in final_soma_coords:
                    # copy n
           # import pdb; pdb.set_trace()
