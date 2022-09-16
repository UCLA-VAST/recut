# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:29:39 2022

@author: YangRecon2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated script to digest CSV annotation file to generate marker files for recut
Created on Wed Aug 24 18:29:49 2022
@author: mingyan
"""
#### make sure the column name starts from row 3 in csv
#### make sure the x,y,z info is in the 1st, 2nd and 3rd column



from pathlib import Path
from pandas import read_csv
import pandas
import sys
import numpy as np

#####################
# file paths
annotations = Path(sys.argv[1])
recut = annotations.parent / 'soma_recut'
recut.mkdir(exist_ok=True)


# specs (radius)
SOMA_SIZE = 12

# volume of the sphere
SOMA_VOLUME = int(4 * np.pi * (SOMA_SIZE**3) / 3)

# # variable name for x,y,z coordinates
# X_name = 'Pt Position X'
# Y_name = 'Pt Position Y'
# Z_name = 'Pt Position Z'

# digest CSV (accommodate for CSV that has redundant rows at the beginning and only keep relevant columns)
annot_dict = {}
num = 0

# grab relevant data
f = open(annotations, 'r')
data = f.read()
# print(data)
for i, line in enumerate(data.splitlines()[3:]):
    line_sep = line.split(',')
    # print(line_sep)
    
    # if len(line_sep) > 6 and X_name in line_sep:
    #     X_ind = line_sep.index(X_name)
    #     Y_ind = line_sep.index(Y_name)
    #     Z_ind = line_sep.index(Z_name)
    #     print("INDEX: ", X_ind, Y_ind, Z_ind)
    # grab data for relevant columns        
    # elif len(line_sep) > 6 and X_name not in line_sep:
    #     print("NEW INDEX: ", X_ind, Y_ind, Z_ind)
        # print("ANOTHER INDEX: ", X_ind, Y_ind, Z_ind)
        # annot_dict[num] = {}
        # annot_dict[num]['x'] = int(round(float(line_sep[X_ind]),0))
        # annot_dict[num]['y'] = int(round(float(line_sep[Y_ind]),0))
        # annot_dict[num]['z'] = int(round(float(line_sep[Z_ind]),0))
        # num += 1   
    annot_dict[num] = {}
    annot_dict[num]['x'] = int(round(float(line_sep[0]),0))
    annot_dict[num]['y'] = int(round(float(line_sep[1]),0))
    annot_dict[num]['z'] = int(round(float(line_sep[2]),0))
    num += 1      
f.close()

# print("dict: ",annot_dict)

# convert to df
annotations_df = pandas.DataFrame.from_dict(annot_dict, orient='index')


# discard rows ith negative 
annotations_df_filtered = annotations_df[annotations_df['x'] >= 0]
annotations_df_filtered = annotations_df[annotations_df['y'] >= 0]
annotations_df_filtered = annotations_df[annotations_df['z'] >= 0]

# write individual marker files
for row in annotations_df_filtered.itertuples():
    with open(recut/f"marker_{row.x}_{row.y}_{row.z}_{SOMA_VOLUME}", 'w') as soma_file:
        soma_file.write("# x,y,z,radius\n")
        soma_file.write(f"{row.x},{row.y},{row.z},{SOMA_SIZE}")