#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated script to digest CSV annotation file to generate marker files for recut

Created on Wed Aug 24 18:29:49 2022

@author: mingyan
"""

from pathlib import Path
from pandas import read_csv
import pandas
import sys


#####################
# file paths
annotations = Path(sys.argv[1])
recut = annotations.parent / 'soma_recut'
recut.mkdir(exist_ok=True)

# specs
SOMA_SIZE = 12

# variable name for x,y,z coordinates
X_name = 'Pt Position X'
Y_name = 'Pt Position Y'
Z_name = 'Pt Position Z'

# digest CSV (accommodate for CSV that has redundant rows at the beginning and only keep relevant columns)
annot_dict = {}
num = 0

# grab relevant data
f = open(annotations, 'r')
data = f.read()
for i, line in enumerate(data.splitlines()):
    line_sep = line.split(',')
    # grab indices for relevant columns
    if len(line_sep) > 10 and X_name in line_sep:
        X_ind = line_sep.index(X_name)
        Y_ind = line_sep.index(Y_name)
        Z_ind = line_sep.index(Z_name)
    # grab data for relevant columns        
    elif len(line_sep) > 10 and X_name not in line_sep:
        annot_dict[num] = {}
        annot_dict[num]['x'] = int(round(float(line_sep[X_ind]),0))
        annot_dict[num]['y'] = int(round(float(line_sep[Y_ind]),0))
        annot_dict[num]['z'] = int(round(float(line_sep[Z_ind]),0))
        num += 1              
f.close()

# convert to df
annotations_df = pandas.DataFrame.from_dict(annot_dict, orient='index')

# discard rows ith negative 
annotations_df_filtered = annotations_df[annotations_df['x'] >= 0]
annotations_df_filtered = annotations_df[annotations_df['y'] >= 0]
annotations_df_filtered = annotations_df[annotations_df['z'] >= 0]

# write individual marker files
for row in annotations_df_filtered.itertuples():
    with open(recut/f"marker_{row.x}_{row.y}_{row.z}", 'w') as soma_file:
        soma_file.write("# x,y,z\n")
        soma_file.write(f"{row.x},{row.y},{row.z}")
