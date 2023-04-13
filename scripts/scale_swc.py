# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:12:11 2022

@author: yanyanming77

To scale the SWC reconstructed from 30x image and saved in neutube (in voxel unit),
to be comparable to the scale of lightsheet reconstructions (in um unit)

scaling factor: 
    x: 0.201
    y: 0.201
    z: 0.998
    radius: 0.201
"""


import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

def scaling(input_path, scale_x = 0.201, scale_y = 0.201, scale_z = 0.998, scale_radii = 0.201):
    """
    Parameters
    ----------
    input_path : path
        input path for swcs to be scaled
    scale_x : number, optional
        The default is 0.201.
    scale_y : number, optional
        The default is 0.201.
    scale_z : number, optional
        The default is 0.998.
    scale_radii : number, optional
        The default is 0.201.

    Returns
    -------
    None.

    """
    output_path = input_path.parent/f"{input_path.name}_scaled"
    output_path.mkdir(exist_ok=True)
    
    for file in tqdm(list(input_path.rglob("*.swc")), desc = "reading input files"):
       # comment line
       # note that the number of comment lines may be different in proofread and junk SWCs
       comment_lines = [l for l in open(file,'r').readlines() if '#' in l]
       num_comment_lines = len(comment_lines)
       # read data
       df = pd.read_csv(file, delimiter=' ', skiprows=num_comment_lines, names = ['node_num1', 'type', 'x', 'y', 'z', 'radii', 'node_num2'])
       
       # scaling
       df['x'] = df['x'] * scale_x
       df['x'] = df['x'].round(decimals=3)
       df['y'] = df['y'] * scale_y
       df['y'] = df['y'].round(decimals=3)
       df['z'] = df['z'] * scale_z
       df['z'] = df['z'].round(decimals=3)
       df['radii'] = df['radii'] * scale_radii
       df['radii'] = df['radii'].round(decimals=5)
       
       with open(output_path/f"{file.name}",'w') as scaled_file:
           for cmt_line in comment_lines:
               scaled_file.write(cmt_line)
           for row in df.itertuples():
               scaled_file.write(f"{row.node_num1} {row.type} {row.x} {row.y} {row.z} {row.radii} {row.node_num2}\n")
       scaled_file.close()
       
       
# input_path = Path(r'C:\Users\yanyanming77\Desktop\precision_recall\TMD\proofread_swc')
       
if __name__ == '__main__':
    
    parser = ArgumentParser(description="SWC scaling")
    # if you need to train model first
    parser.add_argument("--input", "-i", type=str, required=True, help="path to input folder")
    args = parser.parse_args()

    input_path = Path(rf"{args.input}")
    scaling(input_path)
       
       
       
       
       
       
       
       