#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for calculating precision and recall for soma detection model
Created on Tue Aug  9 15:32:05 2022
@author: mingyan
"""

import sys
import os
import argparse
import numpy as np

### define a function to retrieve xyz and radius
def gather_markers(path):
    markers = []
    adjust = 0
    for root, dir, files in os.walk(path):
        # print("root: ",root)
        # print("dir: ",dir)
        # print("files: ",files)
        for file in files:
            name = os.path.join(root, file)
            if 'marker_' in file:
                x, y, z, volume = [int(i) for i in file.split('_')[1:]] 
                # print(x,y,z,volume)
                radius = np.cbrt(volume * 3 / (4 * np.pi))
                coord = (x - adjust, y - adjust, z - adjust)
                markers.append((coord[0], coord[1], coord[2], radius))
    return markers



# label = gather_markers("/Users/mingyan/Desktop/UCLA/Recut/recut-repo-new/marker_files")
# inference = gather_markers("/Users/mingyan/Desktop/UCLA/Recut/recut-repo-new/marker_files")

### definea function to calculate euclidean distance
def euc_distance(label, inference):
     
     """ label: marker files for ground truth;
         inference: marker files for inferenced soma"""
     
     # shortest distance to truth for each inferenced soma 
     euc_dist = []
     for soma_inf in inference[:5]: 
         # x_inf, y_inf, z_inf = [soma_inf[0], soma_inf[1], soma_inf[2]]
         # print("radius: ", soma_inf[3])
         soma_inf_xyz = np.asarray(soma_inf[:-1])
         # print("inferenced soma: ", soma_inf_xyz)
         shortest = []
         for soma_lab in label[:5]:
            soma_lab_xyz = np.asarray(soma_lab[:-1])
            # print("labeled soma: ", soma_lab_xyz)
            dist = np.linalg.norm(soma_inf_xyz - soma_lab_xyz)
            # print("distance: ", dist)
            if len(shortest) == 0:
                shortest.append(dist)
            elif len(shortest) == 1:
                if dist < shortest[0]:
                    shortest[0] = dist
            # print("shortest: ", shortest[0])
         euc_dist.append(shortest[0])
     print(euc_dist)        
            
                    
            
# euc_distance(label, inference)



# define threshold for deciding close or not
# set as 1/4 of the chosen radius by default
DISTANCE_THRESH = 0.25 * RADIUS

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('labeled_path', help = "input path for labeled soma location")
    parser.add_argument('inferenced_path', help = "input path for inferenced soma location")
    args = parser.parse_args()
    
    labeled_path = args.labeled_path
    inferenced_path = args.inferenced_path
    
    # gather markers
    labeled_markers = gather_markers(labeled_path)
    inferenced_markers = gather_markers(inferenced_path)
    
    print("labeled soma locations: \n", labeled_markers)
    print("inferenced soma locations: \m", inferenced_markers)
    