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



# markers = gather_markers("/Users/mingyan/Desktop/UCLA/Recut/recut-repo-new/marker_files")
# print(markers)

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
    