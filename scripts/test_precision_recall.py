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

# define threshold for deciding close or not
# set as 1/4 of the chosen radius by default
# DISTANCE_THRESH = 0.25 * RADIUS
DISTANCE_THRESH = 0

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

### define a function to calculate euclidean distance between 1 inferenced point
### and all known ground truth points
def euc_distance(label, inference):
     
     """ label: marker files for ground truth;
         inference: marker files for inferenced soma'
         will return a list of tuples of the euclidean distances
         between each one of inferenced point and all known ground truth points"""
     
     # shortest distance to truth for each inferenced soma 
     euc_dist = []
     for soma_inf in inference: 
         # x_inf, y_inf, z_inf = [soma_inf[0], soma_inf[1], soma_inf[2]]
         # print("radius: ", soma_inf[3])
         soma_inf_xyz = np.asarray(soma_inf[:-1])
         # print("inferenced soma: ", soma_inf_xyz)
         each_dist = []
         for soma_lab in label:
            soma_lab_xyz = np.asarray(soma_lab[:-1])
            # print("labeled soma: ", soma_lab_xyz)
            dist = np.linalg.norm(soma_inf_xyz - soma_lab_xyz)
            each_dist.append(dist)
         euc_dist.append(tuple(each_dist))
     # print(euc_dist)  
     # print(len(euc_dist))      
     return euc_dist
            
                   
### define a function to decide match/non-match            
def is_match():
    
    """ return a list of tuples indicating match/non-match for each pair based on the threshold;
        1: match, 0: not-match"""
        
    distances = euc_distance(label, inference)
    print(type(distances))
    # print(distances)
    is_match = []
    for soma_inf in distances:
        is_match_each = []
        soma_inf_lst = list(soma_inf)
        is_match_each = [1 if d <= DISTANCE_THRESH else 0 for d in soma_inf_lst]
        is_match.append(tuple(is_match_each))
    return is_match
            

### define a function to calculate precision and recall
def precision_recall():
    
    """ calculate the precision and recall, and print them out"""
    
    match = is_match()
    TP, FP, FN = 0, 0, 0
    for soma_inf_match in match:
        soma_inf_match_lst = list(soma_inf_match)
        
        cnt_match = soma_inf_match_lst.count(1)
        # cnt_notmatch = soma_inf_match_lst.count(0)
        n_pairs = len(soma_inf_match_lst)
        
        # if there are at least one match
        if cnt_match >= 1:
            TP += 1
            # if there are more than 1 match
            if cnt_match > 1:
                FP += (cnt_match - 1)
        # if there are no match
        elif cnt_match == 0:
            FN += 1
    # print("TP:, ", TP, "FP: ", FP, "FN: ", FN)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("Precision: {}, Recall: {}, Total number of neurons: {}".format(precision, recall, n_pairs))
    
            
# precision_recall()


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
    
    # print("labeled soma locations: \n", labeled_markers)
    # print("inferenced soma locations: \m", inferenced_markers)
    
    # get precision & recall
    precision_recall()
    
