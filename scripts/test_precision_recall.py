
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for calculating precision and recall for soma detection model
Last updated on Tue Sep 13
@author: mingyan
"""

import sys
import os
import argparse
import numpy as np
from collections import Counter
from pathlib import Path
from datetime import datetime

# define threshold for deciding close or not
# set as 1/4 of the chosen radius by default

# RADIUS = 25.6 / 2
# DISTANCE_THRESH = 0.25 * RADIUS * 2

DISTANCE_THRESH = 25.6 / 2
print("distance threshold: ", DISTANCE_THRESH)
# DISTANCE_THRESH = 0


### define a function to retrieve xyz and radius
def gather_markers(path):
    markers = []
    adjust = 0
    for root, dir, files in os.walk(path):
        # print("root: ",root)
        # print("dir: ",dir)
        # print("files: ",files)
        for file in files:
            file = file.replace('.000000','')
            file = file.replace('.txt', '')
            name = os.path.join(root, file)
            if 'marker_' in file:
                x, y, z, volume = [int(i) for i in file.split('_')[1:]] 
                # print(x,y,z,volume)
                # radius = np.cbrt(volume * 3 / (4 * np.pi))
                coord = (x - adjust, y - adjust, z - adjust)
                # markers.append((coord[0], coord[1], coord[2], radius))
                markers.append((coord[0], coord[1], coord[2]))
    return markers



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
        
    euc_dist = euc_distance(label, inference)
    ###
    # print("distance to each truth: ", distances)
    # print(type(distances))
    # print(distances)
    is_match = []
    for soma_inf in euc_dist:
        is_match_each = []
        soma_inf_lst = list(soma_inf)
        is_match_each = [1 if d <= DISTANCE_THRESH else 0 for d in soma_inf_lst]
        is_match.append(tuple(is_match_each))
    return is_match
   
    
### define a function to calculate precision and recall
def precision_recall():
    
    """ calculate the precision and recall, and print them out"""

    match = is_match() # all matches and unmatches
    euc_dist = euc_distance(label, inference)
    
    TP, FP, FN = 0, 0, 0
        
    # if two identical macthes, only remain the one with the lowest Euclidean distance as the match
    # and set others to 0 (non-match)
    euc_dist = [list(ele) for ele in euc_dist] # temporarily convert to list of lists
    match = [list(ele) for ele in match]
        
    ### capture the case of multiple inferences that have maches to the same soma label
    ### only keep the match with closest Euc dist as real match, set others to non-match
    inx1_all = [i for i in range(0,len(label))]
    for inx1 in inx1_all:
        # print("inx1: ", inx1)
        minval = min([euc_dist[i][inx1] for i,ele in enumerate(euc_dist)])
        # print("minval: ", minval)
        for i,ele in enumerate(euc_dist):
            if euc_dist[i][inx1] > minval:
                match[i][inx1] = 0
 
  
    ### then capture the case of same inference has matches to multiple somas
    ### after considering the above case
    for i,m in enumerate(match):
        cnt_match = m.count(1)
        # if there are multiple matches with soma inside one inference
        if cnt_match > 1:
            # calculate the distance and only set the closest match as match
            indx1s = [i for i,ele in enumerate(m) if ele == 1]
            minval = min([euc_dist[i][indx1] for indx1 in indx1s])
            for indx1 in indx1s:
                if euc_dist[i][indx1] > minval:
                    match[i][indx1] = 0        

    euc_dist = [tuple(ele) for ele in euc_dist] # convert back to list of tuples
    match = [tuple(ele) for ele in match]



    ### validate for each element in match, there is as many as one '1'
    sanity_list = []
    for i,m in enumerate(match):
        cnt_match = m.count(1)
        sanity_list.append(cnt_match)
    if tuple(set(sanity_list)) == (0,1):
        print("Sanity check is good")
  

  
    # true_match_list = [] # placeholder for all matches
    inference_coord_list = []
    label_coord_list = []
    closest_dist_list = []
    
    # loop again to generate outputs, now each inference should have as many as 1 soma match
    # and one soma should only match to 1 closest inference
    for i,soma_inf_match in enumerate(match):
        soma_inf_match_lst = list(soma_inf_match)
        
        cnt_match = soma_inf_match_lst.count(1) # number of matches according to threshold in for each inference
        
        if cnt_match >=1:
            # get index of inference that has a match
            ind_inf = match.index(soma_inf_match)
            
            # get index of matched label in each inference
            ind_lab = soma_inf_match_lst.index(1)
            
            # get the euc dist for the match
            closest_dist = euc_dist[i][ind_lab]
            
            # print("index in inference {}, index in label {}".format(ind_inf, ind_lab))
            # print("match:\ninference: {}, label: {}, shortest dist: {}".format(inference[ind_inf], label[ind_lab], closest_dist))
            
            inference_coord_list.append(inference[ind_inf])
            label_coord_list.append(label[ind_lab])
            closest_dist_list.append(closest_dist)
            
    
    n_pairs = len(label) # number of true labels
    
    # TP = len(Counter(true_match_list).keys()) # TPs: non-duplicate matches
    TP = len(inference_coord_list)
    TP_list = inference_coord_list

    
    # print out all FNs (somas in the label but not detected)
    FN_list = []
    for e in label:
        if e not in label_coord_list:
            FN_list.append(e)
    

    # print out all FPs (somas in the inference but are not true labels)
    # and their closest distance to labeled somas
    FP_list = []
    FP_closest_dist_list = []
    # FP_all_dist_list = []
    for i,e in enumerate(inference):
        if e not in inference_coord_list:
            FP_list.append(e)
            min_dist = min(euc_dist[i])
            FP_closest_dist_list.append(min_dist)
            # FP_all_dist_list.append(euc_dist[i])
    
    

    FP = len(inference) - TP
    FN = len(label) - TP
    print("TP: ", TP, "FP: ", FP, "FN: ", FN)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("Precision: {}, Recall: {},\nTotal number of true labels: {}, Total number of inferences: {}".format(precision, recall, n_pairs, len(match)))


    # save matched pairs and all information of this run to a txt file
    time_now = datetime.now()
    current_time = time_now.strftime("%H:%M:%S").replace(':', '_')
    # print(current_time)
    with open(path/f"precision_recall_{current_time}.txt", 'w') as f:
        f.write(f"Distance threshold: {DISTANCE_THRESH}\n")
        # f.write(f"Number of duplicate match of true labels: {num_dup_match}\n")
        f.write(f"TP: {TP}, FP: {FP}, FN: {FN}\n")
        f.write(f"Precision: {precision}, Recall: {recall}\n")
        f.write(f"Total number of true labels: {n_pairs}, Total number of inferences: {len(match)}\n")
        f.write("\nAll matched pairs: \n")
        for inf, lab, dist in zip(inference_coord_list, label_coord_list, closest_dist_list):
            f.write(f"inference: {inf}, label: {lab}, closest dist: {dist}\n")
        f.write("\n")
        f.write("FPs (among inferenced markers): \n")
        for inf_fp,dist in zip(FP_list, FP_closest_dist_list):
            f.write(f"inference: {inf_fp}, closest distance: {dist}\n")
        # f.write(f"{FP_list}")
        f.write("\n")
        f.write("FNs (among true markers): \n")
        for lab_fn in FN_list:
            f.write(f"lab: {lab_fn}\n")



if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('labeled_path', help = "input path for labeled soma location")
    parser.add_argument('inferenced_path', help = "input path for inferenced soma location")
    args = parser.parse_args()
    
    labeled_path = args.labeled_path
    inferenced_path = args.inferenced_path
    
    # gather markers
    label = gather_markers(labeled_path)
    inference = gather_markers(inferenced_path)
    
    path = Path(os. path. realpath(__file__)).parent
     
    
    # get precision & recall
    precision_recall()
    



               

    
    




