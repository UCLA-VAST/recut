# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for calculating precision and recall for soma detection model
Last updated on Tue Sep 13
@author: Ming Yan
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from datetime import datetime
from numpy import ndarray, array
from numpy.linalg import norm
from typing import List, Tuple

# define threshold for deciding close or not
# set as 1/4 of the chosen radius by default

# RADIUS = 25.6 / 2
# DISTANCE_THRESH = 0.25 * RADIUS * 2

DISTANCE_THRESH = 12.8
print("distance threshold: ", DISTANCE_THRESH)
# DISTANCE_THRESH = 0

# radii of the label (um)
RADII = DISTANCE_THRESH

time_now = datetime.now()
current_time = time_now.strftime("%H:%M:%S").replace(':', '_')


def gather_markers(seeds_path):
    """a function to retrieve xyz and radius"""
    markers = []
    markers_with_radii = []
    adjust = 0
    for root, dir_, files in os.walk(seeds_path):
        # print("root: ", root)
        # print("dir: ", dir_)
        # print("files: ", files)
        for file in files:
            file = file.replace('.000000', '').replace('.txt', '')
            # name = os.path.join(root, file)
            if 'marker_' in file:
                x, y, z, volume = [int(i) for i in file.split('_')[1:]]
                # print(x,y,z,volume)
                radius = np.cbrt(volume * 3 / (4 * np.pi))
                coord = (x - adjust, y - adjust, z - adjust)
                markers.append((coord[0], coord[1], coord[2]))
                markers_with_radii.append((coord[0], coord[1], coord[2], radius))
    return markers, markers_with_radii


def euc_distance(soma_labels: list, soma_inferences: list, voxel_sizes: ndarray = array([1, 1, 1], dtype=float)):
    """
    a function to calculate euclidean distance between inferences and known ground truth points

    soma_labels: list
        marker files for ground truth

    soma_inferences: list
        marker files for soma inferences

    voxel_sizes: numpy array
        voxel sizes in x, y, z order

    -------

    returns
        a list of tuples of the euclidean distances between each one of inferences and all known ground truth points
    """

    # shortest distance to truth for each inferenced soma
    euc_dist = []
    soma_inferences_xyz = array(soma_inferences)
    soma_labels_xyz = array(soma_labels)
    for soma_inf_xyz in soma_inferences_xyz:
        each_dist = tuple((norm((soma_inf_xyz - soma_lab_xyz) * voxel_sizes) for soma_lab_xyz in soma_labels_xyz))
        euc_dist.append(each_dist)

    # print(euc_dist)
    # print(len(euc_dist))
    return euc_dist


def is_match(euc_dist: List[Tuple[int, ...]]):
    """
    a function to decide match/non-match

    euc_dist: List[Tuple[int, ...]]
        The list members corresponds to each element in the inference list
        The tuple members corresponds to each element in the labels list
    ---

    return
        a list of tuples indicating match/non-match for each pair based on the threshold;
        1: match, 0: not-match
    """

    # euc_dist = euc_distance(label, inference)
    # print("distance to each truth: ", distances)
    # print(type(distances))
    # print(distances)
    match_list = []
    for soma_inf in euc_dist:
        soma_inf_lst = list(soma_inf)
        is_match_each = [1 if d <= DISTANCE_THRESH else 0 for d in soma_inf_lst]
        match_list.append(tuple(is_match_each))
    return match_list


def precision_recall():
    """
    calculate the precision and recall, and print them out

    """
    euc_dist = euc_distance(label, inference)
    match = is_match(euc_dist)  # all matches and unmatches

    TP, FP, FN = 0, 0, 0

    # if two identical macthes, only remain the one with the lowest Euclidean distance as the match
    # and set others to 0 (non-match)
    euc_dist = [list(ele) for ele in euc_dist]  # temporarily convert to list of lists
    match = [list(ele) for ele in match]

    ### capture the case of multiple inferences that have maches to the same soma label
    ### only keep the match with closest Euc dist as real match, set others to non-match
    inx1_all = [i for i in range(0, len(label))]
    for inx1 in inx1_all:
        # print("inx1: ", inx1)
        minval = min([euc_dist[i][inx1] for i, ele in enumerate(euc_dist)])
        # print("minval: ", minval)
        for i, ele in enumerate(euc_dist):
            if euc_dist[i][inx1] > minval:
                match[i][inx1] = 0

    ### then capture the case of same inference has matches to multiple somas
    for i, m in enumerate(match):
        cnt_match = m.count(1)
        # if there are multiple matches with soma inside one inference
        if cnt_match > 1:
            # calculate the distance and only set the closest match as match
            indx1s = [i for i, ele in enumerate(m) if ele == 1]
            minval = min([euc_dist[i][indx1] for indx1 in indx1s])
            for indx1 in indx1s:
                if euc_dist[i][indx1] > minval:
                    match[i][indx1] = 0

    euc_dist = [tuple(ele) for ele in euc_dist]  # convert back to list of tuples
    match = [tuple(ele) for ele in match]

    ### validate for each element in match, there is as many as one '1'
    sanity_list = []
    for i, m in enumerate(match):
        cnt_match = m.count(1)
        sanity_list.append(cnt_match)
    if tuple(set(sanity_list)) in [(0, 1), (0), (1)]:
        print("Sanity check is good")

    inference_coord_list = []  # placeholder for inferences that has a match
    label_coord_list = []  # placeholder for label that is matched
    closest_dist_list = []  # placeholder for the closest distances
    inference_coord_radii_list = []  # placeholder for inferences that has a match (xyz and radii)

    # loop again to generate outputs, now each inference should have as many as 1 soma match
    # and one soma should only match to 1 closest inference
    for i, soma_inf_match in enumerate(match):
        soma_inf_match_lst = list(soma_inf_match)

        cnt_match = soma_inf_match_lst.count(1)  # number of matches according to threshold in for each inference

        if cnt_match >= 1:
            # get index of inference that has a match
            ind_inf = match.index(soma_inf_match)

            # get index of matched label in each inference
            ind_lab = soma_inf_match_lst.index(1)

            # get the euc dist for the match
            closest_dist = euc_dist[i][ind_lab]

            # print("index in inference {}, index in label {}".format(ind_inf, ind_lab))
            # print("match:\ninference: {}, label: {}, shortest dist: {}".format(inference[ind_inf], label[ind_lab], closest_dist))

            inference_coord_list.append(inference[ind_inf])
            inference_coord_radii_list.append(inference_with_radii[ind_inf])
            label_coord_list.append(label[ind_lab])
            closest_dist_list.append(closest_dist)

    n_pairs = len(label)  # number of true labels

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
    FP_closest_label_list = []
    FP_coord_radii_list = []  # placeholder for FPs (xyz and radii)

    # FP_all_dist_list = []
    for i, e in enumerate(inference):
        if e not in inference_coord_list:
            FP_list.append(e)
            FP_coord_radii_list.append(inference_with_radii[i])
            min_dist = min(euc_dist[i])

            min_dist_indx = euc_dist[i].index(min_dist)
            closest_label = label[min_dist_indx]

            FP_closest_label_list.append(closest_label)
            FP_closest_dist_list.append(min_dist)

    FP = len(inference) - TP
    FN = len(label) - TP
    print("TP: ", TP, "FP: ", FP, "FN: ", FN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(f"Precision: {precision}, Recall: {recall},\n"
          f"Total number of true labels: {n_pairs}, Total number of inferences: {len(match)}")

    # save matched pairs and all information of this run to a txt file
    # print(current_time)
    with open(path_result / f"precision_recall_{current_time}.txt", 'w') as f:
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
        for inf_fp, closest_lab, dist in zip(FP_list, FP_closest_label_list, FP_closest_dist_list):
            f.write(f"inference: {inf_fp}, closest_label: {closest_lab}, closest distance: {dist}\n")
        # f.write(f"{FP_list}")
        f.write("\n")
        f.write("FNs (among true markers): \n")
        for lab_fn in FN_list:
            f.write(f"lab: {lab_fn}\n")

    # make a directory to save all the FNs marker files
    path_FN_markers = path_result / 'FN_markers'
    path_FN_markers.mkdir(exist_ok=True)

    for FN in FN_list:
        FN_x = FN[0]
        FN_y = FN[1]
        FN_z = FN[2]
        with open(path_FN_markers / f"marker_{FN_x}_{FN_y}_{FN_z}_{int(round(RADII, 0))}", 'w') as marker_file:
            marker_file.write("# x,y,z,radius\n")
            marker_file.write(f"{FN_x},{FN_y},{FN_z},{RADII}")

    FP_df = pd.DataFrame(FP_list, columns=("x", "y", "z"))
    FP_df["comment"] = "FP"
    FP_df["volsize"] = 4/3 * np.pi * array([ele[3] for ele in FP_coord_radii_list])**3

    FN_df = pd.DataFrame(FN_list, columns=("x", "y", "z"))
    FN_df["comment"] = "FN"
    FN_df["volsize"] = RADII**3 * 4/3 * np.pi

    TP_df = pd.DataFrame(TP_list, columns=("x", "y", "z"))
    TP_df["comment"] = "TP"
    TP_df["volsize"] = 4/3 * np.pi * array([ele[3] for ele in inference_coord_radii_list])**3

    df = pd.concat([FP_df, FN_df, TP_df])
    df["color_r"] = np.where(df['comment'] == "FP", 255, 0)
    df["color_g"] = np.where(df['comment'] == "FN", 255, 0)
    df["color_b"] = np.where(df['comment'] == "TP", 255, 0)
    with open(path_result/"test.ano.apo", 'w') as apo_file:
        apo_file.write(
            "##n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b\n")
        for idx, row in enumerate(df.itertuples(), start=1):
            apo_file.write(
                f"{idx},,,{row.comment},{row.z},{row.x},{row.y},0.000,0.000,0.000,{row.volsize},0.000,,,,"
                f"{row.color_r},{row.color_g},{row.color_b}\n")

    with open(path_result/"test.ano.eswc", 'w') as eswc_file:
        eswc_file.write("#name test\n")
        eswc_file.write("#comment terafly_annotations\n")
        eswc_file.write("#n type x y z radius parent seg_id level mode timestamp TFresindex\n")

    with open(path_result/"test.ano", 'w') as ano_file:
        ano_file.write("APOFILE=test.ano.apo\n")
        ano_file.write("SWCFILE=test.ano.eswc\n")

    # generate histogram and cdf for TP and FP
    plot_his_cdf(inference_coord_radii_list, FP_coord_radii_list)


### define a function to plot the histogram and CDF for TPs and FPs
def plot_his_cdf(inference_coord_radii_list, FP_coord_radii_list):
    TP_radii = np.array([ele[3] for ele in inference_coord_radii_list])
    FP_radii = np.array([ele[3] for ele in FP_coord_radii_list])

    radii_min = min(min(TP_radii), min(FP_radii))
    radii_max = max(max(TP_radii), max(FP_radii))

    summary_stats_TP = "Summary statistics for TP radii:\nmean: {:.2f}, std: {:.2f},\nmin: {:.2f}, 25%: {:.2f}, median: {:.2f}, 75%: {:.2f}, max: {:.2f}" \
        .format(np.mean(TP_radii), np.std(TP_radii), np.min(TP_radii), np.percentile(TP_radii, 25), np.median(TP_radii),
                np.percentile(TP_radii, 75), np.max(TP_radii))
    summary_stats_FP = "Summary statistics for FP radii:\nmean: {:.2f}, std: {:.2f},\nmin: {:.2f}, 25%: {:.2f}, median: {:.2f}, 75%: {:.2f}, max: {:.2f}" \
        .format(np.mean(FP_radii), np.std(FP_radii), np.min(FP_radii), np.percentile(FP_radii, 25), np.median(FP_radii),
                np.percentile(FP_radii, 75), np.max(FP_radii))

    ### add ttest
    # if equal variance/not equal vairance (division compared to 4)
    # if <4, equal varinace assumption is met
    if max(np.var(TP_radii), np.var(FP_radii)) / min(np.var(TP_radii), np.var(FP_radii)) <= 4:
        tstats, pval = stats.ttest_ind(a=TP_radii, b=FP_radii, equal_var=True)
        tstats = round(tstats, 2)
        pval = round(pval, 4)
        # print(tstats, pval)
    # if >4, perform Welch's ttest
    elif max(np.var(TP_radii), np.var(FP_radii)) / min(np.var(TP_radii), np.var(FP_radii)) > 4:
        tstats, pval = stats.ttest_ind(a=TP_radii, b=FP_radii, equal_var=False)
        tstats = round(tstats, 2)
        pval = round(pval, 4)
        # print(tstats, pval)

    ### get x limits 
    # x lims:
    xlim_max = max(max(TP_radii), max(FP_radii))

    ### generate figures
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    # plt.figure(figsize=(15, 15)) 

    ### 1,2 plot histograms side by side
    # binwidth=0.5
    axs[0, 0].hist(TP_radii,
                   alpha=0.5,  # the transaparency parameter
                   # bins=np.arange(min(TP_radii), max(TP_radii) + binwidth, binwidth),
                   label='TP radii',
                   color='blue',
                   range=(radii_min, radii_max))
    axs[0, 0].set_xlim(left=0, right=xlim_max)
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].set_title("Distribution of radii of TPs")

    # plt.subplot(3, 2, 2)
    axs[0, 1].hist(FP_radii,
                   alpha=0.5,
                   # bins=np.arange(min(FP_radii), max(FP_radii) + binwidth, binwidth),
                   label='FP radii',
                   color='red',
                   range=(radii_min, radii_max))
    axs[0, 1].set_xlim(left=0, right=xlim_max)
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].set_title("Distribution of radii of FPs")

    # plt.savefig(path/f"histogram_cdf_{current_time}.pdf", format="pdf", bbox_inches="tight")  

    ### 3,4 plot CDFs side by side
    # getting data of the histogram
    cnt_TP, bins_count_TP = np.histogram(TP_radii, range=(radii_min, radii_max))
    cnt_FP, bins_count_FP = np.histogram(FP_radii, range=(radii_min, radii_max))
    # finding the PDF of the histogram using count values
    pdf_TP = cnt_TP / sum(cnt_TP)
    pdf_FP = cnt_FP / sum(cnt_FP)

    # calculate cdf using pdf
    cdf_TP = np.cumsum(pdf_TP)
    cdf_FP = np.cumsum(pdf_FP)

    # plotting PDF and CDF
    # plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    axs[1, 0].plot(bins_count_TP[1:], cdf_TP, label="CDF of TP", color='blue')
    axs[1, 0].set_xlim(left=0, right=xlim_max)
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].legend(loc='upper right')
    axs[1, 0].set_title("CDF of radii of TPs")

    axs[1, 1].plot(bins_count_FP[1:], cdf_FP, label="CDF of FP", color='red')
    axs[1, 1].set_xlim(left=0, right=xlim_max)
    axs[1, 1].set_ylim(bottom=0)
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].set_title("CDF of radii of FPs")

    ### 5,6 plot scatter plots of TP/FP radii vs. true label radii
    axs[2, 0].scatter(np.linspace(0, 1, len(TP_radii)), TP_radii, label='TPs', color='blue')
    axs[2, 0].axhline(y=RADII, linestyle='--', c='green')
    axs[2, 0].set_ylim(bottom=0, top=radii_max)
    axs[2, 0].legend(loc='upper right')
    axs[2, 0].set_title("Distribution of radii of TPs compared to labeld radii")

    axs[2, 1].scatter(np.linspace(0, 1, len(FP_radii)), FP_radii, label='FPs', color='red')
    axs[2, 1].axhline(y=RADII, linestyle='--', c='green')
    axs[2, 1].set_ylim(bottom=0, top=radii_max)
    axs[2, 1].legend(loc='upper right')
    axs[2, 1].set_title("Distribution of radii of FPs compared to labeld radii")

    ### 7,8 plot histogram/CDF overlayed
    axs[3, 0].hist(TP_radii,
                   alpha=0.5,  # the transaparency parameter
                   # bins=np.arange(min(TP_radii), max(TP_radii) + binwidth, binwidth),
                   label='TP radii',
                   color='blue',
                   range=(radii_min, radii_max))
    axs[3, 0].hist(FP_radii,
                   alpha=0.5,
                   # bins=np.arange(min(FP_radii), max(FP_radii) + binwidth, binwidth),
                   label='FP radii',
                   color='red',
                   range=(radii_min, radii_max))
    axs[3, 0].set_xlim(left=0, right=xlim_max)
    axs[3, 0].set_ylim(bottom=0)
    axs[3, 0].legend(loc='upper right')
    axs[3, 0].set_title('Distribution of radii of TPs vs. FPs')

    axs[3, 1].plot(bins_count_TP[1:], cdf_TP, label="CDF of TP", color='blue')
    axs[3, 1].plot(bins_count_FP[1:], cdf_FP, label="CDF of FP", color='red')
    axs[3, 1].set_xlim(left=0, right=xlim_max)
    axs[3, 1].set_ylim(bottom=0)
    axs[3, 1].legend(loc='upper right')
    axs[3, 1].set_title("CDF of radii of TPs vs. FPs")
    axs[3, 1].text(radii_min, -0.3, f'Two sample t-test:\nstatistic: {tstats}, pval: {pval}')
    axs[3, 1].text(radii_min, -0.5, summary_stats_TP)
    axs[3, 1].text(radii_min, -0.8, summary_stats_FP)

    fig.tight_layout(pad=2.5)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    # plt.savefig(path/f"histogram_cdf.pdf", format="pdf")
    plt.savefig(path_result / f"histogram_cdf_{current_time}.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('labeled_path', help="input path for labeled soma location")
    parser.add_argument('inferenced_path', help="input path for inferenced soma location")
    args = parser.parse_args()

    labeled_path = args.labeled_path
    inferenced_path = Path(args.inferenced_path)

    # gather markers
    label, label_with_radii = gather_markers(labeled_path)
    inference, inference_with_radii = gather_markers(inferenced_path)

    # print(label[:10])
    # print(inference[:10])
    # print(euc_distance(label[:10], inference[:10]))
    # exit()

    # directory that stores the precision&recall summary file and the plots
    path_result = inferenced_path.parent / f"result_run_{current_time}"
    path_result.mkdir(exist_ok=True)

    # get precision & recall
    precision_recall()

# # for testing purpose
# label, label_with_radii = gather_markers('C:\\Users\\yanyanming77\\Desktop\\precision_recall\\marker_truth_full')
# inference, inference_with_radii = gather_markers('C:\\Users\\yanyanming77\\Desktop\\precision_recall\\marker_ms_full')
# path = Path("C:\\Users\\yanyanming77\\Desktop\\precision_recall")
# path_result = path/f'result_run_{current_time}'
# path_result.mkdir(exist_ok = True)
