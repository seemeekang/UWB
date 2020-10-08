import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

from itertools import combinations  

import sys
sys.path.insert(1, 'C:/Users/Jonathan/Documents/GitHub/UWB')
from Standard_Modules.reusable import *
from MDS.skl_classical_MDS import compute_mds

def compute_mds_tag(og_anchor_coordinates, og_anchor_distance_matrix, og_tag_coordinates):
    cal_anchor_coordinates = compute_mds(og_anchor_distance_matrix)
    MSE_dist_anchor = calculate_dist_MSE(og_anchor_distance_matrix, cal_anchor_coordinates)
    MSE_loc_anchor = calculate_MSE_loc(og_anchor_coordinates, cal_anchor_coordinates)
    CSR = calculate_ClusterSuccessRate(og_anchor_distance_matrix, cal_anchor_coordinates, 1)

    cal_tag_coordinates = trilaterate_post_sc(cal_anchor_coordinates, og_tag_coordinates)
    MSE_loc_tag = calculate_MSE_loc(og_tag_coordinates, cal_tag_coordinates)

    return cal_anchor_coordinates, MSE_dist_anchor, MSE_loc_anchor, CSR, cal_tag_coordinates, MSE_loc_tag


def main():
    # Non - Flipped Nodes Example
    # og_anchor_x = [0,0,2,2,28,10,100]
    # og_anchor_y = [0,2,0,2,35,80,100]

    # Flipped Nodes Example
    og_anchor_x = [0,0,2,2,100]
    og_anchor_y = [0,2,0,2,100]

    # Original Anchor Coordinates and Distance Matrix
    og_anchor_coordinates = list(zip(og_anchor_x, og_anchor_y))
    og_anchor_distance_matrix = euclidean_distances(og_anchor_coordinates)

    # Original Tag Node Coordinates.
    og_tag_x = [25, 50, 75]
    og_tag_y = [25, 50, 75]
    og_tag_coordinates = list(zip(og_tag_x, og_tag_y))
    print("Original Tag Coordinates \n",og_tag_coordinates)

    # Add noise to input distance matrix
    n_anchor_distance_matrix = add_noise_distance_matrix(og_anchor_distance_matrix)
    anchor_in_noise_avg = calculate_InputAvgNoise(og_anchor_distance_matrix, n_anchor_distance_matrix)

    cal_a_crdnt, MSE_dist_a, MSE_loc_a, CSR, cal_t_crdnt, MSE_loc_t = compute_mds_tag(
            og_anchor_coordinates, og_anchor_distance_matrix, og_tag_coordinates)
    

    n_cal_a_crdnt, n_MSE_dist_a, n_MSE_loc_a, n_CSR, n_cal_t_crdnt, n_MSE_loc_t = compute_mds_tag(
            og_anchor_coordinates, n_anchor_distance_matrix, og_tag_coordinates)

    # Plot the two graphs
    fig = plt.figure()

    # Original Coordinate Plot
    ax1 = fig.add_subplot(1,3,1)
    ax1.title.set_text('Original')
    og_anchor_coordinates = [list(ele) for ele in og_anchor_coordinates]
    og_tag_coordinates = [list(ele) for ele in og_tag_coordinates]
    og_anchor_coordinates = np.array(og_anchor_coordinates)
    og_tag_coordinates = np.array(og_tag_coordinates)
    plt.scatter(og_anchor_coordinates[:, 0], og_anchor_coordinates[:, 1], c='blue')
    plt.scatter(og_tag_coordinates[:, 0], og_tag_coordinates[:, 1], c='red')
    

    ax2 = fig.add_subplot(1,3,2)
    ax2.title.set_text("MDS" + 
    "\nInter-anchor distance MSE: " + str(MSE_dist_a) +
    "\nAnchor Location MSE: " + str(MSE_loc_a) +
    "\nCSR: " + str(CSR) + "%" + 
    "\nTag Location MSE: " + str(MSE_loc_t))
    plt.scatter(cal_a_crdnt[:, 0], cal_a_crdnt[:, 1], c='blue')
    plt.scatter(cal_t_crdnt[:, 0], cal_t_crdnt[:, 1], c='red')

    ax3 = fig.add_subplot(1,3,3)
    ax3.title.set_text("Noisy MDS \n" + "Avg i/p noise: " + str(anchor_in_noise_avg) +
    "\nInter-anchor distance MSE: " + str(n_MSE_dist_a) +
    "\nAnchor Location MSE: " + str(n_MSE_loc_a) +
    "\nCSR: " + str(n_CSR) + "%" + 
    "\nTag Location MSE: " + str(n_MSE_loc_t))
    plt.scatter(n_cal_a_crdnt[:, 0], n_cal_a_crdnt[:, 1], c='blue')
    plt.scatter(n_cal_t_crdnt[:, 0], n_cal_t_crdnt[:, 1], c='red')

    plt.show()
    
if __name__ == "__main__":
    main()