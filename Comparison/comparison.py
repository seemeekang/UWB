import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

from itertools import combinations  

from statistics import mean 


import sys
sys.path.insert(1, 'C:/Users/Jonathan/Documents/GitHub/UWB')
from Standard_Modules.reusable import *
from MDS.skl_classical_MDS import compute_mds
from Initalized_Positions.ip import compute_intialized_positions
from Robust_Quads.algo2 import compute_RQ_algo2 
from RTRR.rtrr import compute_RTRR

# Non - Flipped Nodes Example
# x_og_data = [0,0,2,2,28,10,100]
# y_og_data = [0,2,0,2,35,80,100]

# Flipped Nodes Example
x_og_data = [0,0,2,2,100]
y_og_data = [0,2,0,2,100]

# Total nuber of Anchor nodes
node_count = len(x_og_data)

# The Original (Global) Coordinates of the Anchor Nodes.
og_coordinates = list(zip(x_og_data, y_og_data))
print("Original Coordinates \n",og_coordinates)

# Create a Distance Matrix from the Original Coordinates.
og_distance_matrix = euclidean_distances(og_coordinates)
print("Original Distance Matrix \n",og_distance_matrix)

# Add noise
noise_og_distance_matrix = add_noise_distance_matrix(og_distance_matrix)
in_noise_avg = calculate_InputAvgNoise(og_distance_matrix, noise_og_distance_matrix)
in_noise_stddev = calculate_StdDevNoise(og_distance_matrix, noise_og_distance_matrix)

# MDS & MDS with noise
cal_coordinates_MDS = compute_mds(og_distance_matrix)
MSE_MDS = calculate_dist_MSE(og_distance_matrix, cal_coordinates_MDS)
CSR_MDS = calculate_ClusterSuccessRate(og_distance_matrix, cal_coordinates_MDS, 1)

noise_cal_coordinates_MDS = compute_mds(noise_og_distance_matrix)
noise_MSE_MDS = calculate_dist_MSE(og_distance_matrix, noise_cal_coordinates_MDS)
noise_CSR_MDS = calculate_ClusterSuccessRate(og_distance_matrix, noise_cal_coordinates_MDS, 1)

# IP & IP with noise
cal_coordinates_IP = compute_intialized_positions(og_distance_matrix)
MSE_IP = calculate_dist_MSE(og_distance_matrix, cal_coordinates_IP)
CSR_IP = calculate_ClusterSuccessRate(og_distance_matrix, cal_coordinates_IP, 1)

noise_cal_coordinates_IP = compute_intialized_positions(noise_og_distance_matrix)
noise_MSE_IP = calculate_dist_MSE(og_distance_matrix, noise_cal_coordinates_IP)
noise_CSR_IP = calculate_ClusterSuccessRate(og_distance_matrix, noise_cal_coordinates_IP, 1)

# RQ & RQ with noise
dmin = 0.3
robust_nodes, cal_coordinates_RQ, non_trilaterated_nodes, robust_tris, sort_loc_best_RQ = compute_RQ_algo2(og_distance_matrix, dmin)
MSE_RQ = calculate_dist_MSE_redundant(og_coordinates, sort_loc_best_RQ)
CSR_RQ = calculate_ClusterSuccessRate(og_distance_matrix, cal_coordinates_RQ, 1)

robust_nodes, noise_cal_coordinates_RQ, non_trilaterated_nodes, robust_tris, noise_sort_loc_best_RQ = compute_RQ_algo2(noise_og_distance_matrix, dmin)
noise_MSE_RQ = calculate_dist_MSE_redundant(og_coordinates, noise_sort_loc_best_RQ)
noise_CSR_RQ = calculate_ClusterSuccessRate(og_distance_matrix, noise_cal_coordinates_RQ, 1)

# RTRR & RTRR with noise
# dmin = in_noise_stddev / 3
robust_tri_trilaterate_count = 3
robust_nodes, trilaterated_robust_node_data, rtrr_node_coordinate, cal_coordinates_RTRR = compute_RTRR(og_distance_matrix, dmin, robust_tri_trilaterate_count)
MSE_RTRR = calculate_dist_MSE(og_distance_matrix, cal_coordinates_RTRR)
CSR_RTRR = calculate_ClusterSuccessRate(og_distance_matrix, cal_coordinates_RTRR, 1)


robust_nodes, trilaterated_robust_node_data, rtrr_node_coordinate, noise_cal_coordinates_RTRR = compute_RTRR(noise_og_distance_matrix, dmin, robust_tri_trilaterate_count)
noise_MSE_RTRR = calculate_dist_MSE(og_distance_matrix, noise_cal_coordinates_RTRR)
noise_CSR_RTRR = calculate_ClusterSuccessRate(og_distance_matrix, noise_cal_coordinates_RTRR, 1)

# Plot the two graphs
fig = plt.figure()

# Original Coordinate Plot
ax1 = fig.add_subplot(3,4,1)
ax1.title.set_text('Original')
og_coordinates = [list(ele) for ele in og_coordinates]
og_coordinates = np.array(og_coordinates)
plt.scatter(og_coordinates[:, 0], og_coordinates[:, 1])

ax2 = fig.add_subplot(3,4,5)
ax2.title.set_text("Multidimensional Scaling(MDS) \n" + 
"Inter-node distance MSE: " + str(MSE_MDS) + 
"\nNodes localized: " + str(CSR_MDS) + "%")
plt.scatter(cal_coordinates_MDS[:, 0], cal_coordinates_MDS[:, 1])

ax3 = fig.add_subplot(3,4,6)
ax3.title.set_text("Initialized Positions(IP) \n" + 
"Inter-node distance MSE: " + str(MSE_IP) + 
"\nNodes localized: " + str(CSR_IP) + "%")
plt.scatter(cal_coordinates_IP[:, 0], cal_coordinates_IP[:, 1])

ax4 = fig.add_subplot(3,4,7)
ax4.title.set_text("Robust Quads(RQ) \n" + 
"Inter-node distance MSE: " + str(MSE_RQ) + 
"\nNodes localized: " + str(CSR_RQ) + "%")
plt.scatter(cal_coordinates_RQ[:, 0], cal_coordinates_RQ[:, 1])

ax5 = fig.add_subplot(3,4,8)
ax5.title.set_text("Robust Triangle & Radio Range(RTRR) \n" + 
"Inter-node distance MSE: " + str(MSE_RTRR) + 
"\nNodes localized: " + str(CSR_RTRR) + "%")
plt.scatter(cal_coordinates_RTRR[:, 0], cal_coordinates_RTRR[:, 1])

ax6 = fig.add_subplot(3,4,9)
ax6.title.set_text("Noisy MDS \n" + "Avg i/p noise: " + str(in_noise_avg) + 
"\nInter-node distance MSE: " + str(noise_MSE_MDS) + 
"\nNodes localized: " + str(noise_CSR_MDS) + "%")
plt.scatter(noise_cal_coordinates_MDS[:, 0], noise_cal_coordinates_MDS[:, 1])

ax7 = fig.add_subplot(3,4,10)
ax7.title.set_text("Noisy IP \n" + "Avg i/p noise: " + str(in_noise_avg) + 
"\nInter-node distance MSE: " + str(noise_MSE_IP) + 
"\nNodes localized: " + str(noise_CSR_IP) + "%")
plt.scatter(noise_cal_coordinates_IP[:, 0], noise_cal_coordinates_IP[:, 1])

ax8 = fig.add_subplot(3,4,11)
ax8.title.set_text("Noisy RQ \n" + "Avg i/p noise: " + str(in_noise_avg) + 
"\nInter-node distance MSE: " + str(noise_MSE_RQ) + 
"\nNodes localized: " + str(noise_CSR_RQ) + "%")
plt.scatter(noise_cal_coordinates_RQ[:, 0], noise_cal_coordinates_RQ[:, 1])

ax9 = fig.add_subplot(3,4,12)
ax9.title.set_text("Noisy RTRR \n" + "Avg i/p noise: " + str(in_noise_avg) + 
"\nInter-node distance MSE: " + str(noise_MSE_RTRR) + 
"\nNodes localized: " + str(noise_CSR_RTRR) + "%")
plt.scatter(noise_cal_coordinates_RTRR[:, 0], noise_cal_coordinates_RTRR[:, 1])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.80,
                    wspace=0.35)

plt.show()