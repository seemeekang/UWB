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

def trilaterate_post_sc(anchor_coordinates, tag_coordinates):
    anchor_x = [node[0] for node in  anchor_coordinates]
    anchor_y = [node[1] for node in  anchor_coordinates]

    anchor_node_count = len(anchor_coordinates)
    tag_node_count = len(tag_coordinates)

    cal_tag_coordinates = []

    for tag_node in tag_coordinates:
        tag_node = np.array(tag_node)
        anchor_tag_coordinates = np.insert(anchor_coordinates, 0, tag_node, axis=0)
        print("anchor_tag_coordinates", anchor_tag_coordinates)

        anchor_distance_matrix = euclidean_distances(anchor_coordinates)
        anchor_tag_distance_matrix = euclidean_distances(anchor_tag_coordinates)

        r1_sq = pow(anchor_tag_distance_matrix[0,1],2)
        r2_sq = pow(anchor_tag_distance_matrix[0,2],2)
        r3_sq = pow(anchor_tag_distance_matrix[0,3],2)

        # Solve a linear matrix equation where x,y is the Tag coordinate:
        # Ax + By = C
        # Dx + Ey = F
        A = (-2*anchor_x[0]) + (2*anchor_x[1])
        B = (-2*anchor_y[0]) + (2*anchor_y[1])
        C = r1_sq - r2_sq - pow(anchor_x[0],2) + pow(anchor_x[1],2) - pow(anchor_y[0],2) + pow(anchor_y[1],2) 
        D = (-2*anchor_x[1]) + (2*anchor_x[2])
        E = (-2*anchor_y[1]) + (2*anchor_y[2])
        F = r2_sq - r3_sq - pow(anchor_x[1],2) + pow(anchor_x[2],2) - pow(anchor_y[1],2) + pow(anchor_y[2],2) 

        a = np.array([[A, B], [D, E]])
        b = np.array([C, F])
        tag_coordinates = np.linalg.solve(a, b)
        print("Tag Coordinate:", tag_coordinates)

        cal_tag_coordinates.append(tag_coordinates)
    
    cal_tag_coordinates = np.array(cal_tag_coordinates)
    print("cal_tag_coordinatese:", cal_tag_coordinates)

    return cal_tag_coordinates

def main():
    # Non - Flipped Nodes Example
    # og_anchor_x = [0,0,2,2,28,10,100]
    # og_anchor_y = [0,2,0,2,35,80,100]

    # Flipped Nodes Example
    og_anchor_x = [0,0,2,2,100]
    og_anchor_y = [0,2,0,2,100]


    # The Original (Global) Coordinates of the Anchor Nodes.
    og_anchor_coordinates = list(zip(og_anchor_x, og_anchor_y))
    print("Original Anchor Coordinates \n",og_anchor_coordinates)

    # Create a Distance Matrix from the Original Coordinates.
    og_anchor_distance_matrix = euclidean_distances(og_anchor_coordinates)
    print("Original Anchor Distance Matrix \n",og_anchor_distance_matrix)

    # Original Tag Node Coordinates.
    og_tag_x = [25, 50, 75]
    og_tag_y = [25, 50, 75]

    # The Original (Global) Coordinates of the Tag Nodes.
    og_tag_coordinates = list(zip(og_tag_x, og_tag_y))
    print("Original Anchor Coordinates \n",og_tag_coordinates)


    cal_anchor_coordinates_MDS = compute_mds(og_anchor_distance_matrix)
    cal_tag_coordinates = trilaterate_post_sc(cal_anchor_coordinates_MDS, og_tag_coordinates)

    # Add noise to input distance matrix
    noise_anchor_distance_matrix = add_noise_distance_matrix(og_anchor_distance_matrix)
    anchor_in_noise_avg = calculate_InputAvgNoise(og_anchor_distance_matrix, noise_anchor_distance_matrix)

    noise_cal_anchor_coordinates_MDS = compute_mds(noise_anchor_distance_matrix)
    noise_cal_tag_coordinates = trilaterate_post_sc(noise_cal_anchor_coordinates_MDS, og_tag_coordinates)

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

    # MDS Coordinate Plot
    ax2 = fig.add_subplot(1,3,2)
    ax2.title.set_text('MDS')
    plt.scatter(cal_anchor_coordinates_MDS[:, 0], cal_anchor_coordinates_MDS[:, 1], c='blue')
    plt.scatter(cal_tag_coordinates[:, 0], cal_tag_coordinates[:, 1], c='red')


    # Noise MDS Coordinate Plot
    ax2 = fig.add_subplot(1,3,3)
    ax2.title.set_text('Noisy MDS')
    plt.scatter(noise_cal_anchor_coordinates_MDS[:, 0], noise_cal_anchor_coordinates_MDS[:, 1], c='blue')
    plt.scatter(noise_cal_tag_coordinates[:, 0], noise_cal_tag_coordinates[:, 1], c='red')    

    plt.show()

if __name__ == "__main__":
    main()