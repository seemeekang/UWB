import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math


def compute_mds(og_distance_matrix):
    # Intialize MDS parameters:
    # n_components = 2 since the it's a 2-axis graph.
    # dissimilarity = "precomputed". Since the Distance Matrix is precomputed.
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=1,
                    dissimilarity="precomputed", n_jobs=1)

    # Compute the local coordinates from the Distance Matrix using MDS.
    cal_coordinates = mds.fit(og_distance_matrix).embedding_
    # print("Calculated Coordinates \n",cal_coordinates)

    return cal_coordinates

# Standard Modules
def calculate_InputAvgNoise(og_distance_matrix, noise_og_distance_matrix):
    in_noise_avg = np.square(og_distance_matrix - noise_og_distance_matrix).mean()
    return round(in_noise_avg, 2)

def calculate_dist_MSE(og_distance_matrix, cal_coordinates):
    cal_distance_matrix = euclidean_distances(cal_coordinates)
    dist_MSE = np.square(og_distance_matrix - cal_distance_matrix).mean()
    return round(dist_MSE, 5)

def add_noise_distance_matrix(og_distance_matrix):
    node_count = len(og_distance_matrix)
    noise = np.random.rand(node_count, node_count)
    noise = noise + noise.T
    noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
    noise_og_distance_matrix = og_distance_matrix + noise

    # noise_og_distance_matrix = np.array([[0., 2.96828362, 3.08062801, 3.43643702, 76.32032917],
    # [2.96828362, 0., 4.78298261, 3.25729649, 74.66646075],
    # [3.08062801, 4.78298261, 0., 2.45696503, 74.34736739],
    # [3.43643702, 3.25729649, 2.45696503, 0., 72.89990305],
    # [76.32032917, 74.66646075, 74.34736739, 72.89990305, 0.]])

    return noise_og_distance_matrix

def main():
    # Non - Flipped Nodes Example
    # x_og_data = [0,0,2,2,28,10,100]
    # y_og_data = [0,2,0,2,35,80,100]

    # Flipped Nodes Example
    x_og_data = [0,0,2,2,100]
    y_og_data = [0,2,0,2,100]

    # Total nuber of Anchor nodes
    # node_count = len(x_og_data)

    # The Original (Global) Coordinates of the Anchor Nodes.
    og_coordinates = list(zip(x_og_data, y_og_data))
    print("Original Coordinates \n",og_coordinates)

    # Create a Distance Matrix from the Original Coordinates.
    og_distance_matrix = euclidean_distances(og_coordinates)
    print("Original Distance Matrix \n",og_distance_matrix)

    # Plot the two graphs
    fig = plt.figure()

    # Original Coordinate Plot
    ax1 = fig.add_subplot(131)
    ax1.title.set_text('Original')
    og_coordinates = [list(ele) for ele in og_coordinates]
    og_coordinates = np.array(og_coordinates)
    # print(og_coordinates) 
    plt.scatter(og_coordinates[:, 0], og_coordinates[:, 1])

    # Calculated Coordinate Plot
    cal_coordinates = compute_mds(og_distance_matrix)
    anchor_dist_MSE = calculate_dist_MSE(og_distance_matrix, cal_coordinates)
    ax2 = fig.add_subplot(132)
    ax2.title.set_text("Calculated MDS \n with Avg Distance Error: " + str(anchor_dist_MSE))
    plt.scatter(cal_coordinates[:, 0], cal_coordinates[:, 1])

    # Add noise to input distance matrix
    noise_og_distance_matrix = add_noise_distance_matrix(og_distance_matrix)
    in_noise_avg = calculate_InputAvgNoise(og_distance_matrix, noise_og_distance_matrix)

    # Noisy Input Coordinate Plot
    cal_coordinates = compute_mds(noise_og_distance_matrix)
    anchor_dist_MSE = calculate_dist_MSE(og_distance_matrix, cal_coordinates)
    ax2 = fig.add_subplot(133)
    ax2.title.set_text("Calculated MDS (Noisy Input) \n with Avg Distance Error: " + str(anchor_dist_MSE) 
    + "\n Noisy Input Avg: " + str(in_noise_avg))
    plt.scatter(cal_coordinates[:, 0], cal_coordinates[:, 1])

    plt.show()

if __name__ == "__main__":
    main()