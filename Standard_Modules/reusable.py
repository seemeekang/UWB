import numpy as np

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

# Add noise to a inter-node distance matrix
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

# Calculate inter-node avg input noise added to a distance matrix
def calculate_InputAvgNoise(og_distance_matrix, noise_og_distance_matrix):
    in_noise_avg = np.square(og_distance_matrix - noise_og_distance_matrix).mean()
    return round(in_noise_avg, 2)

# Calculate inter-node stddev input noise added to a distance matrix
def calculate_StdDevNoise(og_distance_matrix, noise_og_distance_matrix):
    # in_noise_stddev = np.square(og_distance_matrix - noise_og_distance_matrix).std()
    in_noise_stddev = np.square(og_distance_matrix - noise_og_distance_matrix).std()
    print("in_noise_stddev", in_noise_stddev)
    return round(in_noise_stddev, 2)

# Calculate the inter-node MSE between the Orginal Distance Matrix and Calculated Coordinates (Calculated Distance Matrix).
def calculateMSE(og_distance_matrix, cal_coordinates):
    cal_distance_matrix = euclidean_distances(cal_coordinates)
    MSE = np.square(og_distance_matrix - cal_distance_matrix).mean()
    print(MSE)
    return round(MSE, 2)