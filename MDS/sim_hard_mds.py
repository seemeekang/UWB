import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

import serial


def compute_mds(og_distance_matrix):
    # Intialize MDS parameters:
    # n_components = 2 since the it's a 2-axis graph.
    # dissimilarity = "precomputed". Since the Distance Matrix is precomputed.
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=1,
                    dissimilarity="precomputed", n_jobs=1)

    # Compute the local coordinates from the Distance Matrix using MDS.
    cal_coordinates = mds.fit(og_distance_matrix).embedding_
    print("Calculated Coordinates \n",cal_coordinates)

    return cal_coordinates

def compute_tag_location(tag_coordinates, og_coordinates, cal_coordinates, measured_distance):
    # Insert tag coordinates to np.array containing anchor coordinates
    anchor_tag = og_coordinates
    anchor_tag = np.insert(anchor_tag, 0, tag_coordinates, axis=0)

    # Calculate the distance between tag and anchor nodes i.e 1st row of distance_matrix
    distance_matrix = euclidean_distances(anchor_tag)
    # print("Euclidean Distances",distance_matrix)

    # Insert Measured Distance optionally
    r1_sq = pow(distance_matrix[0,1],2)
    if measured_distance != None:
        r1_sq = pow(measured_distance,2)
    r2_sq = pow(distance_matrix[0,2],2)
    r3_sq = pow(distance_matrix[0,3],2)

    anchor_x = cal_coordinates[:, 0]
    anchor_y = cal_coordinates[:, 1]

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
    return tag_coordinates

def tag_distance_matrix_error(tag_coordinates, og_coordinates, cal_tag_coordinates, cal_coordinates):
    anchor_tag = og_coordinates
    anchor_tag = np.insert(anchor_tag, 0, tag_coordinates, axis=0)
    og_distance_matrix = euclidean_distances(anchor_tag)

    cal_anchor_tag = cal_coordinates
    cal_anchor_tag = np.insert(cal_anchor_tag, 0, cal_tag_coordinates, axis=0)
    cal_distance_matrix = euclidean_distances(cal_anchor_tag)

    tag_MSE = np.square(og_distance_matrix - cal_distance_matrix).mean()
    print("Tag MSE \n",tag_MSE)

    node_count = len(og_coordinates)
    sum_og_distance_matrix = np.sum(og_distance_matrix)
    avg_og_distance_matrix = (sum_og_distance_matrix) / (2 * node_count)
    tag_MSE_pcnt = (tag_MSE * 100) / avg_og_distance_matrix

    return tag_MSE, tag_MSE_pcnt

# Standard Modules
def calculate_InputAvgNoise(og_distance_matrix, noise_og_distance_matrix):
    in_noise_avg = np.square(og_distance_matrix - noise_og_distance_matrix).mean()
    return round(in_noise_avg, 2)

def calculate_dist_MSE(og_distance_matrix, cal_coordinates):
    cal_distance_matrix = euclidean_distances(cal_coordinates)
    MSE = np.square(og_distance_matrix - cal_distance_matrix).mean()
    return round(MSE, 5)

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
    # Configure Serial Communication
    ser = serial.Serial(
        port='COM5',\
        baudrate=115200,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
        timeout=None)
    print("connected to: " + ser.portstr)

    # Measured hardware distance between 1st anchor and tag node.
    measured_distance = float(ser.readline(12).decode('utf-8'))

    # Non - Flipped Nodes Example
    x_og_data = [0,58,58]
    y_og_data = [43,43,0]

    # Placed Tag Location. This is only required for Simulation with only 2 hardware nodes.
    tag_placed_at = 40  # (10,10) - (40,40)

    # Tag Node
    tag_coordinates = [(tag_placed_at,tag_placed_at)]

    # Total nuber of Anchor nodes
    # node_count = len(x_og_data)

    # The Original (Global) Coordinates of the Anchor Nodes. These are ideal values.
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
    plt.scatter(tag_coordinates[0][0], tag_coordinates[0][1],c='red')

    # Calculated Coordinate Plot
    cal_coordinates = compute_mds(og_distance_matrix)
    MSE = calculate_dist_MSE(og_distance_matrix, cal_coordinates)
    cal_tag_coordinates = compute_tag_location(tag_coordinates, og_coordinates, cal_coordinates, None)
    cal_tag_MSE, cal_tag_MSE_pcnt = tag_distance_matrix_error(tag_coordinates, og_coordinates, cal_tag_coordinates, cal_coordinates)

    ax2 = fig.add_subplot(132)
    ax2.title.set_text("Calculated MDS \n Avg Distance Error: " + str(MSE)
    + "\n Avg Tag Distance Error: " + str(round(cal_tag_MSE,10)))
    plt.scatter(cal_coordinates[:, 0], cal_coordinates[:, 1])
    plt.scatter(cal_tag_coordinates[0], cal_tag_coordinates[1],c='red')

    # Add noise to the Original Distance matrix
    noise_og_distance_matrix = add_noise_distance_matrix(og_distance_matrix)
    in_noise_avg = calculate_InputAvgNoise(og_distance_matrix, noise_og_distance_matrix)

    # Noisy Input Coordinate Plot with Measured Distance from DWM1001 nodes
    noise_cal_coordinates = compute_mds(noise_og_distance_matrix)
    noise_MSE = calculate_dist_MSE(og_distance_matrix, noise_cal_coordinates)
    noise_cal_tag_coordinates = compute_tag_location(tag_coordinates, og_coordinates, noise_cal_coordinates, measured_distance)
    noise_cal_tag_MSE, noise_cal_tag_MSE_pcnt = tag_distance_matrix_error(tag_coordinates, og_coordinates, noise_cal_tag_coordinates, noise_cal_coordinates)

    ax2 = fig.add_subplot(133)
    ax2.title.set_text("Calculated MDS + Hardware (Noisy) \n Avg Input Noise: " + str(in_noise_avg) 
    + "\n Avg Tag Distance Error: " + str(round(noise_cal_tag_MSE_pcnt,3)) + " %"
    + "\n Measured Distance: " + str(round(measured_distance,3)) + " cm")

    plt.scatter(noise_cal_coordinates[:, 0], noise_cal_coordinates[:, 1])
    plt.scatter(noise_cal_tag_coordinates[0], noise_cal_tag_coordinates[1],c='red')

    plt.show()

if __name__ == "__main__":
    main()