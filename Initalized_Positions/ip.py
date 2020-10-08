# Based on algorithm used in "Self-Calibrating Ultra-Wideband Network Supporting Multi-Robot Localization"

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

from itertools import combinations  



# Trilaterate nodes
def trilateration(node_to_trilaterate, initial_localized_nodes, initial_localized_coordinates, distance_matrix):
    # Compute square of distance between the first three initial nodes and the node to be trilaterated
    r1_sq = pow(distance_matrix[node_to_trilaterate, initial_localized_nodes[0]], 2)
    r2_sq = pow(distance_matrix[node_to_trilaterate, initial_localized_nodes[1]], 2)
    r3_sq = pow(distance_matrix[node_to_trilaterate, initial_localized_nodes[2]], 2)

    initial_localized_coordinates = np.array(initial_localized_coordinates) # Convert to np array
    anchor_x = initial_localized_coordinates[:,0]
    anchor_y = initial_localized_coordinates[:,1]

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
    anchor_coordinates = np.linalg.solve(a, b)
    anchor_coordinates = np.array(anchor_coordinates).tolist()
    # print("Tag Coordinate:", anchor_coordinates)
    return anchor_coordinates


# Get list of robust nodes with trilaterated coordinates
def compute_intialized_positions(distance_matrix):
    node_count = len(distance_matrix)
    node_name = list(range(0, node_count))
    
    loc_best = [] #Locsbest
    trilaterated_nodes = []
    trilaterated_nodes_coordinates = []
    initial_localized_coordinates = []

    d_ab = distance_matrix[node_name[0], node_name[1]]
    d_ac = distance_matrix[node_name[0], node_name[2]]
    d_bc = distance_matrix[node_name[1], node_name[2]]

    # Calculate Initialized Nodes p0 & p1
    loc_best.append([node_name[0],(0,0)])       # p0
    loc_best.append([node_name[1],(d_ab,0)])    # p1
    
    # Calculate Initialized Nodes p2
    alpha = (pow(d_ab,2) + pow(d_ac,2) - pow(d_bc,2)) / (2*d_ab*d_ac)
    loc_best.append([node_name[2],(d_ac*alpha,d_ac*math.sqrt(1-pow(alpha,2)))]) # p2

    # Keep a tab of Intially localized nodes. 
    trilaterated_nodes = [node_name[0], node_name[1], node_name[2]]
    initial_localized_coordinates = [loc_best[0][1], loc_best[1][1], loc_best[2][1]]
    trilaterated_nodes_coordinates = initial_localized_coordinates
    initial_localized_nodes = trilaterated_nodes
    
    # Localize the remaining anchor nodes using the Intially Localized nodes.
    for node in node_name:
        if node not in trilaterated_nodes:
            node_coordinates = trilateration(node, initial_localized_nodes, initial_localized_coordinates, distance_matrix)
            loc_best.append([node, node_coordinates])
            trilaterated_nodes.append(node)
            trilaterated_nodes_coordinates.append(node_coordinates)

    # print(initial_localized_coordinates)    
    # print("Loc_best", loc_best)
    # print("trilaterated_nodes", trilaterated_nodes)
    # print("trilaterated_nodes_coordinates", trilaterated_nodes_coordinates)

    return np.array(trilaterated_nodes_coordinates)

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

    # Plot Initialized Positions Algorithm Output
    ip_coordinates = compute_intialized_positions(og_distance_matrix)
    ip_MSE = calculate_dist_MSE(og_distance_matrix, ip_coordinates)
    ax2 = fig.add_subplot(132)
    ax2.title.set_text("Trilaterated Nodes using Initialized Anchors \n Inter-node distance MSE: " + str(ip_MSE))
    plt.scatter(ip_coordinates[:,0], ip_coordinates[:,1])

    # Add noise to the Original Distance matrix
    noise_og_distance_matrix = add_noise_distance_matrix(og_distance_matrix)

    # Calculate avg inter-node input error
    in_noise_avg = calculate_InputAvgNoise(og_distance_matrix, noise_og_distance_matrix)

    # Plot Noisy Initialized Positions Algorithm Output
    noisy_ip_coordinates = compute_intialized_positions(noise_og_distance_matrix)
    noisy_ip_MSE = calculate_dist_MSE(og_distance_matrix, noisy_ip_coordinates)
    ax3 = fig.add_subplot(133)
    ax3.title.set_text("Noisy Results \n" + "Avg i/p noise: " + str(in_noise_avg) + 
    "\n Inter-node distance MSE: " + str(noisy_ip_MSE))
    plt.scatter(noisy_ip_coordinates[:, 0], noisy_ip_coordinates[:, 1])

    plt.show()

if __name__ == "__main__":
    main()