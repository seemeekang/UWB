import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

from itertools import combinations

import sys
sys.path.insert(1, 'C:/Users/Jonathan/Documents/GitHub/UWB/Robust_Quads')
from algo2 import compute_RQ_algo2


# Trilaterate nodes
def trilateration(node_to_trilaterate, robust_triangle_nodes, distance_matrix, sort_loc_best):
    # Compute square of distance between the first three initial nodes and the node to be trilaterated
    r1_sq = pow(distance_matrix[node_to_trilaterate, robust_triangle_nodes[0]], 2)
    r2_sq = pow(distance_matrix[node_to_trilaterate, robust_triangle_nodes[1]], 2)
    r3_sq = pow(distance_matrix[node_to_trilaterate, robust_triangle_nodes[2]], 2)

    robust_triangle_nodes_coordinates  = []
    for node in robust_triangle_nodes:
        for data in sort_loc_best:
            if node == data[0]:
                robust_triangle_nodes_coordinates.append(data[1])

    robust_triangle_nodes_coordinates = np.array(robust_triangle_nodes_coordinates) # Convert to np array
    anchor_x = robust_triangle_nodes_coordinates[:,0]
    anchor_y = robust_triangle_nodes_coordinates[:,1]

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
    tag_coordinates = np.array(tag_coordinates).tolist()
    # print("Tag Coordinate:", tag_coordinates)
    return tag_coordinates


def main():
    # Threshold Measurement Noise
    dmin = 1.0

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

    robust_nodes, trilaterated_robust_node_data, untrilated_nodes, robust_triangles, sort_robust_loc  = compute_RQ_algo2(og_distance_matrix, dmin)

    robust_tri_trilaterate_count = 3

    for non_trilaterated_node in untrilated_nodes:
        x_list = []
        y_list = []
        trilaterated_loc = []  
        for robust_triangle_no in range(0, robust_tri_trilaterate_count):
            triangle_nodes = robust_triangles[robust_triangle_no]
            tag_coordinates = trilateration(non_trilaterated_node, triangle_nodes, og_distance_matrix, sort_robust_loc)
            x_list.append(tag_coordinates[0])
            y_list.append(tag_coordinates[1])
        x_avg = round(sum(x_list) / len(x_list), 2)
        y_avg = round(sum(y_list) / len(y_list), 2)
        trilaterated_loc.append([non_trilaterated_node, [x_avg,y_avg]])
    sort_trilaterated_loc = sorted(trilaterated_loc, key=lambda x: x[0])

    rtrr_node_coordinate = np.array([data[1] for data in sort_trilaterated_loc])

    all_loc = sort_robust_loc
    all_loc.extend(sort_trilaterated_loc)
    sort_all_loc = sorted(all_loc, key=lambda x: x[0])

    print("sort_all_loc", sort_all_loc)
    print("rtrr_node_coordinate", rtrr_node_coordinate)






    # Plot the two graphs
    fig = plt.figure()

    # Original Coordinate Plot
    ax1 = fig.add_subplot(141)
    ax1.title.set_text('Original')
    og_coordinates = [list(ele) for ele in og_coordinates]
    og_coordinates = np.array(og_coordinates)
    # print(og_coordinates) 
    plt.scatter(og_coordinates[:, 0], og_coordinates[:, 1])

    ax2 = fig.add_subplot(142)
    ax2.title.set_text("Robust Node Quads")
    plt.scatter(og_coordinates[:, 0], og_coordinates[:, 1], c='red')
    plt.scatter(og_coordinates[robust_nodes, 0], og_coordinates[robust_nodes, 1], c='green')

    ax3 = fig.add_subplot(143)
    ax3.title.set_text("Trilaterated Robust Nodes")
    plt.scatter(trilaterated_robust_node_data[:,0], trilaterated_robust_node_data[:,1], c='green')

    ax4 = fig.add_subplot(144)
    ax4.title.set_text("RTRR")
    plt.scatter(rtrr_node_coordinate[:,0], rtrr_node_coordinate[:,1], c='yellow')
    plt.scatter(trilaterated_robust_node_data[:,0], trilaterated_robust_node_data[:,1], c='green')

    plt.show()

if __name__ == "__main__":
    main()