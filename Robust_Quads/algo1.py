import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

from itertools import combinations  


# Test if all test_nodes are already Robust
def test_node_robustness(test_quad, robust_nodes):
    result =  all(elem in robust_nodes  for elem in test_quad)
    return result

# Get triangle angle/side data in order to test it's robustness
def get_triangle_angles(node1, node2, node3, distance_matrix):
    # length of sides be a, b, c  
    a = distance_matrix[node1, node2]  
    b = distance_matrix[node1, node3]  
    c = distance_matrix[node2, node3]

    # print("a ",a)
    # print("b ",b)
    # print("c ",c)

    # Square of lengths be a2, b2, c2  
    a2 = pow(a, 2)  
    b2 = pow(b, 2)  
    c2 = pow(c, 2)

    # From Cosine law
    temp_alpha = round(((b2 + c2 - a2) / (2 * b * c)), 3)
    temp_betta = round(((a2 + c2 - b2) / (2 * a * c)), 3)
    temp_gamma = round(((a2 + b2 - c2) / (2 * a * b)), 3)   
    alpha = math.acos(temp_alpha)  
    betta = math.acos(temp_betta)  
    gamma = math.acos(temp_gamma)  
 
    # Converting to degree  
    alpha = alpha * 180 / math.pi  
    betta = betta * 180 / math.pi  
    gamma = gamma * 180 / math.pi

    # alpha = round(math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))))
    # betta = round(math.degrees(math.acos((c ** 2 + a ** 2 - b ** 2) / (2 * c * a))))
    # gamma = round(math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))))

    angles = [alpha, betta, gamma]
    sides = [a, b, c]
    return angles, sides   

# Test if a Triangle is robust
def test_triangle_robustnes(node1, node2, node3, distance_matrix, dmin):
    angles, sides = get_triangle_angles(node1, node2, node3, distance_matrix)
    smallest_angle = min(angles)
    shortest_side = min(sides)

    if ((shortest_side) * (pow(math.sin(smallest_angle),2)) > dmin):
        triangle_robustness = True
    else:
        triangle_robustness = False
    return triangle_robustness


# Test if a Quadilateral is robust
def test_quad_robustness(test_quad, distance_matrix, dmin):
    d012 = test_triangle_robustnes(test_quad[0], test_quad[1], test_quad[2], distance_matrix, dmin)
    d013 = test_triangle_robustnes(test_quad[0], test_quad[1], test_quad[3], distance_matrix, dmin)
    d023 = test_triangle_robustnes(test_quad[0], test_quad[2], test_quad[3], distance_matrix, dmin)
    d123 = test_triangle_robustnes(test_quad[1], test_quad[2], test_quad[3], distance_matrix, dmin)

    if d012 & d013 & d023 & d123:
        quad_robustness = True
        robust_triangles = [(test_quad[0], test_quad[1], test_quad[2]), 
                            (test_quad[0], test_quad[1], test_quad[3]), 
                            (test_quad[0], test_quad[2], test_quad[3]), 
                            (test_quad[1], test_quad[2], test_quad[3])]
    else:
        quad_robustness = False
        robust_triangles = []
    return quad_robustness, robust_triangles

def compute_RQ_algo1(distance_matrix, dmin):
    node_count = len(distance_matrix)

    # Node Name List Creation
    node_name = list(range(0, node_count))
    print("All nodes:", node_name)
    test_node_comb = list(combinations(node_name,4))
    test_quad_name = []
    for test_quad in test_node_comb:
        if test_quad[0] == 0:
            test_quad_name.append(test_quad)
    print(test_quad_name)

    # Create list of Robust Nodes and Robust Quads
    robust_nodes = []
    robust_quads = []
    robust_tris = []

    for test_quad in test_quad_name:
        if test_node_robustness(test_quad, robust_nodes) == True:
            continue
        else:
            quad_robust_check, robust_tris_per_quad = test_quad_robustness(test_quad, distance_matrix, dmin)

        if quad_robust_check == True:
            # Append the Robust Nodes and Robust Triangles
            robust_nodes.extend(test_quad)
            robust_tris.extend(robust_tris_per_quad)
            # Remove duplicate Robust Nodes and Robust Triangles
            robust_nodes = list(set(robust_nodes))
            robust_tris = list(set(robust_tris))
            # Append the Robust Quad
            robust_quads.append(test_quad)

    print("Robust Nodes", robust_nodes)
    print("Robust Quads", robust_quads)
    print("Robust Triangles", robust_tris)

    return robust_nodes, robust_quads, robust_tris


def main():
    # Threshold Measurement Noise
    dmin = 0.0

    # Non - Flipped Nodes Example
    # x_og_data = [0,0,2,2,28,10,100]
    # y_og_data = [0,2,0,2,35,80,100]

    # Flipped Nodes Example
    x_og_data = [0,0,2,2,100]
    y_og_data = [0,2,0,2,100]

    # The Original (Global) Coordinates of the Anchor Nodes.
    og_coordinates = list(zip(x_og_data, y_og_data))
    print("Original Coordinates \n",og_coordinates)

    # Create a Distance Matrix from the Original Coordinates.
    og_distance_matrix = euclidean_distances(og_coordinates)
    print("Original Distance Matrix \n",og_distance_matrix)

    robust_nodes, robust_quads, robust_tris = compute_RQ_algo1(og_distance_matrix, dmin)

    # Plot the two graphs
    fig = plt.figure()

    # Original Coordinate Plot
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Original')
    og_coordinates = [list(ele) for ele in og_coordinates]
    og_coordinates = np.array(og_coordinates)
    # print(og_coordinates) 
    plt.scatter(og_coordinates[:, 0], og_coordinates[:, 1])

    ax2 = fig.add_subplot(122)
    ax2.title.set_text("Robust Node Quads")
    plt.scatter(og_coordinates[:, 0], og_coordinates[:, 1], c='red')
    plt.scatter(og_coordinates[robust_nodes, 0], og_coordinates[robust_nodes, 1], c='green')

    plt.show()

if __name__ == "__main__":
    main()