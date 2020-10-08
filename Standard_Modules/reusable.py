import numpy as np

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

from itertools import combinations  
from statistics import mean 

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
    # print("in_noise_stddev", in_noise_stddev)
    return round(in_noise_stddev, 2)

# Calculate the inter-node MSE between the Orginal Distance Matrix and Calculated Coordinates (Calculated Distance Matrix).
def calculate_dist_MSE(og_distance_matrix, cal_coordinates):
    cal_distance_matrix = euclidean_distances(cal_coordinates)
    dist_MSE = np.square(og_distance_matrix - cal_distance_matrix).mean()
    # print(dist_MSE)
    return round(dist_MSE, 2)

# Calculate node location MSE in 2D Euclidean space.
def calculate_MSE_loc(og_coordinates, cal_coordinates):
    og_x = [coordinate[0] for coordinate in og_coordinates]
    og_y = [coordinate[1] for coordinate in og_coordinates]
    cal_x = [coordinate[0] for coordinate in cal_coordinates]
    cal_y = [coordinate[1] for coordinate in cal_coordinates]

    sum = 0
    for node_count in range(0, len(og_coordinates)):
        sum = sum + pow((cal_x[node_count] - og_x[node_count]), 2) + pow((cal_y[node_count] - og_y[node_count]), 2) 
    MSE_loc = sum / len(og_coordinates)
    # print("MSE_loc", MSE_loc)
    return round(MSE_loc, 2)
    
# Calculate the inter-node MSE for systems with redundant nodes (E.g: Robust Quads).
def calculate_dist_MSE_redundant(og_coordinates, sort_loc_best):
    # Using Original Coordinates and Sorted Node/Location data from RQ to create a new distance matrix
    # with only coordinates localized by RQ (redundant nodes)
    og_coordinates_redundant = []
    for node_count in range(0, len(sort_loc_best)):
        sort_loc_best_node = sort_loc_best[node_count]
        sort_loc_best_node_count = sort_loc_best_node[0]
        # sort_loc_best_node_coordinate = sort_loc_best_node[1]
        og_coordinates_redundant.append(og_coordinates[sort_loc_best_node_count]) 

    # print("og_coordinates_redundant", og_coordinates_redundant)
    og_distance_matrix_redundant = euclidean_distances(og_coordinates_redundant)
    # Form Calculated Coordinates from the Sorted Node/Location data from RQ
    cal_coordinates = np.array([node[1] for node in sort_loc_best])
    # print("cal_coordinates",cal_coordinates)
    cal_distance_matrix = euclidean_distances(cal_coordinates)
    # print("og_distance_matrix_redundant",og_distance_matrix_redundant)
    dist_MSE = np.square(og_distance_matrix_redundant - cal_distance_matrix).mean()
    # print(dist_MSE)
    return round(dist_MSE, 2)

# Calculate the node-location in 2D euclidean space MSE for systems with redundant nodes (E.g: Robust Quads).
def calculate_loc_MSE_redundant(og_coordinates, sort_loc_best):
    robust_nodes = [node[0] for node in sort_loc_best]
    calc_robust_node_coordinates = [node[1] for node in sort_loc_best]
    robust_node_og_coordinates = []

    for node_count in robust_nodes:
        robust_node_og_coordinates.append(og_coordinates[node_count])

    MSE_loc = calculate_MSE_loc(robust_node_og_coordinates, calc_robust_node_coordinates)
    return round(MSE_loc, 2)

# Calculate average percentage of nodes that were localizable per cluster.
def calculate_ClusterSuccessRate(og_distance_matrix, cal_coordinates, total_clusters):
    Li = len(cal_coordinates)
    ki = len(og_distance_matrix)
    N = total_clusters
    N = 1   # Default = 1 
    # CSR = (1/N) - Summation(N)  - (Li/ki)
    CSR = (Li/ki) * 100
    # print("CSR", CSR)
    return round(CSR, 2)

def trialateration_get_triangle_list(anchor_coordinates):
    anchor_node_count = len(anchor_coordinates)
    
    # Create a list of triangles formed from the anchor nodes.
    node_name = list(range(0, anchor_node_count))
    test_node_comb = list(combinations(node_name,3))

    new_triangles = test_node_comb

    # print("new_triangles", new_triangles)
    return new_triangles


def trilaterate_post_sc(anchor_coordinates, tag_coordinates):
    # Split anchor coordinates into x,y list.
    anchor_x = [node[0] for node in  anchor_coordinates]
    anchor_y = [node[1] for node in  anchor_coordinates]

    # Get list of all triangles formed by anchor nodes
    triangle_list = trialateration_get_triangle_list(anchor_coordinates)

    cal_tag_coordinates = []

    # For each tag node in the system calculate it's coordinates,
    # by averaging the tag coordinates computed from each anchor triangle.
    for tag_node in tag_coordinates:
        tag_node = np.array(tag_node)

        tag_x = []
        tag_y = []

        for triangle in triangle_list:
            # For every triangle decompose it's anchor names
            node0 = triangle[0]
            node1 = triangle[1]
            node2 = triangle[2]

            # Form the tag-triangle distance matrix 
            anchor_tag_coordinates = np.insert(anchor_coordinates, 0, tag_node, axis=0)
            anchor_tag_distance_matrix = euclidean_distances(anchor_tag_coordinates)

            # Calculate squared values needed for trilateration
            r1_sq = pow(anchor_tag_distance_matrix[0,(node0 + 1)],2)
            r2_sq = pow(anchor_tag_distance_matrix[0,(node1 + 1)],2)
            r3_sq = pow(anchor_tag_distance_matrix[0,(node2 + 1)],2)

            # Solve a linear matrix equation where x,y is the Tag coordinate:
            # Ax + By = C
            # Dx + Ey = F
            A = (-2*anchor_x[node0]) + (2*anchor_x[node1])
            B = (-2*anchor_y[node0]) + (2*anchor_y[node1])
            C = r1_sq - r2_sq - pow(anchor_x[node0],2) + pow(anchor_x[node1],2) - pow(anchor_y[node0],2) + pow(anchor_y[node1],2) 
            D = (-2*anchor_x[node1]) + (2*anchor_x[node2])
            E = (-2*anchor_y[node1]) + (2*anchor_y[node2])
            F = r2_sq - r3_sq - pow(anchor_x[node1],2) + pow(anchor_x[node2],2) - pow(anchor_y[node1],2) + pow(anchor_y[node2],2) 

            a = np.array([[A, B], [D, E]])
            b = np.array([C, F])

            # Fixes the following error: numpy.linalg.LinAlgError: Singular matrix
            # numpy.linalg.solve(a, b) Raises LinAlgError If 'a' is singular or not square.
            # This statement checks if matrix 'a' is Invetible(Not Singular == True). 
            if ((a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]) == True):

                tag_coordinates = np.linalg.solve(a, b)
                # print("Tag Coordinate:", tag_coordinates)

                tag_x.append(tag_coordinates[0])
                tag_y.append(tag_coordinates[1])
        
        tag_avg = (mean(tag_x), mean(tag_y)) 
        cal_tag_coordinates.append(tag_avg)

    cal_tag_coordinates = np.array(cal_tag_coordinates)
    # print("cal_tag_coordinatese:", cal_tag_coordinates)

    return cal_tag_coordinates




