import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

from itertools import combinations  


# Threshold Measurement Noise
dmin = 1.0

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


# Test if all test_nodes are already Robust
def test_node_robustness(test_quad, robust_nodes):
    result =  all(elem in robust_nodes  for elem in test_quad)
    return result

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
    else:
        quad_robustness = False
    return quad_robustness

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
    tag_coordinates = np.linalg.solve(a, b)
    tag_coordinates = np.array(tag_coordinates).tolist()
    # print("Tag Coordinate:", tag_coordinates)
    return tag_coordinates


# Get list of robust nodes with trilaterated coordinates
def trilaterate_robust_nodes(robust_nodes, robust_quads, distance_matrix):
    loc_best = [] #Locsbest
    trilaterated_robust_nodes = []
    trilaterated_robust_nodes_coordinates = []
    initial_localized_coordinates = []

    d_ab = distance_matrix[robust_quads[0][0], robust_quads[0][1]]
    d_ac = distance_matrix[robust_quads[0][0], robust_quads[0][2]]
    d_bc = distance_matrix[robust_quads[0][1], robust_quads[0][2]]


    loc_best.append([robust_quads[0][0],(0,0)])       # p0
    loc_best.append([robust_quads[0][1],(d_ab,0)])    # p1
    
    alpha = (pow(d_ab,2) + pow(d_ac,2) - pow(d_bc,2)) / (2*d_ab*d_ac)
    loc_best.append([robust_quads[0][2],(d_ac*alpha,d_ac*math.sqrt(1-pow(alpha,2)))]) # p2

    trilaterated_robust_nodes = [robust_quads[0][0], robust_quads[0][1], robust_quads[0][2]]
    initial_localized_coordinates = [loc_best[0][1], loc_best[1][1], loc_best[2][1]]
    trilaterated_robust_nodes_coordinates = initial_localized_coordinates
    initial_localized_nodes = trilaterated_robust_nodes
    


    for node in robust_nodes:
        if node not in trilaterated_robust_nodes:
            node_coordinates = trilateration(node, initial_localized_nodes, initial_localized_coordinates, distance_matrix)
            loc_best.append([node, node_coordinates])
            trilaterated_robust_nodes.append(node)
            trilaterated_robust_nodes_coordinates.append(node_coordinates)

    # print(initial_localized_coordinates)    
    print("Loc_best", loc_best)
    # print("trilaterated_robust_nodes", trilaterated_robust_nodes)
    # print("trilaterated_robust_nodes_coordinates", trilaterated_robust_nodes_coordinates)

    return np.array(trilaterated_robust_nodes_coordinates)


# Node Name List Creation
node_name = list(range(0, node_count))
test_node_comb = list(combinations(node_name,4))
test_quad_name = []
for test_quad in test_node_comb:
    if test_quad[0] == 0:
        test_quad_name.append(test_quad)
print(test_quad_name)

# Create list of Robust Nodes and Robust Quads
robust_nodes = []
robust_quads = []

for test_quad in test_quad_name:
    if test_node_robustness(test_quad, robust_nodes) == True:
        continue
    else:
        quad_robust_check = test_quad_robustness(test_quad, og_distance_matrix, dmin)

    if quad_robust_check == True:
        # Append the Robust Nodes
        robust_nodes.extend(test_quad)
        # Remove duplicate Robust Nodes
        robust_nodes = list(set(robust_nodes)) 
        # Append the Robust Quad
        robust_quads.append(test_quad)

print("Robust Nodes", robust_nodes)
print("Robust Quads", robust_quads)

trilaterated_robust_node_data = trilaterate_robust_nodes(robust_nodes, robust_quads, og_distance_matrix)
print("trilaterated_robust_node_data", (trilaterated_robust_node_data))


# Plot the two graphs
fig = plt.figure()

# Original Coordinate Plot
ax1 = fig.add_subplot(131)
ax1.title.set_text('Original')
og_coordinates = [list(ele) for ele in og_coordinates]
og_coordinates = np.array(og_coordinates)
# print(og_coordinates) 
plt.scatter(og_coordinates[:, 0], og_coordinates[:, 1])

ax2 = fig.add_subplot(132)
ax2.title.set_text("Robust Node Quads")
plt.scatter(og_coordinates[:, 0], og_coordinates[:, 1], c='red')
plt.scatter(og_coordinates[robust_nodes, 0], og_coordinates[robust_nodes, 1], c='green')

ax2 = fig.add_subplot(133)
ax2.title.set_text("Trilaterated Robust Nodes")
plt.scatter(trilaterated_robust_node_data[:,0], trilaterated_robust_node_data[:,1], c='green')


plt.show()