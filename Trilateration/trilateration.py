import math
from sklearn.metrics import euclidean_distances
import numpy as np


def main():
    # Anchor Known Locations
    # anchor = [(1,1),(1,10),(10,10)]
    anchor_x = [1,1,10]
    anchor_y = [1,10,10]

    # Unknown Location of Tag 
    # tag = [(5,5)]
    # tag_x = 5
    # tag_y = 5

    # Locations of all nodes
    anchor_tag = [(5,5),(1,1),(1,10),(10,10)]

    # Assuming we can find the distance between each node using a ToF scheme.
    # For this simulation we will instead, calculate the Euclidean distance for each anchor to the tag
    distance_matrix = euclidean_distances(anchor_tag)
    # print("Euclidean Distances",distance_matrix)

    r1_sq = pow(distance_matrix[0,1],2)
    r2_sq = pow(distance_matrix[0,2],2)
    r3_sq = pow(distance_matrix[0,3],2)

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

if __name__ == "__main__":
    main()