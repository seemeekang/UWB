import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

import math

# Non - Flipped Nodes Example
x_og_data = [0,0,2,2,28,10,100]
y_og_data = [0,2,0,2,35,80,100]

# Flipped Nodes Example
# x_og_data = [0,0,2,2,100]
# y_og_data = [0,2,0,2,100]

# Total nuber of Anchor nodes
node_count = len(x_og_data)

# The Original (Global) Coordinates of the Anchor Nodes.
og_coordinates = list(zip(x_og_data, y_og_data))
print("Original Coordinates \n",og_coordinates)

# Create a Distance Matrix from the Original Coordinates.
og_distance_matrix = euclidean_distances(og_coordinates)
print("Original Distance Matrix \n",og_distance_matrix)

def compute_mds(og_distance_matrix):
    # Intialize MDS parameters:
    # n_components = 2 since the it's a 2-axis graph.
    # dissimilarity = "precomputed". Since the Distance Matrix is precomputed.
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=1,
                    dissimilarity="precomputed", n_jobs=1)

    # Compute the local coordinates from the Distance Matrix using MDS.
    cal_coordinates = mds.fit(og_distance_matrix).embedding_
    print("Calculated Coordinates \n",cal_coordinates)

    # Recompute the Distance Matrix from the Original Distance Matrix
    cal_distance_matrix = euclidean_distances(cal_coordinates)
    print("Calculated Distance Matrix \n",cal_distance_matrix)

    # Calculate the Mean Square Error between the Original Distance Matrix and Recomputed Distance Matrix.
    MSE = np.square(og_distance_matrix - cal_distance_matrix).mean()
    print("Mean Square Error \n",MSE)
    return cal_coordinates, MSE

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
cal_coordinates, MSE = compute_mds(og_distance_matrix)
ax2 = fig.add_subplot(132)
ax2.title.set_text("Calculated MDS \n with Avg Distance Error: " + str(round(MSE,10)))
plt.scatter(cal_coordinates[:, 0], cal_coordinates[:, 1])


# Add noise to the Original Distance matrix
noise = np.random.rand(node_count, node_count)
noise = noise + noise.T
noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
noise_og_distance_matrix = og_distance_matrix + noise
in_noise_avg = np.square(og_distance_matrix - noise_og_distance_matrix).mean()

# Noisy Input Coordinate Plot
cal_coordinates, MSE = compute_mds(noise_og_distance_matrix)
ax2 = fig.add_subplot(133)
ax2.title.set_text("Calculated MDS (Noisy Input) \n with Avg Distance Error: " + str(round(MSE,6)) 
+ "\n Noisy Input Avg: " + str(round(in_noise_avg,6)))
plt.scatter(cal_coordinates[:, 0], cal_coordinates[:, 1])

plt.show()