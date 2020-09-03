# Multidimensional Scaling (MDS) ([skl_classical_MDS.py](https://github.com/jonathanrjpereira/UWB/blob/master/MDS/skl_classical_MDS.py))

This self-calibrating algorithm is discussed in detail here: [Wiki](https://github.com/jonathanrjpereira/UWB/wiki/Multidimensional-Scaling-(MDS))

## compute_mds
Computes coordinates of the all anchor nodes using Multidimensional Scaling.

 1. Compute the inter-node anchor distance matrix.
 2. Calculate MDS according to the [Wiki](https://github.com/jonathanrjpereira/UWB/wiki/Multidimensional-Scaling-(MDS)). In our code we have used the inbuilt [sklearn MDS module](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html). The output will be the anchor coordinates. 


### Parameters
- **og_distance_matrix**: The original anchor inter-node distance matrix. 
### Return Values
- **cal_coordinates**: Coordinates of the anchor nodes.
 
## Standard Modules
Common function descriptions can be found [here](https://github.com/jonathanrjpereira/UWB/tree/master/Standard_Modules).

- **calculate_InputAvgNoise**
- **calculateMSE**
- **add_noise_distance_matrix**
