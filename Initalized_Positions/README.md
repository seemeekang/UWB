# Initialized Positions ([ip.py](https://github.com/jonathanrjpereira/UWB/blob/master/Initalized_Positions/ip.py))

This self-calibrating algorithm is discussed in detail here: [Wiki](https://github.com/jonathanrjpereira/UWB/wiki/Initialized-Positions)

**Code Flow:**

 1. Select three anchor nodes as the Intialized nodes. Localize these three Intialized nodes. The anchor nodes are identified by their *number-names*.
 2. Trilaterate remaining anchor nodes.
 3. Run algorithm with noisy input and evaluate results.

## compute_intialized_positions
Computes coordinates of the Initialized nodes and all the remaining anchor-nodes.

 1. Select three anchor nodes as the Intialized nodes. Retrieve the distances between each one of these three nodes.
 2. Localize the Initialized nodes `p0, p1, p2` as described by the formula in the [Wiki](https://github.com/jonathanrjpereira/UWB/wiki/Initialized-Positions). 
 3. Keep a tab of the coordinates of the Initalized nodes. This information will be used to trilaterate the remaining anchor nodes
 4. Trilaterate the remaining anchor nodes.

### Parameters
- **distance_matrix**: The original anchor inter-node distance matrix. 
### Return Values
- **trilaterated_nodes_coordinates**: Coordinates of the Initialized nodes and all the remaining anchor-nodes.


## trilateration
Trilaterate an anchor nodes from the Intialized nodes.

### Parameters
- **node_to_trilaterate**: The anchor node(number-names) to be trilaterated.
- **initial_localized_nodes**: List of Initialized anchor nodes(number-names).
- **initial_localized_coordinates**:  List of coordinate-pairs of the Intialized nodes.
- **distance_matrix**: Inter-node anchor distance matrix.

### Return Values
- **anchor_coordinates**: Trilaterated anchor coordinates.
 
## Standard Modules
Common function descriptions can be found [here](https://github.com/jonathanrjpereira/UWB/tree/master/Standard_Modules).

- **calculate_InputAvgNoise**
- **calculateMSE**
- **add_noise_distance_matrix**
