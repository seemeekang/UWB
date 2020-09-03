## add_noise_distance_matrix
Add randomly generated noise to an inter-node distance matrix.
### Parameters
- **og_distance_matrix**: The original distance matrix to which noise will be added.
### Return Values
- **noise_og_distance_matrix**: Noisy distance matrix

## calculate_InputAvgNoise
Calculate inter-node avg input noise added to a distance matrix
### Parameters
- **og_distance_matrix**: The original distance matrix.
- **noise_og_distance_matrix**: The distance matrix to which noise has been added.
### Return Values
- **in_noise_avg**: Average inter-node input noise

## calculateMSE
Calculate the inter-node MSE between the Orginal distance matrix and Calculated coordinates.  
### Parameters

- **og_distance_matrix**: The original distance matrix.
- **cal_coordinates**: Calculated coordinates returned after running a self-calibration algorithm. A distance matrix is recalculated from from the calculated coordinates. The MSE is calculated between the original distance matrix and recalculated distance matrix.
### Return Values
- **MSE**: Inter-node MSE between the Orginal distance matrix and calculated coordinates.


