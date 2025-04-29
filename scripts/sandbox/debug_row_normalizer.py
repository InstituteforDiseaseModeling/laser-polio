import numpy as np
from laser_core.migration import gravity
from laser_core.migration import radiation
from laser_core.migration import row_normalizer

# Example
pops = np.array([99510, 595855, 263884]) / 1e3  # Scale pops down so they don't overflow
dist = np.array([[0, 4, 66], [4, 0, 827], [66, 827, 0]])
k = 100
a = 1
b = 1
c = 2.0
max_migr_frac = 0.3
gnet = gravity(pops, dist, k, a, b, c)  # k * (pop^a * pop^b) / dist^c
rnet = radiation(pops, dist, k, include_home=False)  # Radiation model
print("Gravity Network:\n", gnet)
print("Radiation Network:\n", rnet)

rnet = row_normalizer(rnet, max_migr_frac)  # Normalize rows to sum to max_migr_frac

# ----- Reproduce the issue with row_normalizer -----

network = np.array([[0, 6, 2], [10, 0, 13], [15, 10, 0]])
max_rowsum = 0.3
network = row_normalizer(network, max_migr_frac)  # Normalize rows to sum to max_migr_frac
print("Renormalized Network:\n", network)

# ----- Reproduce the issue manually -----

rowsums = network.sum(axis=1)
rows_to_renorm = rowsums > max_rowsum
network[rows_to_renorm] = network[rows_to_renorm] * max_rowsum / rowsums[rows_to_renorm, np.newaxis]
print("Renormalized Network:\n", network)

# ----- Fix by promoting array to float -----

network = np.array([[0, 6, 2], [10, 0, 13], [15, 10, 0]])
network = network.astype(float)  # Ensure the array is of type float
# network /= 150
max_rowsum = 0.3
print("Unnormalized Network:\n", network)
network = row_normalizer(network, max_migr_frac)  # Normalize rows to sum to max_migr_frac
print("Renormalized Network:\n", network)


rnet * max_migr_frac / np.sum(rnet, axis=1, keepdims=True)


# Normalize rows to sum to 1
row_sums = rnet.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0  # To avoid division by zero
migration_probs = rnet / row_sums


# Katherine's gravity diffusion model
mixing_scale = 0.001
dist_exp = 1.5
np.fill_diagonal(dist, 100000000)  # Prevent divide by zero errors and self migration
diffusion_matrix = pops / (dist + 10) ** dist_exp  # minimum distance prevents excessive neighbor migration
np.fill_diagonal(diffusion_matrix, 0)

# normalize average total outbound migration to 1
diffusion_matrix = diffusion_matrix / np.mean(np.sum(diffusion_matrix, axis=1))

diffusion_matrix *= mixing_scale
diagonal = 1 - np.sum(diffusion_matrix, axis=1)  # normalized outbound migration by source
np.fill_diagonal(diffusion_matrix, diagonal)

print("Done.")
