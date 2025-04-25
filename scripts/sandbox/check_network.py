import numpy as np

# ----------- Check row sums -------------

# Example
pops = np.array([1000, 800, 500])
network = np.array([[0, 50, 30], [20, 0, 10], [5, 15, 0]])

migration_out_rates = []
stay_home_rates = []

for i in range(len(pops)):
    outflow = np.sum(network[i, :])
    migration_out_rate = outflow / pops[i]
    stay_home_rate = 1 - migration_out_rate
    migration_out_rates.append(migration_out_rate)
    stay_home_rates.append(stay_home_rate)

print("Migration-out rates:", migration_out_rates)
print("Stay-at-home rates:", stay_home_rates)


# Seems like max k is 3ish
# np.any(radiation(init_pops, dist_matrix, 3, include_home=False) < 0)


# ----------- Diff approach to networks from GPT -------------
# Example gravity-based migration matrix (unnormalized)
network = np.array([[0, 5, 10], [2, 0, 3], [1, 4, 0]])

# Zero diagonal (optional)
np.fill_diagonal(network, 0)

# Normalize rows
row_sums = network.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1, row_sums)  # Prevent div by zero
normalized_network = network / row_sums

# Set migration fraction
migration_fraction = 0.1  # 10% leak

# Final transmission matrix
transmission_matrix = (1 - migration_fraction) * np.eye(3) + migration_fraction * normalized_network

print("Transmission matrix:")
print(transmission_matrix)
