import numpy as np
from laser_core.migration import gravity
from laser_core.propertyset import PropertySet

import laser_polio as lp

# ----------- Reproduce negative values manually -------------

pops = np.array([99510, 595855, 263884])
dist = np.array([[0, 4, 66], [4, 0, 827], [66, 827, 0]])
k = 3000
a = 1
b = 1
c = 2.0
i = 0
j = 2
value = k * (pops[i] ** a * pops[j] ** b) / dist[i, j] ** c  # Check one entry
# assert value > 0, "The value should be positive"


# ------ reproduce error in laser_polio -----

pars = PropertySet(
    {
        "n_ppl": np.array([99510, 595855, 263884]),  # Population of each node
        "distances": np.array([[0, 4, 66], [4, 0, 827], [66, 827, 0]]),  # Distance in km between nodes
        "gravity_k": 3000,  # Gravity scaling constant
        "gravity_a": 1,  # Origin population exponent
        "gravity_b": 1,  # Destination population exponent
        "gravity_c": 2.0,  # Distance exponent
    }
)
sim = lp.SEIR_ABM(pars)
sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM]
sim.run()


# ----------- Gravity function with logs -------------


def gravity_log(pops, distances, k=1.0, a=1.0, b=1.0, c=2.0):
    pops = np.asarray(pops)
    distances = np.asarray(distances)

    n = len(pops)
    pops_i = pops[:, np.newaxis]
    pops_j = pops[np.newaxis, :]

    # Avoid division by zero for self-distances
    safe_distances = np.where(distances == 0, 1e-6, distances)

    # Compute in log space
    log_flow = np.log(k) + a * np.log(pops_i) + b * np.log(pops_j) - c * np.log(safe_distances)
    network = np.exp(log_flow)

    # Zero diagonal if you want no self-migration
    np.fill_diagonal(network, 0)

    return network


# Example
pops = np.array([99510, 595855, 263884]) / 1e3  # Scale pops down so they don't overflow
dist = np.array([[0, 4, 66], [4, 0, 827], [66, 827, 0]])
k = 3000
a = 1
b = 1
c = 2.0
max_migr_frac = 0.3
net = gravity(pops, dist, k, a, b, c)  # k * (pop^a * pop^b) / dist^c
net_log = gravity_log(pops, dist, k, a, b, c)  # Log version

assert np.all(net == net_log), "The two methods should yield the same result"

i = 0
j = 2
pops = pops / 1e3
k * (pops[i] ** a * pops[j] ** b) / dist[i, j] ** c  # Check one entry


# Use logs to stablize calculations


# # Do a manual check


# network /= np.power(pops.sum(), c)  # Normalize
# network = row_normalizer(network, max_migr_frac)

# migration_out_rates = []
# stay_home_rates = []

# for i in range(len(pops)):
#     outflow = np.sum(network[i, :])
#     migration_out_rate = outflow / pops[i]
#     stay_home_rate = 1 - migration_out_rate
#     migration_out_rates.append(migration_out_rate)
#     stay_home_rates.append(stay_home_rate)

# print("Migration-out rates:", migration_out_rates)
# print("Stay-at-home rates:", stay_home_rates)


# # Seems like max k is 3ish
# # np.any(radiation(init_pops, dist_matrix, 3, include_home=False) < 0)


# # # ----------- Diff approach to networks from GPT -------------
# # # Example gravity-based migration matrix (unnormalized)
# # network = np.array([[0, 5, 10], [2, 0, 3], [1, 4, 0]])

# # # Zero diagonal (optional)
# # np.fill_diagonal(network, 0)

# # # Normalize rows
# # row_sums = network.sum(axis=1, keepdims=True)
# # row_sums = np.where(row_sums == 0, 1, row_sums)  # Prevent div by zero
# # normalized_network = network / row_sums

# # # Set migration fraction
# # migration_fraction = 0.1  # 10% leak

# # # Final transmission matrix
# # transmission_matrix = (1 - migration_fraction) * np.eye(3) + migration_fraction * normalized_network

# # print("Transmission matrix:")
# # print(transmission_matrix)
