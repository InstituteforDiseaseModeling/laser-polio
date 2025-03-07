import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy.stats as stats
import ctypes
import numpy.ctypeslib as npct
import time
import laser_polio as lp
from laser_core.migration import gravity, row_normalizer

# Load the shared library
infect_lib = npct.load_library("infect_cdf_binsearch.so", ".")

# Define argument types
infect_lib.parallel_infect.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS"), # nodes
    ctypes.c_int, # num_nodes
    np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS"), # node_id, 
    ctypes.c_int, # num_people
    np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS"), # is_sus
    np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS"), # acq_risk_multiplier
    np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS"), # new_infections
    np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")  # disease_state
]

# Define the wrapper function
def faster_infect(nodes, node_id, is_sus, acq_risk_multiplier, new_infections, disease_state):
    # Call the C function
    infect_lib.parallel_infect(
        nodes.astype(np.int32),
        len(nodes),
        node_id,
        len(node_id),
        is_sus.astype(np.int32),
        acq_risk_multiplier,
        new_infections,
        disease_state
    )

@nb.njit(parallel=True)
def compute_beta_ind_sums(node_ids, daily_infectivity, disease_state, num_nodes):
    num_threads = nb.get_num_threads()

    # Create a thread-local storage (TLS) array for each thread
    beta_sums_tls = np.zeros((num_threads, num_nodes), dtype=np.float64)

    # Each thread works on its own local accumulator
    for i in nb.prange(len(node_ids)):
        if disease_state[i] == 2:  # Only process infected individuals
            thread_id = nb.get_thread_id()
            node = node_ids[i]
            beta_sums_tls[thread_id, node] += daily_infectivity[i]  # Local accumulation

    # Final reduction step: sum up TLS arrays into a single result
    beta_sums = np.zeros(num_nodes, dtype=np.float64)
    for t in range(num_threads):
        for n in range(num_nodes):
            beta_sums[n] += beta_sums_tls[t, n]

    return beta_sums

@nb.njit(parallel=True)
def compute_infections_nb(
    disease_state, node_id, acq_risk_multiplier, beta_per_node
):
    """
    Return an array "exposure_sums" where exposure_sums[node] is the sum of
    probabilities for susceptible individuals in that node.
    """
    num_people = len(disease_state)
    num_nodes = len(beta_per_node)

    # Thread-local storage
    n_threads = nb.get_num_threads()
    local_sums = np.zeros((n_threads, num_nodes), dtype=np.float64)

    # Parallel loop
    for i in nb.prange(num_people):
        if disease_state[i] == 0:  # susceptible
            nd = node_id[i]
            # base probability from node-level infection rate
            prob_infection = beta_per_node[nd] * acq_risk_multiplier[i]
            # Keep a sum of these probabilities
            tid = nb.get_thread_id()
            local_sums[tid, nd] += prob_infection

    # Merge
    exposure_sums = np.zeros(num_nodes, dtype=np.float32)
    for t in range(n_threads):
        for nd in range(num_nodes):
            exposure_sums[nd] += local_sums[t, nd]

    return exposure_sums


class Transmission_ABM:
    def __init__(self, sim):
        self.sim = sim
        self.people = sim.people
        self.nodes = np.arange(len(sim.pars.n_ppl))
        self.pars = sim.pars

        # Calcultate geographic R0 modifiers based on underweight data (one for each node)
        underwt = self.pars.beta_spatial  # Placeholder for now
        self.beta_spatial = 1 / (1 + np.exp(24 * (np.mean(underwt) - underwt))) + 0.2

        # Pre-compute individual risk of acquisition and infectivity with correlated sampling
        # Step 0: Add properties to people
        self.people.add_scalar_property("acq_risk_multiplier", dtype=np.float32, default=1.0)  # Individual-level acquisition risk multiplier (multiplied by base probability for an agent becoming infected)
        self.people.add_scalar_property("daily_infectivity", dtype=np.float32, default=1.0)  # Individual daily infectivity (e.g., number of infections generated per day in a fully susceptible population; mean = R0/dur_inf = 14/24)
        # Step 1: Define parameters for Lognormal & convert to log-space parameters
        mean_lognormal = 1
        variance_lognormal = self.pars.risk_mult_var
        mu_ln = np.log(mean_lognormal**2 / np.sqrt(variance_lognormal + mean_lognormal**2))
        sigma_ln = np.sqrt(np.log(variance_lognormal / mean_lognormal**2 + 1))
        # Step 2: Define parameters for Gamma
        mean_gamma = 14/24
        shape_gamma = 1  # makes this equivalent to an exponential distribution
        scale_gamma = mean_gamma / shape_gamma 
        # Step 3: Generate correlated normal samples
        rho = 0.8  # Desired correlation      
        cov_matrix = np.array([[1, rho], [rho, 1]])  # Create covariance matrix
        L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition
        # Generate standard normal samples
        n_samples = self.people.true_capacity 
        z = np.random.normal(size=(n_samples, 2))
        z_corr = z @ L.T  # Apply Cholesky to introduce correlation
        # Step 4: Transform normal variables into target distributions
        acq_risk_multiplier = np.exp(mu_ln + sigma_ln * z_corr[:, 0])  # Lognormal transformation
        daily_infectivity = stats.gamma.ppf(stats.norm.cdf(z_corr[:, 1]), a=shape_gamma, scale=scale_gamma)  # Gamma transformation
        self.people.acq_risk_multiplier[:self.people.true_capacity] = acq_risk_multiplier
        self.people.daily_infectivity[:self.people.true_capacity] = daily_infectivity

        # Compute the infection migration network
        sim.results.add_vector_property("network", length=len(sim.nodes), dtype=np.float32)
        self.network = sim.results.network
        init_pops = sim.pars.n_ppl
        k, a, b, c = self.pars.gravity_k, self.pars.gravity_a, self.pars.gravity_b, self.pars.gravity_c
        dist_matrix = self.pars.distances
        self.network = gravity(init_pops, dist_matrix, k, a, b, c)
        self.network /= np.power(init_pops.sum(), c)  # Normalize
        self.network = row_normalizer(self.network, self.pars.max_migr_frac)

        self.beta_sum_time = 0
        self.spatial_beta_time = 0
        self.seasonal_beta_time = 0
        self.probs_time = 0
        self.calc_ni_time = 0
        self.do_ni_time = 0


    def step(self):
        # 1) Sum up the total amount of infectivity shed by all infectious agents within a node. 
        # This is the daily number of infections that these individuals would be expected to generate 
        # in a fully susceptible population sans spatial and seasonal factors.
        disease_state = self.people.disease_state[:self.people.count]
        node_ids = self.people.node_id[:self.people.count]
        infectivity = self.people.daily_infectivity[:self.people.count]
        risk = self.people.acq_risk_multiplier[:self.people.count]

        def default_beta():
            is_infected = disease_state == 2
            beta_ind_sums = np.bincount(node_ids[is_infected],
                                        weights=infectivity[is_infected],
                                        minlength=len(self.nodes))
            return beta_ind_sums
        def fast_beta():
            beta_ind_sums = compute_beta_ind_sums(node_ids,
                                  infectivity,
                                  disease_state,
                                  len(self.nodes))
            return beta_ind_sums
        beta_ind_sums = fast_beta()

        check_time = time.perf_counter()
        is_infected = disease_state == 2
        node_beta_sums = np.bincount(node_ids[is_infected], 
                                    weights=infectivity[is_infected], 
                                    minlength=len(self.nodes)).astype(np.float64)
        #new_check_time = time.perf_counter()
        #elapsed = new_check_time - check_time
        #self.beta_sum_time += elapsed
        #check_time = new_check_time
        
        # 2) Spatially redistribute infectivity among nodes
        transfer = (node_beta_sums * self.network).astype(np.float64)  # Don't round here, we'll handle fractional infections later
        # Ensure net contagion remains positive after movement
        node_beta_sums += transfer.sum(axis=1) - transfer.sum(axis=0)
        node_beta_sums = np.maximum(node_beta_sums, 0)  # Prevent negative contagion
        #new_check_time = time.perf_counter()
        #elapsed = new_check_time - check_time
        #self.spatial_beta_time += elapsed
        #check_time = new_check_time

        # 3) Apply seasonal & geographic modifiers
        beta_seasonality = lp.get_seasonality(self.sim)
        beta_spatial = self.pars.beta_spatial  # TODO: This currently uses a placeholder. Update it with IHME underweight data & make the 
        beta = node_beta_sums * beta_seasonality * beta_spatial  # Total node infection rate
        #new_check_time = time.perf_counter()
        #elapsed = new_check_time - check_time
        #self.seasonal_beta_time += elapsed
        #check_time = new_check_time

        # 4) Calculate base probability for each agent to become exposed    
        # Surely the alive count is available from report (sum)?
        #import pdb
        #pdb.set_trace()
        #is_alive = self.people.disease_state >= 0  
        #alive_counts = np.bincount(node_ids[is_alive], minlength=len(self.nodes))  # Count number of alive agents in each node
        alive_counts = self.sim.results.S[self.sim.t] + self.sim.results.E[self.sim.t] + self.sim.results.I[self.sim.t] + self.sim.results.R[self.sim.t]
        per_agent_infection_rate = beta / np.clip(alive_counts, 1, None) 
        base_prob_infection = 1 - np.exp(-per_agent_infection_rate)
        #new_check_time = time.perf_counter()
        #elapsed = new_check_time - check_time
        #self.probs_time += elapsed
        #check_time = new_check_time

        # 5) Calculate infections
        is_sus = disease_state == 0  # Filter to susceptibles
        exposure_sums = compute_infections_nb(
            disease_state,
            node_ids,
            risk,
            base_prob_infection
        )
        new_infections = np.random.poisson(exposure_sums).astype(np.int32)
        #new_check_time = time.perf_counter()
        #elapsed = new_check_time - check_time
        #self.calc_ni_time += elapsed
        #check_time = new_check_time
        #print( f"{n_expected_exposures=}" )

        # 6) Draw n_expected_exposures for each node according to their exposure_probs
        # v1
        def default_infect():
            for node in self.nodes:
                n_to_draw = new_infections[node]
                if n_to_draw > 0:
                    node_sus_indices = np.where((node_ids == node) & is_sus)[0]
                    node_exposure_probs = risk[node_sus_indices]
                    if len(node_sus_indices) > 0:
                        new_exposed_inds = np.random.choice(node_sus_indices, size=n_to_draw, p=node_exposure_probs/node_exposure_probs.sum(), replace=True)
                        new_exposed_inds = np.unique(new_exposed_inds)  # Ensure unique individuals
                        # Mark them as exposed
                        disease_state[new_exposed_inds] = 1

        faster_infect( self.nodes, node_ids, is_sus, risk, new_infections, disease_state )
        #new_check_time = time.perf_counter()
        #elapsed = new_check_time - check_time
        #self.do_ni_time += elapsed
        #check_time = new_check_time

        # # v2
        # if n_expected_exposures.sum() > 0:
        #     new_exposed_indices = assign_exposures(node_ids_sus, exposure_probs[is_sus], n_expected_exposures)

        #     # Assign new state
        #     self.people.disease_state[new_exposed_indices] = 1
        #     self.people.exposure_timer[new_exposed_indices] = self.pars.dur_exp(len(new_exposed_indices))

    def log(self, t):
        pass

    def plot(self, save=False, results_path="" ):
        """
        print( f"{self.beta_sum_time=}" )
        print( f"{self.spatial_beta_time=}" )
        print( f"{self.seasonal_beta_time=}" )
        print( f"{self.probs_time=}" )
        print( f"{self.calc_ni_time=}" )
        print( f"{self.do_ni_time=}" )
        """

