import numba as nb
import numpy as np
import scipy.stats as stats
from laser_core.migration import gravity
from laser_core.migration import row_normalizer

import laser_polio as lp


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
def compute_infections_nb(disease_state, node_id, acq_risk_multiplier, beta_per_node):
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


@nb.njit(parallel=True)
def fast_infect(node_ids, exposure_probs, disease_state, new_infections):
    """
    A Numba-accelerated version of faster_infect.
    Parallelizes over nodes, computing a CDF for each node's susceptible population.
    Selects 'n_to_draw' indices via binary search of random values, and marks them infected.

    NOTE: This version does NOT enforce uniqueness of selected indices within the same node.
    """
    num_nodes = len(new_infections)
    # Precompute which individuals are susceptible
    is_sus = disease_state == 0
    n_people = len(node_ids)

    for node in nb.prange(num_nodes):
        n_to_draw = new_infections[node]
        if n_to_draw <= 0:
            continue

        # 1) Gather susceptible indices for this node
        count = 0
        for i in range(n_people):
            if node_ids[i] == node and is_sus[i]:
                count += 1

        if count == 0:
            continue

        sus_indices = np.empty(count, dtype=np.int64)
        sus_probs = np.empty(count, dtype=np.float32)

        idx = 0
        previous = 0.0  # build CDF simultaneously
        for i in range(n_people):
            if node_ids[i] == node and is_sus[i]:
                sus_indices[idx] = i
                sus_probs[idx] = previous + exposure_probs[i]
                previous = sus_probs[idx]
                idx += 1

        # 2) Build a CDF in-place in sus_probs
        # for i in range(1, count):
        #     sus_probs[i] += sus_probs[i - 1]

        total = sus_probs[count - 1]
        if total <= 0.0:
            continue

        # don't use prange() here since we're already in a parallelized loop
        # for i in range(count):
        #     sus_probs[i] /= total

        # 3) Draw 'n_to_draw' times via binary search
        for _ in range(n_to_draw):
            r = np.random.uniform(0, total)
            left = 0
            right = count - 1

            while left < right:
                mid = (left + right) // 2
                if sus_probs[mid] < r:
                    left = mid + 1
                else:
                    right = mid

            # Infect the chosen individual
            disease_state[sus_indices[left]] = 1


@nb.njit((nb.int32[:], nb.int32[:], nb.int32[:], nb.int32), nogil=True, cache=True)
def count_SEIRP(node_id, disease_state, paralyzed, n_nodes):
    """
    Go through each person exactly once and increment counters for their node.

    node_id:        array of node IDs for each individual
    disease_state:  array storing each person's disease state (-1=dead/inactive, 0=S, 1=E, 2=I, 3=R)
    paralyzed:      array (0 or 1) if the person is paralyzed
    n_nodes:        total number of nodes

    Returns: S, E, I, R, P arrays, each length n_nodes
    """

    alive = disease_state >= 0  # Only count those who are alive
    S = np.zeros(n_nodes, dtype=np.int64)
    E = np.zeros(n_nodes, dtype=np.int64)
    I = np.zeros(n_nodes, dtype=np.int64)
    R = np.zeros(n_nodes, dtype=np.int64)
    P = np.zeros(n_nodes, dtype=np.int64)

    # Single pass over the entire population
    for i in nb.prange(len(alive)):
        if alive[i]:  # Only count those who are alive
            nd = node_id[i]
            ds = disease_state[i]

            if ds == 0:  # Susceptible
                S[nd] += 1
            elif ds == 1:  # Exposed
                E[nd] += 1
            elif ds == 2:  # Infected
                I[nd] += 1
            elif ds == 3:  # Recovered
                R[nd] += 1

            # Check paralyzed
            if paralyzed[i] == 1:
                P[nd] += 1

    return S, E, I, R, P


class Transmission_ABM:
    def __init__(self, sim):
        self.sim = sim
        self.people = sim.people
        self.nodes = np.arange(len(sim.pars.n_ppl))
        self.pars = sim.pars
        self.results = sim.results

        # Calcultate geographic R0 modifiers based on underweight data (one for each node)
        underwt = self.pars.beta_spatial  # Placeholder for now
        self.beta_spatial = 1 / (1 + np.exp(24 * (np.mean(underwt) - underwt))) + 0.2

        # Pre-compute individual risk of acquisition and infectivity with correlated sampling
        # Step 0: Add properties to people
        self.people.add_scalar_property(
            "acq_risk_multiplier", dtype=np.float32, default=1.0
        )  # Individual-level acquisition risk multiplier (multiplied by base probability for an agent becoming infected)
        self.people.add_scalar_property(
            "daily_infectivity", dtype=np.float32, default=1.0
        )  # Individual daily infectivity (e.g., number of infections generated per day in a fully susceptible population; mean = R0/dur_inf = 14/24)
        # Step 1: Define parameters for Lognormal & convert to log-space parameters
        mean_lognormal = 1
        variance_lognormal = self.pars.risk_mult_var
        mu_ln = np.log(mean_lognormal**2 / np.sqrt(variance_lognormal + mean_lognormal**2))
        sigma_ln = np.sqrt(np.log(variance_lognormal / mean_lognormal**2 + 1))
        # Step 2: Define parameters for Gamma
        mean_gamma = 14 / 24
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
        self.people.acq_risk_multiplier[: self.people.true_capacity] = acq_risk_multiplier
        self.people.daily_infectivity[: self.people.true_capacity] = daily_infectivity

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
        # check_time = time.perf_counter()
        disease_state = self.people.disease_state[: self.people.count]
        node_ids = self.people.node_id[: self.people.count]
        infectivity = self.people.daily_infectivity[: self.people.count]
        risk = self.people.acq_risk_multiplier[: self.people.count]

        def default_beta():
            is_infected = disease_state == 2
            beta_ind_sums = np.bincount(node_ids[is_infected], weights=infectivity[is_infected], minlength=len(self.nodes))
            return beta_ind_sums

        def fast_beta():
            beta_ind_sums = compute_beta_ind_sums(node_ids, infectivity, disease_state, len(self.nodes))
            return beta_ind_sums

        node_beta_sums = fast_beta()

        # new_check_time = time.perf_counter()
        # elapsed = new_check_time - check_time
        # self.beta_sum_time += elapsed
        # check_time = new_check_time

        # 2) Spatially redistribute infectivity among nodes
        transfer = (node_beta_sums * self.network).astype(np.float64)  # Don't round here, we'll handle fractional infections later
        transfer *= 10
        # Ensure net contagion remains positive after movement
        node_beta_sums += transfer.sum(axis=1) - transfer.sum(axis=0)
        node_beta_sums = np.maximum(node_beta_sums, 0)  # Prevent negative contagion
        # new_check_time = time.perf_counter()
        # elapsed = new_check_time - check_time
        # self.spatial_beta_time += elapsed
        # check_time = new_check_time

        # 3) Apply seasonal & geographic modifiers
        beta_seasonality = lp.get_seasonality(self.sim)
        beta_spatial = self.pars.beta_spatial  # TODO: This currently uses a placeholder. Update it with IHME underweight data & make the
        beta = node_beta_sums * beta_seasonality * beta_spatial  # Total node infection rate
        # new_check_time = time.perf_counter()
        # elapsed = new_check_time - check_time
        # self.seasonal_beta_time += elapsed
        # check_time = new_check_time

        # 4) Calculate base probability for each agent to become exposed
        # Surely the alive count is available from report (sum)?
        alive_counts = (
            self.sim.results.S[self.sim.t]
            + self.sim.results.E[self.sim.t]
            + self.sim.results.I[self.sim.t]
            + self.sim.results.R[self.sim.t]
        )
        per_agent_infection_rate = beta / np.clip(alive_counts, 1, None)
        base_prob_infection = 1 - np.exp(-per_agent_infection_rate)
        # new_check_time = time.perf_counter()
        # elapsed = new_check_time - check_time
        # self.probs_time += elapsed
        # check_time = new_check_time

        # 5) Calculate infections
        is_sus = disease_state == 0  # Filter to susceptibles
        exposure_sums = compute_infections_nb(disease_state, node_ids, risk, base_prob_infection)
        new_infections = np.random.poisson(exposure_sums).astype(np.int32)
        # new_check_time = time.perf_counter()
        # elapsed = new_check_time - check_time
        # self.calc_ni_time += elapsed
        # check_time = new_check_time
        # print( f"{n_expected_exposures=}" )

        # 6) Draw n_expected_exposures for each node according to their exposure_probs
        # v1
        def default_infect():
            for node in self.nodes:
                n_to_draw = new_infections[node]
                if n_to_draw > 0:
                    node_sus_indices = np.where((node_ids == node) & is_sus)[0]
                    node_exposure_probs = risk[node_sus_indices]
                    if len(node_sus_indices) > 0:
                        new_exposed_inds = np.random.choice(
                            node_sus_indices, size=n_to_draw, p=node_exposure_probs / node_exposure_probs.sum(), replace=True
                        )
                        new_exposed_inds = np.unique(new_exposed_inds)  # Ensure unique individuals
                        # Mark them as exposed
                        disease_state[new_exposed_inds] = 1

        fast_infect(node_ids, risk, disease_state, new_infections)
        # faster_infect( self.nodes, node_ids, is_sus, risk, new_infections, disease_state )
        # new_check_time = time.perf_counter()
        # elapsed = new_check_time - check_time
        # self.do_ni_time += elapsed
        # check_time = new_check_time

        # # v2
        # if n_expected_exposures.sum() > 0:
        #     new_exposed_indices = assign_exposures(node_ids_sus, exposure_probs[is_sus], n_expected_exposures)

        #     # Assign new state
        #     self.people.disease_state[new_exposed_indices] = 1
        #     self.people.exposure_timer[new_exposed_indices] = self.pars.dur_exp(len(new_exposed_indices))

    def log(self, t):
        # Get the counts for each node in one pass
        S_counts, E_counts, I_counts, R_counts, P_counts = count_SEIRP(
            self.people.node_id,
            self.people.disease_state,
            self.people.paralyzed,
            np.int32(len(self.nodes)),
        )

        # Store them in results
        self.results.S[t, :] = S_counts
        self.results.E[t, :] = E_counts
        self.results.I[t, :] = I_counts
        # Note that we add to existing non-zero EULA values for R
        self.results.R[t, :] += R_counts
        self.results.paralyzed[t, :] = P_counts

    def plot(self, save=False, results_path=""):
        """
        print( f"{self.beta_sum_time=}" )
        print( f"{self.spatial_beta_time=}" )
        print( f"{self.seasonal_beta_time=}" )
        print( f"{self.probs_time=}" )
        print( f"{self.calc_ni_time=}" )
        print( f"{self.do_ni_time=}" )
        """
