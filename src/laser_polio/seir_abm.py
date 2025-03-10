import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from laser_core.laserframe import LaserFrame
from laser_core.migration import gravity, row_normalizer
from laser_core.utils import calc_capacity
import datetime as dt
import pandas as pd
from pathlib import Path
import scipy.stats as stats
from alive_progress import alive_bar
import time
import laser_polio as lp
from laser_core.demographics.pyramid import load_pyramid_csv, AliasedDistribution
from laser_core.demographics.kmestimator import KaplanMeierEstimator

__all__ = ['SEIR_ABM', 'DiseaseState_ABM', 'Transmission_ABM', 'VitalDynamics_ABM', 'RI_ABM', 'SIA_ABM']

# SEIR Model
class SEIR_ABM:
    '''
    An AGENT-BASED SEIR Model for polio
    Each entry in the population is an agent with a disease state and a node ID
    Disease state codes: 0=S, 1=E, 2=I, 3=R
    '''

    def __init__(self, pars):
        self.pars = pars       
        pars = self.pars
        self.t = 0
        self.dates = lp.daterange(self.pars['start_date'], days=self.pars.timesteps)

        # Initialize the population
        pars.n_ppl = np.atleast_1d(pars.n_ppl).astype(int)  # Ensure pars.n_ppl is an array
        if (pars.cbr is not None) & (len(pars.cbr) == 1):
            capacity = int(1.1*calc_capacity(np.sum(pars.n_ppl), pars.timesteps, pars.cbr))
        elif (pars.cbr is not None) & (len(pars.cbr) > 1):
            capacity = int(1.1*calc_capacity(np.sum(pars.n_ppl), pars.timesteps, np.mean(pars.cbr)))
        else:
            capacity = int(np.sum(pars.n_ppl))      
        self.people = LaserFrame(capacity=capacity, initial_count=int(np.sum(pars.n_ppl)))
        # We initialize disease_state here since it's required for most other components (which facilitates testing)
        self.people.add_scalar_property("disease_state", dtype=np.int32, default=-1)  # -1=Dead/inactive, 0=S, 1=E, 2=I, 3=R
        self.people.disease_state[:np.sum(self.pars.n_ppl)] = 0  # Set initial population as susceptible
        self.results = LaserFrame(capacity=1)

        # Setup spatial component with node IDs
        self.nodes = np.arange(len(np.atleast_1d(pars.n_ppl)))
        self.people.add_scalar_property("node_id", dtype=np.int32, default=0)
        node_ids = np.concatenate([np.full(count, i) for i, count in enumerate(pars.n_ppl)])
        self.people.node_id[0:np.sum(pars.n_ppl)] = node_ids  # Assign node IDs to initial people
        self.results.add_array_property("node_pop", shape=(pars.timesteps, len(self.nodes)), dtype=np.int32)

        # Components
        self.components = []

    @property
    def components(self) -> list:
        """
        Retrieve the list of model components.

        Returns:

            list: A list containing the components.
        """

        return self._components

    @components.setter
    def components(self, components: list) -> None:
        """
        Sets up the components of the model and initializes instances and phases.

        This function takes a list of component types, creates an instance of each, and adds each callable component to the phase list.
        It also registers any components with an `on_birth` function with the `Births` component.

        Args:

            components (list): A list of component classes to be initialized and integrated into the model.

        Returns:

            None
        """

        self._components = components
        self.instances = []  # instantiated instances of components
        for component in components:
            instance = component(self)
            self.instances.append(instance)

    def run(self):
        self.component_times = { component: 0.0 for component in self.instances }
        self.component_times["report"] = 0
        with alive_bar(self.pars.timesteps, title='Simulation progress:') as bar:
            for tick in range(self.pars.timesteps):
                for component in self.instances:
                    start_time = time.perf_counter()
                    component.step()
                    end_time = time.perf_counter()
                    self.component_times[component] += (end_time-start_time)

                start_time = time.perf_counter()
                self.log_results(tick)
                end_time = time.perf_counter()
                self.component_times["report"] += (end_time-start_time)
                self.t += 1
                bar()  # Update the progress bar

    def log_results(self, t):
        for component in self.instances:
            component.log(t)
    
    def plot(self, save=False, results_path=None):
        if save:
            plt.ioff()  # Turn off interactive mode
            if results_path is None:
                raise ValueError("Please provide a results path to save the plots.")
            else:
                results_path = Path(results_path)  # Ensure results_path is a Path object
                results_path.mkdir(parents=True, exist_ok=True)
        for component in self.instances:
            component.plot(save=save, results_path=results_path)
        self.plot_node_pop(save=save, results_path=results_path)

        if self.component_times:
            labels = [component.__class__.__name__ for component in self.instances]
            labels.append( "report" )
            print( f"{self.instances=}" )
            print( f"{labels=}" )
            #times = [self.component_times[component] for component in labels ]
            times = [self.component_times[component] for component in self.instances ]
            times.append( self.component_times["report"] )# hack
            plt.figure(figsize=(6, 6))
            plt.pie(times, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
            plt.title("Time Spent in Each Component")
            if save:
                plt.savefig(results_path / "perfpie.png")
            if not save:
                plt.show()

    def plot_node_pop(self, save=False, results_path=None):
        plt.figure(figsize=(10, 6))
        for node in self.nodes:
            pop = self.results.S[:, node] + self.results.E[:, node] + self.results.I[:, node] + self.results.R[:, node]
            plt.plot(pop, label=f"Node {node}")
        plt.title("Node Population")
        plt.xlabel("Time (Timesteps)")
        plt.ylabel("Population")
        plt.grid()
        if save:
            plt.savefig(results_path / "node_population.png")
        if not save:
            plt.show()


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

    alive = disease_state >= 0 # Only count those who are alive
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

            if ds == 0:   # Susceptible
                S[nd] += 1
            elif ds == 1: # Exposed
                E[nd] += 1
            elif ds == 2: # Infected
                I[nd] += 1
            elif ds == 3: # Recovered
                R[nd] += 1

            # Check paralyzed
            if paralyzed[i] == 1:
                P[nd] += 1

    return S, E, I, R, P


@nb.njit(parallel=True)
def step_nb(disease_state, exposure_timer, infection_timer, acq_risk_multiplier, daily_infectivity, paralyzed, p_paralysis):
    for i in nb.prange(disease_state.size):
        if disease_state[i] == 1:  # Exposed -> Infected
            exposure_timer[i] -= 1
            if exposure_timer[i] <= 0:
                disease_state[i] = 2  # Become infected
                # infection_timer already preset for everyone

                # Apply paralysis probability immediately after infection
                if np.random.random() < p_paralysis:
                    paralyzed[i] = 1

        elif disease_state[i] == 2:  # Infected -> Recovered
            infection_timer[i] -= 1
            if infection_timer[i] <= 0:
                disease_state[i] = 3  # Recovered
                acq_risk_multiplier[i] = 0.0  # Reset risk
                daily_infectivity[i] = 0.0  # Reset infectivity


class DiseaseState_ABM:
    def __init__(self, sim):
        self.sim = sim
        self.people = sim.people
        self.pars = sim.pars
        self.nodes = sim.nodes
        self.results = sim.results

        # Setup the SEIR components
        pars = self.pars
        sim.people.add_scalar_property("paralyzed", dtype=np.int32, default=0)
        sim.people.add_scalar_property("exposure_timer", dtype=np.int32, default=0)
        # should probably set for entire population, not just initial, but giving issues. TBD.
        sim.people.exposure_timer[:np.sum(self.pars.n_ppl)] = self.pars.dur_exp(np.sum(self.pars.n_ppl)) # initialize all agents with an infection_timer
        sim.people.add_scalar_property("infection_timer", dtype=np.int32, default=0)
        sim.people.infection_timer[:np.sum(self.pars.n_ppl)] = self.pars.dur_inf(np.sum(self.pars.n_ppl)) # initialize all agents with an infection_timer
        sim.results.add_array_property("S", shape=(pars.timesteps, len(self.nodes)), dtype=np.float32)
        sim.results.add_array_property("E", shape=(pars.timesteps, len(self.nodes)), dtype=np.float32)
        sim.results.add_array_property("I", shape=(pars.timesteps, len(self.nodes)), dtype=np.float32)
        sim.results.add_array_property("R", shape=(pars.timesteps, len(self.nodes)), dtype=np.float32)
        sim.results.add_array_property("paralyzed", shape=(pars.timesteps, len(self.nodes)), dtype=np.float32)

        # Initialize immunity
        if isinstance(pars.init_immun, float):
            # Initialize across total population
            num_recovered = int(sum(pars.n_ppl) * pars.init_immun)
            recovered_indices = np.random.choice(sum(pars.n_ppl), size=num_recovered, replace=False)
            sim.people.disease_state[recovered_indices] = 3
        elif isinstance(pars.init_immun, list) and len(pars.init_immun) == 1:
            # Initialize across total population
            num_recovered = int(sum(pars.n_ppl) * pars.init_immun[0])
            recovered_indices = np.random.choice(sum(pars.n_ppl), size=num_recovered, replace=False)
            sim.people.disease_state[recovered_indices] = 3
        else:
            # Initialize by node
            # Extract age bins dynamically from column names
            age_bins = {}
            for col in pars.init_immun.columns:
                if col.startswith('immunity_'):
                    _, min_age, max_age = col.split('_')
                    min_age_days, max_age_days = int(min_age) * 30.43, (int(max_age) + 1) * 30.43  # We add one here b/c the max is exclusive. See filtering logic below for who is considered eligible. 
                    age_bins[col] = (min_age_days, max_age_days)
                    # Assign recovered status based on immunity data
            for node in self.nodes:
                alive_in_node = (self.people.node_id == node) & (self.people.disease_state >= 0)
                for age_key, (age_min, age_max) in age_bins.items():
                    immune_frac = pars.init_immun.iloc[node][age_key]
                    eligible = (sim.people.date_of_birth[alive_in_node] * -1 >= age_min) & (sim.people.date_of_birth[alive_in_node] * -1 < age_max)
                    exp_n_recovered = sum(eligible) * immune_frac
                    n_recovered = np.minimum(np.random.poisson(exp_n_recovered), sum(eligible))
                    eligible_indices = np.where(alive_in_node)[0][eligible]
                    recovered_indices = np.random.choice(eligible_indices, size=n_recovered, replace=False)
                    # Set as recovered
                    sim.people.disease_state[recovered_indices] = 3
            # Assume everyone older than 15 years of age is immune
            o15 = (sim.people.date_of_birth * -1) >= age_max
            sim.people.disease_state[o15] = 3  # Set as recovered
            
        # Seed infections - (potentially overwrites immunity, e.g., if an individual is drawn as both immune (during immunity initialization above) and infected (below), they will be infected)
        if isinstance(pars.init_prev, float):
            num_infected = int(sum(pars.n_ppl) * pars.init_prev)
            infected_indices = np.random.choice(sum(pars.n_ppl), size=num_infected, replace=False)
        elif isinstance(pars.init_prev, list) and len(pars.init_prev) == 1:
            num_infected = int(sum(pars.n_ppl) * pars.init_prev[0])
            infected_indices = np.random.choice(sum(pars.n_ppl), size=num_infected, replace=False)
        else:
            # Seed infections by node
            infected_indices = []
            for node, prev in enumerate(pars.init_prev):
                num_infected = int(pars.n_ppl[node] * prev)
                alive_in_node = (self.people.node_id == node) & (self.people.disease_state >= 0)
                infected_indices_node = np.random.choice(np.where(alive_in_node)[0], size=num_infected, replace=False)
                infected_indices.extend(infected_indices_node)
        num_infected = len(infected_indices)
        sim.people.disease_state[infected_indices] = 2
        sim.people.infection_timer[infected_indices] = self.pars.dur_inf(num_infected)

    def step(self):

        # Add these if they don't exist from the Transmission_ABM component (e.g., if running DiseaseState_ABM alone for testing)
        if not hasattr(self.people, "acq_risk_multiplier"): 
            self.people.add_scalar_property("acq_risk_multiplier", dtype=np.float32, default=1.0)
        if not hasattr(self.people, "daily_infectivity"):
            self.people.add_scalar_property("daily_infectivity", dtype=np.float32, default=1.0)

        # Do nothing. Susceptibility is lost in the Transmission component.
        step_nb(
            self.people.disease_state,
            self.people.exposure_timer,
            self.people.infection_timer,
            self.people.acq_risk_multiplier,
            self.people.daily_infectivity,
            self.people.paralyzed,
            self.pars.p_paralysis
        )

    def log(self, t):
        # Get the counts for each node in one pass
        S_counts, E_counts, I_counts, R_counts, P_counts = count_SEIRP(
            self.people.node_id,
            self.people.disease_state,
            self.people.paralyzed,
            np.int32(len(self.nodes)),
        )

        # Store them in results
        self.results.S[t, :]         = S_counts
        self.results.E[t, :]         = E_counts
        self.results.I[t, :]         = I_counts
        self.results.R[t, :]         = R_counts
        self.results.paralyzed[t, :] = P_counts

    def plot(self, save=False, results_path=None):
        self.plot_total_seir_counts(save=save, results_path=results_path)
        self.plot_infected_by_node(save=save, results_path=results_path)
        self.plot_infected_map(save=save, results_path=results_path)

    def plot_total_seir_counts(self, save=False, results_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(np.sum(self.results.S, axis=1), label="Susceptible (S)")
        plt.plot(np.sum(self.results.E, axis=1), label="Exposed (E)")
        plt.plot(np.sum(self.results.I, axis=1), label="Infectious (I)")
        plt.plot(np.sum(self.results.R, axis=1), label="Recovered (R)")
        plt.plot(np.sum(self.results.paralyzed, axis=1), label="Paralyzed")
        plt.title("SEIR Dynamics in Total Population")
        plt.xlabel("Time (Timesteps)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid()
        plt.savefig( "total_seir.png" )
        if save:
            plt.savefig(results_path / "total_seir_counts.png")
        if not save:
            plt.show()

    def plot_infected_by_node(self, save=False, results_path=None):
        plt.figure(figsize=(10, 6))
        for node in self.nodes:
            plt.plot(self.results.I[:, node], label=f"Node {node}")
        plt.title("Infected Population by Node")
        plt.xlabel("Time (Timesteps)")
        plt.ylabel("Population")
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(results_path / "n_infected_by_node.png")
        if not save:
            plt.show()

    def plot_infected_map(self, save=False, results_path=None, n_panels=6):
        timepoints = np.linspace(0, self.pars.timesteps - 1, n_panels, dtype=int)
        
        rows, cols = 2, int(np.ceil(n_panels / 2))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), sharex=True, sharey=True)
        axs = axs.ravel()  # Flatten in case of non-square grid

        lats, lons = self.pars.centroids["CENTER_LAT"], self.pars.centroids["CENTER_LON"]
        
        # Get global min and max for consistent color scale
        infection_min = np.min(self.results.I)
        infection_max = np.max(self.results.I)
        
        for i, ax in enumerate(axs[:n_panels]):  # Ensure we don't go out of bounds
            t = timepoints[i]
            infection_counts = self.results.I[t, :]

            scatter = ax.scatter(lons, lats, c=infection_counts, cmap="Reds", edgecolors=None, alpha=0.9, 
                                vmin=infection_min, vmax=infection_max)
            ax.set_title(f"Timepoint {t}")
            
            # Show labels only on the leftmost and bottom plots
            if i % cols == 0:
                ax.set_ylabel("Latitude")
            else:
                ax.set_yticklabels([])
            
            if i >= n_panels - cols:
                ax.set_xlabel("Longitude")
            else:
                ax.set_xticklabels([])
        
        # Add a single colorbar for all plots
        if 'scatter' in locals():  # Ensure scatter was created successfully
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position colorbar next to subplots
            fig.colorbar(scatter, cax=cbar_ax, label="Infection Count")
        # Add title
        fig.suptitle("Infected Population by Node", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Pad the top and right for title and colorbar

        if save:
            if results_path is None:
                raise ValueError("Please provide a results path to save the plots.")
            plt.savefig(f"{results_path}/infected_map.png")
        else:
            plt.show()

# @nb.njit
# def assign_exposures(node_ids, exposure_probs, expected_exposures):
#     """
#     Assign exposures to susceptible individuals using a vectorized multinomial sampling approach.
    
#     Parameters:
#     - node_ids: Array of node IDs for susceptible individuals
#     - exposure_probs: Normalized infection probabilities per susceptible individual
#     - expected_exposures: Expected number of infections per node from Poisson sampling
    
#     Returns:
#     - Array of indices of newly exposed individuals
#     """

#     # Cumulative sum to get sorting order
#     sorted_indices = np.argsort(node_ids)
#     sorted_nodes = node_ids[sorted_indices]
#     sorted_probs = exposure_probs[sorted_indices]

#     # Compute the multinomial draws for each node
#     node_counts = np.bincount(sorted_nodes, minlength=len(expected_exposures))
#     total_to_infect = expected_exposures.sum()

#     # Sample from multinomial to get number of infections per susceptible
#     infected_counts = np.random.multinomial(total_to_infect, sorted_probs / sorted_probs.sum())

#     # Select exposed individuals efficiently using cumulative sum
#     exposure_cumsum = np.cumsum(infected_counts)
#     new_exposed_indices = sorted_indices[:exposure_cumsum[-1]]

#     return new_exposed_indices

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
        n_samples = self.people.capacity 
        z = np.random.normal(size=(n_samples, 2))
        z_corr = z @ L.T  # Apply Cholesky to introduce correlation
        # Step 4: Transform normal variables into target distributions
        acq_risk_multiplier = np.exp(mu_ln + sigma_ln * z_corr[:, 0])  # Lognormal transformation
        daily_infectivity = stats.gamma.ppf(stats.norm.cdf(z_corr[:, 1]), a=shape_gamma, scale=scale_gamma)  # Gamma transformation
        self.people.acq_risk_multiplier[:] = acq_risk_multiplier
        self.people.daily_infectivity[:] = daily_infectivity

        # Compute the infection migration network
        sim.results.add_vector_property("network", length=len(sim.nodes), dtype=np.float32)
        self.network = sim.results.network
        init_pops = sim.pars.n_ppl
        k, a, b, c = self.pars.gravity_k, self.pars.gravity_a, self.pars.gravity_b, self.pars.gravity_c
        dist_matrix = self.pars.distances
        self.network = gravity(init_pops, dist_matrix, k, a, b, c)
        self.network /= np.power(init_pops.sum(), c)  # Normalize
        self.network = row_normalizer(self.network, self.pars.max_migr_frac)
        
    def step(self):
        # 1) Sum up the total amount of infectivity shed by all infectious agents within a node. 
        # This is the daily number of infections that these individuals would be expected to generate 
        # in a fully susceptible population sans spatial and seasonal factors.
        is_infected = self.people.disease_state == 2
        node_beta_sums = np.bincount(self.people.node_id[is_infected], 
                                    weights=self.people.daily_infectivity[is_infected], 
                                    minlength=len(self.nodes)).astype(np.float64)
        
        # 2) Spatially redistribute infectivity among nodes
        transfer = (node_beta_sums * self.network).astype(np.float64)  # Don't round here, we'll handle fractional infections later
        # Ensure net contagion remains positive after movement
        node_beta_sums += transfer.sum(axis=1) - transfer.sum(axis=0)
        node_beta_sums = np.maximum(node_beta_sums, 0)  # Prevent negative contagion

        # 3) Apply seasonal & geographic modifiers
        beta_seasonality = lp.get_seasonality(self.sim)
        beta_spatial = self.beta_spatial
        beta = node_beta_sums * beta_seasonality * beta_spatial  # Total node infection rate

        # 4) Calculate base probability for each agent to become exposed    
        is_alive = self.people.disease_state >= 0  
        alive_counts = np.bincount(self.people.node_id[is_alive], minlength=len(self.nodes))  # Count number of alive agents in each node
        per_agent_infection_rate = beta / np.clip(alive_counts, 1, None) 
        base_prob_infection = 1 - np.exp(-per_agent_infection_rate)

        # 5) Calculate infections
        is_sus = self.people.disease_state == 0  # Filter to susceptibles     
        exposure_probs = base_prob_infection[self.people.node_id] * self.people.acq_risk_multiplier  # Multiply by individual risk multiplier
        node_ids_sus = self.people.node_id[is_sus]
        exposure_prob_sums = np.bincount(node_ids_sus, 
                                    weights=exposure_probs[is_sus], 
                                    minlength=len(self.nodes))
        n_expected_exposures = np.random.poisson(exposure_prob_sums)
        #print( f"{n_expected_exposures=}" )

        # 6) Draw n_expected_exposures for each node according to their exposure_probs
        # v1
        for node in self.nodes:
            n_to_draw = n_expected_exposures[node]
            if n_to_draw > 0:
                node_sus_indices = np.where((self.people.node_id == node) & is_sus)[0]
                node_exposure_probs = exposure_probs[node_sus_indices]
                if len(node_sus_indices) > 0:
                    new_exposed_inds = np.random.choice(node_sus_indices, size=n_to_draw, p=node_exposure_probs/node_exposure_probs.sum(), replace=True)
                    new_exposed_inds = np.unique(new_exposed_inds)  # Ensure unique individuals
                    # Mark them as exposed
                    self.people.disease_state[new_exposed_inds] = 1
                    #self.people.exposure_timer[new_exposed_inds] = self.pars.dur_exp(len(new_exposed_inds))

        # # v2
        # if n_expected_exposures.sum() > 0:
        #     new_exposed_indices = assign_exposures(node_ids_sus, exposure_probs[is_sus], n_expected_exposures)

        #     # Assign new state
        #     self.people.disease_state[new_exposed_indices] = 1
        #     self.people.exposure_timer[new_exposed_indices] = self.pars.dur_exp(len(new_exposed_indices))

    def log(self, t):
        pass

    def plot(self, save=False, results_path=None):
        pass


    # def select_exposure_indices(self, indices, exposure_probs, expected_new_exposures):
    #     """
    #     Efficiently selects individuals for exposure using the inverse transform method using a geometric distribution, 
    #     which allows you to efficiently sample a fixed number of individuals without iterating over the entire population. 
    #     This approach is based on skipping over agents efficiently rather than evaluating each agents probability one by one.

    #     Instead of looping over all agents and sampling individually, we:
    #       1. Sort the agents by their exposure probability (optional, but improves efficiency).
    #       2. Use a geometric-like trick to jump directly to the next exposed agent.
    #       3. Mathematically determine skips using the inverse CDF of the exponential distribution.
        
    #     Parameters:
    #     - indices: indices of the agents to consider for exposure
    #     - exposure_probs: exposure probabilities for each agent
    #     - expected_new_exposures: int -> Number of exposures to generate
        
    #     Returns:
    #     - exposed_indices: List of selected agent indices
    #     """
        
    #     # Sort individuals by infection probability (optional but can improve performance)
    #     sorted_indices = np.argsort(-exposure_probs)  # Sort descending
    #     sorted_probs = exposure_probs[sorted_indices]
    #     sorted_agents = indices[sorted_indices]  # Get sorted agent IDs
        
    #     # Efficient selection using geometric-like skips
    #     exposed_indices = []
    #     u = np.random.uniform(0, 1, expected_new_exposures)  # Generate uniform random values

    #     position = 0  # Start at first agent
    #     for x in range(expected_new_exposures):
    #         while position < len(sorted_agents):
    #             denominator = np.log(1 - sorted_probs[position])
    #             if denominator == 0 or np.isnan(denominator):
    #                 skip_distance = 1  # Default to moving at least one step
    #             else:
    #                 skip_distance = int(np.log(u[x]) / denominator)  # Compute jump
    #             # skip_distance = int(np.log(u[x]) / np.log(1 - sorted_probs[position]))  # Compute jump
    #             position += max(1, skip_distance)  # Move to the next agent
    #             if position < len(sorted_agents):
    #                 exposed_indices.append(sorted_agents[position])
    #                 position += 1  # Move to the next position
    #             else:
    #                 break  # Stop if we run out of agents

    #     return exposed_indices


class VitalDynamics_ABM:
    def __init__(self, sim, step_size=7):
        self.sim = sim
        self.people = sim.people
        self.nodes = sim.nodes
        self.results = sim.results
        self.step_size = step_size  # Number of days between vital dynamics steps

        # Setup the age and vital rate components
        pars = sim.pars
        if pars.age_pyramid_path is not None:
            sim.people.add_scalar_property("date_of_birth", dtype=np.int32, default=-1)
            pyramid = load_pyramid_csv(pars.age_pyramid_path)
            MINCOL = 0
            MAXCOL = 1 
            MCOL = 2
            FCOL = 3          
            sampler = AliasedDistribution(pyramid[:, MCOL] + pyramid[:, FCOL]) # using the male population in this example
            samples = sampler.sample(len(sim.people))
            bin_min_age_days = pyramid[:, MINCOL] * 365          # minimum age for bin, in days (include this value)
            bin_max_age_days = (pyramid[:, MAXCOL] + 1) * 365    # maximum age for bin, in days (exclude this value)
            mask = np.zeros(len(sim.people), dtype=bool)
            ages = np.zeros(len(sim.people), dtype=np.int32)
            for i in range(len(pyramid)):   # for each possible bin value...
                mask[:] = samples == i      # ...find the agents that belong to this bin
                # ...and assign a random age, in days, within the bin
                ages[mask] = np.random.randint(bin_min_age_days[i], bin_max_age_days[i], mask.sum())
            sim.people.date_of_birth[:len(sim.people)] = -ages

        if pars.cbr is not None:
            sim.results.add_array_property("births", shape=(pars.timesteps, len(sim.nodes)), dtype=np.int32)
            sim.results.add_array_property("deaths", shape=(pars.timesteps, len(sim.nodes)), dtype=np.int32)
            sim.people.add_scalar_property("date_of_death", dtype=np.int32, default=0)

            cumulative_deaths = lp.create_cumulative_deaths(np.sum(pars.n_ppl), max_age_years=100)
            sim.death_estimator = KaplanMeierEstimator(cumulative_deaths)          
            lifespans = sim.death_estimator.predict_age_at_death(ages, max_year=100)
            dods = lifespans - ages # we could check that dods is non-negative to be safe
            sim.people.date_of_death[:np.sum(pars.n_ppl)] = dods

        for node in self.nodes:
            if len(pars.cbr) == 1:
                self.birth_rate = pars.cbr / (365 * 1000)
            else:
                self.birth_rate = pars.cbr[node] / (365 * 1000) # Birth rate per day per person

        self.death_estimator = sim.death_estimator

    def step(self):
        t = self.sim.t
        if t % self.step_size != 0:
            return

        pars = self.sim.pars

        # 1) Vectorized mask of all alive people
        alive = (self.people.disease_state >= 0)

        # 2) Count how many alive in each node in one pass
        node_ids_alive = self.people.node_id[alive]
        alive_count_by_node = np.bincount(node_ids_alive, minlength=len(self.nodes))

        # 3) Compute births node by node, but without big boolean masks
        for node in self.nodes:
            expected_births = 7 * self.birth_rate * alive_count_by_node[node]

            # Integer part plus probabilistic fractional part
            birth_integer = int(expected_births)
            birth_fraction = expected_births - birth_integer
            birth_rand = np.random.binomial(1, birth_fraction)  # Bernoulli draw
            births = birth_integer + birth_rand

            # If births occur, add them to the population
            if births > 0:
                start, end = self.people.add(births)

                newborn_ages = np.zeros(births, dtype=np.int32)
                lifespans = self.death_estimator.predict_age_at_death(
                    newborn_ages, max_year=100
                )

                self.people.date_of_birth[start:end] = t
                self.people.disease_state[start:end] = 0
                self.people.date_of_death[start:end] = lifespans + t
                self.people.node_id[start:end] = node
                if any(isinstance(component, RI_ABM) for component in self.sim.components):
                    self.people.ri_timer[start:end] = 182

                self.results.births[t, node] = births

        # 4) Now handle deaths, again in a vectorized way
        #    People die if they're alive and their date_of_death <= t
        dying = alive & (self.people.date_of_death <= t)

        # Count how many are dying in each node
        node_ids_dying = self.people.node_id[dying]
        deaths_count_by_node = np.bincount(node_ids_dying, minlength=len(self.nodes))

        # Mark them dead
        self.people.disease_state[dying] = -1

        # 5) Store the death counts
        for node in self.nodes:
            self.results.deaths[t, node] = deaths_count_by_node[node]

    def log(self, t):
        pass

    def plot(self, save=False, results_path=None):
        self.plot_age_pyramid(save=save, results_path=results_path)
        self.plot_vital_dynamics(save=save, results_path=results_path)

    def plot_age_pyramid(self, save=False, results_path=None):
        
        # Expected age distribution
        pars = self.sim.pars
        exp_ages = pd.read_csv(pars.age_pyramid_path)
        exp_ages["Total"] = exp_ages["M"] + exp_ages["F"]
        exp_ages["Proportion"] = exp_ages["Total"] / exp_ages["Total"].sum()

        # Observed age distribution
        obs_ages = ((self.people.date_of_birth * -1) + self.sim.t) / 365
        pyramid = load_pyramid_csv(pars.age_pyramid_path)
        bins = pyramid[:, 0] 
        # Add 105+ bin
        bins = np.append(bins, 105)
        age_bins = pd.cut(obs_ages, bins=bins, right=False)
        age_bins.value_counts().sort_index()
        obs_age_distribution = age_bins.value_counts().sort_index()
        obs_age_distribution = obs_age_distribution / obs_age_distribution.sum()  # Normalize

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x_labels = exp_ages["Age"]
        x = np.arange(len(x_labels))
        ax.plot(x, exp_ages["Proportion"], label="Expected", color="green")
        ax.plot(x, obs_age_distribution, label="Observed at end of sim", color="blue")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Proportion of Population")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_title("Age Distribution as Proportion of Total Population")
        ax.legend()  # Add legend
        plt.tight_layout()
        if save:
            plt.savefig(results_path / "age_distribution.png")
        if not save:
            plt.show()

    def plot_vital_dynamics(self, save=False, results_path=None):
        # Calculate cumulative sums
        cum_births = np.cumsum(self.results.births, axis=0)
        cum_deaths = np.cumsum(self.results.deaths, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(cum_births, label="Births", color="blue")
        plt.plot(cum_deaths, label="Deaths", color="red")
        plt.title("Cumulative births and deaths")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(results_path / "cum_births_deaths.png")
        if not save:
            plt.show()

@nb.njit(parallel=True)
def fast_vaccination(node_id, disease_state, date_of_birth, ri_timer, sim_t, vx_prob_ri, results_ri_vaccinated, results_ri_protected, rand_vals):
    """
    Optimized vaccination step with thread-local storage and parallel execution.
    """
    if sim_t % 14 != 0:  # Run only every 14th timestep
        return


    num_people = len(node_id)
    num_nodes = results_ri_vaccinated.shape[1]  # Assuming shape (timesteps, nodes)

    # Thread-local storage for results
    local_vaccinated = np.zeros(num_nodes, dtype=np.float32)
    local_protected = np.zeros(num_nodes, dtype=np.int32)

    for i in nb.prange(num_people):
        node = node_id[i]
        if disease_state[i] < 0:  # Skip dead or inactive agents
            continue

        prob_ri = vx_prob_ri if isinstance(vx_prob_ri, float) else vx_prob_ri[node]

        #if sim_t - 14 < date_of_birth[i] + 182 <= sim_t:
        ri_timer[i] -= 14
        if ri_timer[i] <= 0 and ri_timer[i] > -14: # off-by-one?
            if disease_state[i] == 0:  # Must be susceptible
                if rand_vals[i] < prob_ri:  # Vaccination probability
                    disease_state[i] = 3  # Move to Recovered state
                    local_protected[node] += 1

            local_vaccinated[node] += prob_ri  # Expected vaccinated count

    # Merge results back
    for j in nb.prange(num_nodes):
        results_ri_vaccinated[sim_t, j] += int(local_vaccinated[j])
        results_ri_protected[sim_t, j] += local_protected[j]

class RI_ABM:
    def __init__(self, sim):
        self.sim = sim
        self.people = sim.people
        self.nodes = sim.nodes
        self.pars = sim.pars       
        self.people.add_scalar_property("ri_timer", dtype=np.int32, default=-1)
        sim.results.add_array_property("ri_vaccinated", shape=(sim.pars.timesteps, len(sim.nodes)), dtype=np.int32)  # Track number of people vaccinated by RI
        sim.results.add_array_property("ri_protected", shape=(sim.pars.timesteps, len(sim.nodes)), dtype=np.int32)  # Track number of people who enter Recovered state due to RI
        self.results = sim.results

    def step(self):
        # Suppose we have num_people individuals
        rand_vals = np.random.rand(int(1e5)) # this could be done clevererly
        fast_vaccination(
            self.people.node_id,
            self.people.disease_state,
            self.people.date_of_birth,
            self.people.ri_timer,
            self.sim.t,
            self.pars['vx_prob_ri'],
            self.results.ri_vaccinated,
            self.results.ri_protected,
            rand_vals
        )

    def log(self, t):
        pass

    def plot(self, save=False, results_path=None):
        self.plot_cum_ri_vx(save=save, results_path=results_path)

    def plot_cum_ri_vx(self, save=False, results_path=None):
        cum_ri_vaccinated = np.cumsum(self.results.ri_vaccinated, axis=0)
        cum_ri_protected = np.cumsum(self.results.ri_protected, axis=0)

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Plot cumulative RI vaccinated
        axs[0].plot(cum_ri_vaccinated)
        axs[0].set_title("Cumulative RI Vaccinated")
        axs[0].set_xlabel("Time (Timesteps)")
        axs[0].set_ylabel("Cumulative Vaccinated")
        axs[0].grid()

        # Plot cumulative RI protected
        axs[1].plot(cum_ri_protected)
        axs[1].set_title("Cumulative Population Protected by RI")
        axs[1].set_xlabel("Time (Timesteps)")
        axs[1].set_ylabel("Cumulative Protected")
        axs[1].grid()

        plt.tight_layout()
        if save:
            plt.savefig(results_path / "cum_ri_vx.png")
        if not save:
            plt.show()

class SIA_ABM:
    def __init__(self, sim):
        """
        Supplemental Immunization Activity (SIA) component.

        Args:
            sim: The simulation instance.
            sia_schedule: List of vaccination events, each a dict with:
                - 'date': The timestep when the SIA occurs
                - 'nodes': List of nodes to target
                - 'age_range': Tuple (min_age, max_age) in days
                - 'coverage': Vaccine coverage rate (0 to 1)
        """
        self.sim = sim
        self.people = sim.people
        self.nodes = sim.nodes
        self.pars = sim.pars
        self.results = sim.results

        # Add result tracking for SIA
        self.results.add_array_property("sia_vx", shape=(sim.pars.timesteps, len(sim.nodes)), dtype=np.int32)

        # Store vaccination schedule
        self.sia_schedule = sim.pars['sia_schedule']

    def step(self):
        t = self.sim.t  # Current timestep

        # Check if there is an SIA event today
        for event in self.sia_schedule:
            if event['date'] == self.sim.dates[t]:
                self.run_vaccination(event)

    def run_vaccination(self, event):
        """
        Execute vaccination for the given event.

        Args:
            event: Dictionary containing 'nodes', 'age_range', and 'coverage'.
        """
        min_age, max_age = event['age_range']
        nodes_to_vaccinate = event['nodes']

        for node in nodes_to_vaccinate:
            # Find eligible individuals: Alive, susceptible, in the age range
            alive_in_node = (self.people.node_id == node) & (self.people.disease_state >= 0)
            age = (self.sim.t - self.people.date_of_birth)
            in_age_range = (age >= min_age) & (age <= max_age)
            susceptible = self.people.disease_state == 0
            eligible = alive_in_node & in_age_range & susceptible

            # Apply vaccine coverage probability
            sia_eff = self.pars['sia_eff'][node]
            vaccinated = np.random.rand(np.sum(eligible)) < sia_eff
            vaccinated_indices = np.where(eligible)[0][vaccinated]

            # Move vaccinated individuals to the Recovered (R) state
            self.people.disease_state[vaccinated_indices] = 3

            # Track the number vaccinated
            #TODO: clarify that this is the number of people who enter Recovered state, not number vaccinated
            self.results.sia_vx[self.sim.t, node] = vaccinated.sum()

    def log(self, t):
        pass

    def plot(self, save=False, results_path=None):
        self.plot_cum_sia_vx(save=save, results_path=results_path)

    def plot_cum_sia_vx(self, save=False, results_path=None):
        cum_sia_vx = np.cumsum(self.results.sia_vx, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(cum_sia_vx)
        plt.title("Supplemental Immunization Activity (SIA) Vaccination")
        plt.xlabel("Time (Timesteps)")
        plt.ylabel("Cumulative Vaccinated")
        plt.grid()
        if save:
            plt.savefig(results_path / "cum_sia_vx.png")
        if not save:
            plt.show()
