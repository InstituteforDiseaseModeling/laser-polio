import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import scipy.stats as stats
import sciris as sc
from alive_progress import alive_bar
from laser_core.demographics.kmestimator import KaplanMeierEstimator
from laser_core.demographics.pyramid import AliasedDistribution
from laser_core.demographics.pyramid import load_pyramid_csv
from laser_core.laserframe import LaserFrame
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from laser_core.propertyset import PropertySet
from laser_core.utils import calc_capacity
from tqdm import tqdm

import laser_polio as lp

__all__ = ["RI_ABM", "SEIR_ABM", "SIA_ABM", "DiseaseState_ABM", "Transmission_ABM", "VitalDynamics_ABM"]


# SEIR Model
class SEIR_ABM:
    """
    An AGENT-BASED SEIR Model for polio
    Each entry in the population is an agent with a disease state and a node ID
    Disease state codes: 0=S, 1=E, 2=I, 3=R
    """

    def __init__(self, pars: PropertySet = None, verbose=0.1):
        sc.printcyan("Initializing simulation...")

        # Load default parameters and optionally override with user-specified ones
        self.pars = deepcopy(lp.default_pars)
        if pars is not None:
            self.pars += pars  # override default values
        pars = self.pars
        self.verbose = verbose

        # Setup time
        self.t = 0  # Current timestep
        self.nt = (
            pars.dur + 1
        )  # Number of timesteps. We add 1 to include step 0 (initial conditions) and then run for pars.dur steps. Individual components can have their own step sizes
        self.datevec = lp.daterange(self.pars["start_date"], days=self.nt)  # Time represented as an array of datetime objects

        # Initialize the population
        pars.n_ppl = np.atleast_1d(pars.n_ppl).astype(int)  # Ensure pars.n_ppl is an array
        if (pars.cbr is not None) & (len(pars.cbr) == 1):
            capacity = int(1.1 * calc_capacity(np.sum(pars.n_ppl), self.nt, pars.cbr[0]))
        elif (pars.cbr is not None) & (len(pars.cbr) > 1):
            capacity = int(1.1 * calc_capacity(np.sum(pars.n_ppl), self.nt, np.mean(pars.cbr)))
        else:
            capacity = int(np.sum(pars.n_ppl))
        self.people = LaserFrame(capacity=capacity, initial_count=int(np.sum(pars.n_ppl)))
        # We initialize disease_state here since it's required for most other components (which facilitates testing)
        self.people.add_scalar_property("disease_state", dtype=np.int32, default=-1)  # -1=Dead/inactive, 0=S, 1=E, 2=I, 3=R
        self.people.disease_state[: self.people.count] = 0  # Set initial population as susceptible
        self.results = LaserFrame(capacity=1)

        # Setup spatial component with node IDs
        self.nodes = np.arange(len(np.atleast_1d(pars.n_ppl)))
        self.people.add_scalar_property("node_id", dtype=np.int32, default=0)
        node_ids = np.concatenate([np.full(count, i) for i, count in enumerate(pars.n_ppl)])
        self.people.node_id[0 : np.sum(pars.n_ppl)] = node_ids  # Assign node IDs to initial people
        self.results.add_array_property("node_pop", shape=(self.nt, len(self.nodes)), dtype=np.int32)

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
        Sets up the components of the model in the order specified in pars.py and initializes instances and phases.

        This function takes a list of component types, creates an instance of each, and adds each callable component to the phase list.
        It also registers any components with an `on_birth` function with the `Births` component.

        Args:

            components (list): A list of component classes to be initialized and integrated into the model.

        Returns:

            None
        """

        # Get the default order from default pars
        default_order = lp.default_run_order

        # Sort the provided list of component classes based on their string names
        def get_name(cls):
            return cls.__name__

        component_lookup = {cls.__name__: cls for cls in components}
        ordered_subset = [component_lookup[name] for name in default_order if name in component_lookup]

        # Store and instantiate
        self._components = ordered_subset
        self.instances = [cls(self) for cls in ordered_subset]
        print(f"Initialized components: {self.instances}")

    def run(self):
        sc.printcyan("Initialization complete. Running simulation...")
        self.component_times = dict.fromkeys(self.instances, 0.0)
        self.component_times["report"] = 0
        with alive_bar(self.nt, title="Simulation progress:") as bar:
            for tick in range(self.nt):
                if tick == 0:
                    # Just record the initial state on t=0 & don't run any components
                    self.log_results(tick)
                    self.t += 1
                else:
                    for component in self.instances:
                        start_time = time.perf_counter()
                        component.step()
                        end_time = time.perf_counter()
                        self.component_times[component] += end_time - start_time

                    start_time = time.perf_counter()
                    self.log_results(tick)
                    end_time = time.perf_counter()
                    self.component_times["report"] += end_time - start_time
                    self.t += 1
                bar()  # Update the progress bar
        sc.printcyan("Simulation complete.")

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
            sc.printcyan("Saving plots in " + str(results_path))
        for component in self.instances:
            component.plot(save=save, results_path=results_path)
        self.plot_node_pop(save=save, results_path=results_path)

        if self.component_times:
            labels = [component.__class__.__name__ for component in self.instances]
            labels.append("report")
            if self.verbose > 0.1:
                print(f"{self.instances=}")
                print(f"{labels=}")
            # times = [self.component_times[component] for component in labels ]
            times = [self.component_times[component] for component in self.instances]
            times.append(self.component_times["report"])  # hack
            plt.figure(figsize=(6, 6))
            plt.pie(times, labels=labels, autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)
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


@nb.njit(parallel=True, cache=True)
def step_nb(disease_state, exposure_timer, infection_timer, acq_risk_multiplier, daily_infectivity, paralyzed, p_paralysis, active_count):
    for i in nb.prange(active_count):
        if disease_state[i] == 1:  # Exposed
            if exposure_timer[i] <= 0:
                disease_state[i] = 2  # Become infected
                # Apply paralysis probability immediately after infection
                if np.random.random() < p_paralysis:
                    paralyzed[i] = 1
            exposure_timer[i] -= 1  # Decrement exposure timer so that they become infected on the next timestep

        if disease_state[i] == 2:  # Infected
            if infection_timer[i] <= 0:
                disease_state[i] = 3  # Become recovered
                acq_risk_multiplier[i] = 0.0  # Reset risk
                daily_infectivity[i] = 0.0  # Reset infectivity
            infection_timer[i] -= 1  # Decrement infection timer so that they recover on the next timestep


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
        # Initialize all agents with an exposure_timer & infection_timer
        sim.people.add_scalar_property("exposure_timer", dtype=np.int32, default=0)
        # Subtract 1 to account for the fact that we expose people in transmission component after the disease state component (newly exposed miss their first timer decrement)
        sim.people.exposure_timer[:] = self.pars.dur_exp(self.people.capacity) - 1
        sim.people.add_scalar_property("infection_timer", dtype=np.int32, default=0)
        sim.people.infection_timer[:] = self.pars.dur_inf(self.people.capacity)

        sim.results.add_array_property("S", shape=(sim.nt, len(self.nodes)), dtype=np.int32)
        sim.results.add_array_property("E", shape=(sim.nt, len(self.nodes)), dtype=np.int32)
        sim.results.add_array_property("I", shape=(sim.nt, len(self.nodes)), dtype=np.int32)
        sim.results.add_array_property("R", shape=(sim.nt, len(self.nodes)), dtype=np.int32)
        sim.results.add_array_property("paralyzed", shape=(sim.nt, len(self.nodes)), dtype=np.int32)

        def do_init_imm():
            print(f"Before immune initialization, we have {sim.people.count} active agents.")
            # Initialize immunity
            if isinstance(pars.init_immun, (float, list)):  # Handle both float and list cases
                init_immun_value = pars.init_immun[0] if isinstance(pars.init_immun, list) else pars.init_immun
                num_recovered = int(sum(pars.n_ppl) * init_immun_value)
                recovered_indices = np.random.choice(sum(pars.n_ppl), size=num_recovered, replace=False)
                sim.people.disease_state[recovered_indices] = 3
            else:
                # Initialize by node
                # Extract age bins dynamically from column names
                age_bins = {}
                for col in pars.init_immun.columns:
                    if col.startswith("immunity_"):
                        _, min_age, max_age = col.split("_")
                        min_age_days, max_age_days = (
                            int(min_age) * 30.43,
                            (int(max_age) + 1) * 30.43,
                        )  # We add one here b/c the max is exclusive. See filtering logic below for who is considered eligible.
                        age_bins[col] = (min_age_days, max_age_days)
                        # Assign recovered status based on immunity data

                def viz():
                    """
                    Utility function to display histogram of population (active agents) by age. Can
                    be used to view population structure before and after EULA-gizing.
                    """
                    ages = sim.people.date_of_birth[: sim.people.count] * -1 / 365.0
                    plt.figure(figsize=(10, 6))
                    plt.hist(ages, bins=np.arange(0, 101, 1), edgecolor="black", alpha=0.7)  # Bins in 5-year intervals
                    plt.xlabel("Age (years)")
                    plt.ylabel("Number of Individuals")
                    plt.title("Age Distribution of the Population")
                    plt.grid(axis="y", linestyle="--", alpha=0.7)

                    # Show the plot
                    plt.show()

                # viz()

                # Assume everyone older than 15 years of age is immune
                # We EULA-gize the 15+ first to speedup immunity initialization for <15s
                o15 = (sim.people.date_of_birth * -1) >= 15 * 365
                sim.people.disease_state[o15] = 3  # Set as recovered
                active_count_init = sim.people.count  # This gives the active population size
                valid_agents = self.people.disease_state[:active_count_init] >= 0  # Apply only to active agents
                filter_mask = (self.people.disease_state[:active_count_init] < 3) & valid_agents  # Now matches active count

                def get_node_counts_pre_squash(filter_mask, active_count):
                    # Count up R by node before we squash
                    # Ensure everything is properly sliced up to active_count
                    node_ids = sim.people.node_id[:active_count]
                    # Get a mask for elements outside the filter (i.e., those with disease_state >= 3 or inactive)
                    outside_filter_mask = ~filter_mask  # Invert the mask
                    # Get the node IDs corresponding to those outside the filter
                    outside_nodes = node_ids[outside_filter_mask]  # Now should match in length
                    # Use np.bincount to count occurrences efficiently
                    max_node_id = sim.people.node_id.max()  # Get max node_id to define the bin range
                    node_counts = np.bincount(outside_nodes, minlength=max_node_id + 1)
                    return node_counts

                def prepop_eula(node_counts, life_expectancies):
                    # TODO: refine mortality estimates since the following code is just a rough cut

                    # Get simulation parameters
                    T = self.results.R.shape[0]  # Number of timesteps
                    node_count = self.results.R.shape[1]  # Number of nodes

                    # Compute mean date_of_birth per node
                    node_dob_sums = np.bincount(
                        self.people.node_id[: self.people.count],
                        weights=self.people.date_of_birth[: self.people.count],
                        minlength=node_count,
                    )

                    with np.errstate(divide="ignore", invalid="ignore"):
                        mean_dob = np.where(node_counts > 0, node_dob_sums / node_counts, 0)  # Avoid div by zero

                    # Calculate mean age per node
                    mean_ages_years = -mean_dob / 365  # Approximate mean age per node
                    adjusted_life_expectancy = np.maximum(life_expectancies - mean_ages_years, 1)  # Avoid zero or negative values

                    # Compute age-dependent mortality rate (days^-1)
                    # Assume mortality rate λ = 1 / life_expectancy
                    mortality_rates = 1 / (adjusted_life_expectancy * 365)

                    # Generate mortality-adjusted population over time
                    time_range = np.arange(T)[:, None]  # Create time indices
                    self.results.R[:, :] += (node_counts * np.exp(-mortality_rates * time_range)).astype(np.int32)

                # Get our EULA populations
                node_counts = get_node_counts_pre_squash(filter_mask, active_count_init)
                # Add EULA pops and projected populations over time due to mortality to reported R counts for whole sim
                prepop_eula(node_counts, pars.life_expectancies)
                # Now squash
                sim.people.squash(filter_mask)
                deletions = active_count_init - sim.people.count
                # We don't have nice solution for resizing the population LaserFrame yet so we'll make a note of our actual new capacity
                sim.people.true_capacity = sim.people.capacity - deletions

                # This is our faster solution for doing initial immunity
                alive_mask = self.people.disease_state >= 0  # Mask for alive individuals
                node_ids = self.people.node_id
                dob = self.people.date_of_birth * -1

                # Initialize expected recovered counts
                expected_recovered = np.zeros_like(self.people.disease_state, dtype=np.int32)

                # Iterate over age bins, vectorizing across nodes
                for age_key, (age_min, age_max) in age_bins.items():
                    immune_fractions = pars.init_immun[age_key].values  # Immunity fraction per node

                    # Mask for eligible individuals
                    eligible_mask = alive_mask & (dob >= age_min) & (dob < age_max)

                    # Count eligible individuals per node
                    per_node_eligible = np.bincount(node_ids[eligible_mask], minlength=len(self.nodes))

                    # Compute expected recovered count per node, ensuring proper scaling
                    expected_recovered_per_node = np.random.poisson(immune_fractions * per_node_eligible)

                    # Assign expected recoveries per node to individuals
                    eligible_indices = np.where(eligible_mask)[0]
                    if len(eligible_indices) > 0:
                        # Ensure node index is within valid range
                        valid_node_indices = np.clip(node_ids[eligible_indices], 0, len(immune_fractions) - 1)

                        # Compute per-individual recovery probability
                        recovery_prob = expected_recovered_per_node[valid_node_indices] / np.maximum(
                            per_node_eligible[valid_node_indices], 1
                        )

                        # Clip probability to [0, 1] to avoid errors
                        recovery_prob = np.clip(recovery_prob, 0, 1)

                        # Apply probabilistic recovery assignment
                        expected_recovered[eligible_indices] = np.random.binomial(1, recovery_prob)

                # Assign recovery state
                self.people.disease_state[expected_recovered.astype(bool)] = 3  # Set as recovered

                # We're going to squash again to EULA-gize the initial R population in our under 15s
                active_count = sim.people.count  # This gives the active population size
                valid_agents = self.people.disease_state[:active_count] >= 0  # Apply only to active agents
                filter_mask = (self.people.disease_state[:active_count] < 3) & valid_agents  # Now matches active count
                node_counts = get_node_counts_pre_squash(filter_mask, active_count)
                prepop_eula(node_counts, pars.life_expectancies)

                sim.people.squash(filter_mask)
                new_active_count = sim.people.count
                deletions = active_count - new_active_count
                sim.people.true_capacity -= deletions

                print(f"After immune initialization and EULA-gizing, we have {sim.people.count} active agents.")
                # viz()

        do_init_imm()

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
            node_ids = self.people.node_id[: self.people.count]
            disease_states = self.people.disease_state[: self.people.count]
            for node, prev in tqdm(enumerate(pars.init_prev), total=len(pars.init_prev), desc="Seeding infections in nodes"):
                num_infected = int(pars.n_ppl[node] * prev)
                alive_in_node = (node_ids == node) & (disease_states >= 0)
                alive_in_node_indices = np.where(alive_in_node)[0]
                num_infections_to_draw = min(num_infected, len(alive_in_node_indices))
                infected_indices_node = np.random.choice(alive_in_node_indices, size=num_infections_to_draw, replace=False)
                infected_indices.extend(infected_indices_node)
        num_infected = len(infected_indices)
        sim.people.disease_state[infected_indices] = 2

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
            self.pars.p_paralysis,
            self.people.count,
        )

    def log(self, t):
        pass

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
        rows, cols = 2, int(np.ceil(n_panels / 2))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), sharex=True, sharey=True, constrained_layout=True)
        axs = axs.ravel()  # Flatten in case of non-square grid
        timepoints = np.linspace(0, self.pars.dur, n_panels, dtype=int)
        lats, lons = self.pars.centroids["center_lat"], self.pars.centroids["center_lon"]

        # Get global min and max for consistent color scale
        infection_min = np.min(self.results.I)
        infection_max = np.max(self.results.I)

        for i, ax in enumerate(axs[:n_panels]):  # Ensure we don't go out of bounds
            t = timepoints[i]
            infection_counts = self.results.I[t, :]

            scatter = ax.scatter(
                lons, lats, c=infection_counts, cmap="Reds", edgecolors=None, alpha=0.9, vmin=infection_min, vmax=infection_max
            )
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
        fig.colorbar(scatter, ax=axs, location="right", fraction=0.05, pad=0.05, label="Infection Count")

        # Add title
        fig.suptitle("Infected Population by Node", fontsize=16)

        if save:
            if results_path is None:
                raise ValueError("Please provide a results path to save the plots.")
            plt.savefig(f"{results_path}/infected_map.png")
        else:
            plt.show()


@nb.njit(parallel=True)  # , cache=True)
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


@nb.njit(parallel=True, cache=True)
def compute_infections_nb(num_people, num_nodes, disease_state, node_id, acq_risk_multiplier, beta_per_node, local_sums):
    """
    Return an array "exposure_sums" where exposure_sums[node] is the sum of
    probabilities for susceptible individuals in that node.
    """

    # Parallel loop
    for i in nb.prange(num_people):
        if disease_state[i] == 0:  # susceptible
            nd = node_id[i]
            # base probability from node-level infection rate
            prob_infection = beta_per_node[nd] * acq_risk_multiplier[i]
            # Keep a sum of these probabilities
            tid = nb.get_thread_id()
            local_sums[tid, nd] += prob_infection

    return


@nb.njit(parallel=True, cache=True)
def fast_infect(node_ids, exposure_probs, disease_state, new_infections, local_counts, counts):
    """
    A Numba-accelerated version of faster_infect.
    Parallelizes over nodes, computing a CDF for each node's susceptible population.
    Selects 'n_to_draw' indices via binary search of random values, and marks them infected.

    NOTE: This version does NOT enforce uniqueness of selected indices within the same node.
    """
    num_nodes = len(new_infections)
    # Precompute which individuals are susceptible
    # is_sus = disease_state == 0
    n_people = len(node_ids)

    for i in nb.prange(n_people):
        if disease_state[i] == 0:
            local_counts[nb.get_thread_id(), node_ids[i]] += 1

    counts[:] = local_counts.sum(axis=0)

    for node in nb.prange(num_nodes):
        n_to_draw = new_infections[node]
        if n_to_draw <= 0:
            continue

        # 1) Gather susceptible indices for this node
        # count = 0
        # for i in range(n_people):
        #     if node_ids[i] == node and disease_state[i] == 0:
        #         count += 1
        count = counts[node]

        if count == 0:
            continue

        sus_indices = np.empty(count, dtype=np.int64)
        sus_probs = np.empty(count, dtype=np.float32)

        idx = 0
        previous = 0.0  # build CDF simultaneously
        for i in range(n_people):
            if node_ids[i] == node and disease_state[i] == 0:
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

            # Expose the chosen individual
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

        # Stash the R0 scaling factor
        self.r0_scalars = self.pars.r0_scalars
        # # Calcultate geographic R0 modifiers based on underweight data (one for each node)
        # underwt = self.pars.r0_scalars
        # self.r0_scalars = 1 / (1 + np.exp(24 * (np.mean(underwt) - underwt))) + 0.2

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
        # Step 2: Define parameters for daily_infectivity (Gamma distribution)
        mean_gamma = self.pars.r0 / np.mean(self.pars.dur_inf(1000))  # mean_gamma = R0 / mean(dur_inf)
        shape_gamma = 1  # makes this equivalent to an exponential distribution
        scale_gamma = mean_gamma / shape_gamma
        scale_gamma = max(scale_gamma, 1e-10)  # Ensure scale is never exactly 0 since gamma is undefined for scale_gamma=0
        # Step 3: Generate correlated normal samples
        rho = 0.8  # Desired correlation
        cov_matrix = np.array([[1, rho], [rho, 1]])  # Create covariance matrix
        L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition
        # Generate standard normal samples
        if not hasattr(self.people, "true_capacity"):
            self.people.true_capacity = self.people.capacity  # Ensure true_capacity is set even if we don't initialize prevalence by node
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
        # def fast_beta():
        #     beta_ind_sums = compute_beta_ind_sums(node_ids, infectivity, disease_state, len(self.nodes))
        #     return beta_ind_sums

        # 1) Sum up the total amount of infectivity shed by all infectious agents within a node.
        # This is the daily number of infections that these individuals would be expected to generate
        # in a fully susceptible population sans spatial and seasonal factors.
        # check_time = time.perf_counter()
        disease_state = self.people.disease_state[: self.people.count]
        node_ids = self.people.node_id[: self.people.count]
        infectivity = self.people.daily_infectivity[: self.people.count]
        risk = self.people.acq_risk_multiplier[: self.people.count]
        # node_beta_sums = fast_beta()
        node_beta_sums = compute_beta_ind_sums(node_ids, infectivity, disease_state, len(self.nodes))

        # 2) Spatially redistribute infectivity among nodes
        transfer = (node_beta_sums * self.network).astype(np.float64)  # Don't round here, we'll handle fractional infections later
        # Ensure net contagion remains positive after movement
        node_beta_sums += transfer.sum(axis=1) - transfer.sum(axis=0)
        node_beta_sums = np.maximum(node_beta_sums, 0)  # Prevent negative contagion

        # 3) Apply seasonal & geographic modifiers
        beta_seasonality = lp.get_seasonality(self.sim)
        beta = node_beta_sums * beta_seasonality * self.r0_scalars  # Total node infection rate

        # 4) Calculate base probability for each agent to become exposed
        alive_counts = self.people.count + self.sim.results.R[self.sim.t]
        per_agent_infection_rate = beta / np.clip(alive_counts, 1, None)
        base_prob_infection = 1 - np.exp(-per_agent_infection_rate)

        # 5) Calculate infections
        num_people = len(disease_state)
        num_nodes = len(base_prob_infection)
        # Thread-local storage
        n_threads = nb.get_num_threads()
        local_sums = np.zeros((n_threads, num_nodes), dtype=np.float64)
        compute_infections_nb(num_people, num_nodes, disease_state, node_ids, risk, base_prob_infection, local_sums)
        exposure_sums = local_sums.sum(axis=0)  # merge per-thread counts
        new_infections = np.random.poisson(exposure_sums).astype(np.int32)

        # 6) Draw n_expected_exposures for each node according to their exposure_probs
        local_counts = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)
        counts = np.zeros(num_nodes, dtype=np.int32)
        fast_infect(node_ids, risk, disease_state, new_infections, local_counts, counts)

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


class VitalDynamics_ABM:
    def __init__(self, sim):
        self.sim = sim
        self.people = sim.people
        self.nodes = sim.nodes
        self.results = sim.results
        self.step_size = sim.pars.step_size_VitalDynamics_ABM  # Number of days between vital dynamics steps

        # Setup the age and vital rate components
        pars = sim.pars
        if pars.age_pyramid_path is not None:
            sim.people.add_scalar_property("date_of_birth", dtype=np.int32, default=-1)
            pyramid = load_pyramid_csv(pars.age_pyramid_path)
            MINCOL = 0
            MAXCOL = 1
            MCOL = 2
            FCOL = 3
            sampler = AliasedDistribution(pyramid[:, MCOL] + pyramid[:, FCOL])  # using the male population in this example
            samples = sampler.sample(len(sim.people))
            bin_min_age_days = pyramid[:, MINCOL] * 365  # minimum age for bin, in days (include this value)
            bin_max_age_days = (pyramid[:, MAXCOL] + 1) * 365  # maximum age for bin, in days (exclude this value)
            mask = np.zeros(len(sim.people), dtype=bool)
            ages = np.zeros(len(sim.people), dtype=np.int32)
            for i in range(len(pyramid)):  # for each possible bin value...
                mask[:] = samples == i  # ...find the agents that belong to this bin
                # ...and assign a random age, in days, within the bin
                ages[mask] = np.random.randint(bin_min_age_days[i], bin_max_age_days[i], mask.sum())
            # Move births on day 0 to one day prior. This prevents births on day 0 when we only record results, we don't run components.
            ages[ages == 0] = 1
            sim.people.date_of_birth[: len(sim.people)] = -ages

        if pars.cbr is not None:
            sim.results.add_array_property("births", shape=(sim.nt, len(sim.nodes)), dtype=np.int32)
            sim.results.add_array_property("deaths", shape=(sim.nt, len(sim.nodes)), dtype=np.int32)
            sim.people.add_scalar_property("date_of_death", dtype=np.int32, default=0)

            cumulative_deaths = lp.create_cumulative_deaths(np.sum(pars.n_ppl), max_age_years=100)
            sim.death_estimator = KaplanMeierEstimator(cumulative_deaths)
            lifespans = sim.death_estimator.predict_age_at_death(ages, max_year=100)
            # Set pars.life_expectancies to mean lifespans by node.
            # This is just to support placeholder mortality premodeling for EULAs.
            # Would move this code block to EULA section but we've got lifespans here.
            node_ids = sim.people.node_id[: sim.people.count]
            unique_nodes, indices = np.unique(node_ids, return_inverse=True)
            pars.life_expectancies = np.bincount(indices, weights=lifespans / 365) / np.bincount(indices)

            dods = lifespans - ages  # we could check that dods is non-negative to be safe
            sim.people.date_of_death[: np.sum(pars.n_ppl)] = dods

        for node in self.nodes:
            if len(pars.cbr) == 1:
                self.birth_rate = pars.cbr / (365 * 1000)
            else:
                self.birth_rate = pars.cbr[node] / (365 * 1000)  # Birth rate per day per person

        self.death_estimator = sim.death_estimator

    def step(self):
        t = self.sim.t
        if t % self.step_size != 0:
            return

        # 1) Vectorized mask of all alive people
        alive = self.people.disease_state >= 0

        # 2) Count how many alive in each node in one pass
        node_ids_alive = self.people.node_id[alive]
        alive_count_by_node = np.bincount(node_ids_alive, minlength=len(self.nodes))

        # 3) Compute births node by node, but without big boolean masks
        for node in self.nodes:
            expected_births = self.step_size * self.birth_rate * alive_count_by_node[node]

            # Integer part plus probabilistic fractional part
            birth_integer = int(expected_births)
            birth_fraction = expected_births - birth_integer
            birth_rand = np.random.binomial(1, birth_fraction)  # Bernoulli draw
            births = birth_integer + birth_rand

            # If births occur, add them to the population
            if births > 0:
                start, end = self.people.add(births)

                newborn_ages = np.zeros(births, dtype=np.int32)
                lifespans = self.death_estimator.predict_age_at_death(newborn_ages, max_year=100)

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


@nb.njit((nb.int32, nb.int32[:], nb.int32[:], nb.float64[:], nb.int32, nb.float64[:], nb.int32, nb.int32[:, :]), parallel=True, cache=True)
def fast_ri(step_size, node_id, disease_state, ri_timer, sim_t, vx_prob_ri, num_people, local_counts):
    """
    Optimized vaccination step with thread-local storage and parallel execution.
    """

    for i in nb.prange(num_people):
        state = disease_state[i]
        if state < 0:  # Skip dead or inactive agents
            continue

        node = node_id[i]

        prob_vx = vx_prob_ri[node]

        # print(f"Agent {i} in disease state {disease_state[i]}")
        # print("prob_vx=", prob_vx, "prob_take=", prob_take)

        timer = ri_timer[i] - step_size
        ri_timer[i] = timer
        eligible = False
        # If first vx, account for the fact that no components are run on day 0
        if sim_t == step_size:
            eligible = timer <= 0 and timer >= -step_size
        elif sim_t > step_size:
            eligible = timer <= 0 and timer > -step_size

        if eligible:
            if np.random.rand() < prob_vx:
                local_counts[nb.get_thread_id(), node] += 1  # Increment vaccinated count
                if state == 0:  # If susceptible
                    # We don't check for vx_eff here, since that is already accounted for in the prob_vx file
                    disease_state[i] = 3  # Move to Recovered state

    return


class RI_ABM:
    def __init__(self, sim):
        self.sim = sim
        self.step_size = sim.pars.step_size_RI_ABM  # Number of days between RI steps
        self.people = sim.people
        self.nodes = sim.nodes
        self.pars = sim.pars
        # Calc date of RI (assume single point in time between 1st and 3rd dose)
        self.people.add_scalar_property("ri_timer", dtype=np.int32, default=-1)
        dob = self.people.date_of_birth
        days_from_birth_to_ri = np.random.uniform(42, 98, len(self.people.ri_timer))  # Assume 6-14 weeks of age for vx
        self.people.ri_timer = dob + days_from_birth_to_ri
        sim.results.add_array_property(
            "ri_vaccinated", shape=(sim.nt, len(sim.nodes)), dtype=np.int32
        )  # Track number of people vaccinated & protected by RI
        self.results = sim.results

    def step(self):
        if self.pars["vx_prob_ri"] is None:
            return

        vx_prob_ri = self.pars["vx_prob_ri"]  # Includes coverage & efficacy
        num_nodes = len(self.sim.nodes)
        # Promote to 1D arrays if needed
        if np.isscalar(vx_prob_ri):
            vx_prob_ri = np.full(num_nodes, vx_prob_ri, dtype=np.float64)

        # Suppose we have num_people individuals
        # rand_vals = np.random.rand(self.people.count)  # this could be done clevererly

        if self.sim.t % self.step_size == 0:
            local_counts = np.zeros((nb.get_num_threads(), self.results.ri_vaccinated.shape[1]), dtype=np.int32)
            fast_ri(
                self.step_size,
                self.people.node_id,
                self.people.disease_state,
                self.people.ri_timer,
                self.sim.t,
                vx_prob_ri,
                self.people.count,
                local_counts,
            )
            self.results.ri_vaccinated[self.sim.t] = local_counts.sum(axis=0)

    def log(self, t):
        pass

    def plot(self, save=False, results_path=None):
        self.plot_cum_ri_vx(save=save, results_path=results_path)

    def plot_cum_ri_vx(self, save=False, results_path=None):
        # Plot cumulative RI vaccinated
        cum_ri_vaccinated = np.cumsum(self.results.ri_vaccinated, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(cum_ri_vaccinated)
        plt.title("Cumulative RI Vaccinated (includes efficacy)")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Vaccinated")
        plt.grid()
        if save:
            plt.savefig(results_path / "cum_ri_vx.png")
        if not save:
            plt.show()


@nb.njit(parallel=True)  # , cache=True)
def fast_sia(
    node_ids,
    disease_states,
    dobs,
    sim_t,
    vx_prob,
    vx_eff,
    count,
    nodes_to_vaccinate,
    min_age,
    max_age,
    local_vaccinated,
    local_protected,
):
    """
    Numbified supplemental immunization activity (SIA) vaccination step.

    Parameters:
        node_ids: Array of node IDs for each agent.
        disease_states: Array of disease states for each agent.
        dobs: Array of date of birth for each agent.
        sim_t: Current simulation timestep.
        vx_eff: Vaccine efficacy for this vaccine type (scalar).
        vx_prob_sia: Array of coverage probabilities by node.
        results_vaccinated: Output array for vaccinated counts (timesteps x nodes).
        results_protected: Output array for protected counts (timesteps x nodes).
        rand_vals: Random array of uniform [0,1] values, length >= count.
        count: Number of active agents.
        nodes_to_vaccinate: Array of nodes targeted by this campaign.
        min_age, max_age: Integers, age range eligibility in days.
    """
    num_people = count

    for i in nb.prange(num_people):
        # Skip if agent is not alive, not in targeted node, or not in age range
        if disease_states[i] < 0:
            continue

        age = sim_t - dobs[i]
        if not (min_age <= age <= max_age):
            continue

        node = node_ids[i]
        if nodes_to_vaccinate[node] == 0:
            continue

        r = np.random.rand()
        prob_vx = vx_prob[node]

        if r < prob_vx:  # Check probability of vaccination
            thread_id = nb.get_thread_id()
            local_vaccinated[thread_id, node] += 1  # Increment vaccinated count
            if disease_states[i] == 0:  # If susceptible
                if r < prob_vx * vx_eff:  # Check probability that vaccine takes/protects
                    disease_states[i] = 3  # Move to Recovered state
                    local_protected[thread_id, node] += 1  # Increment protected count

    return


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
                - 'vaccinetype': The vaccine type which is used to determine efficacy
        """
        self.sim = sim
        self.people = sim.people
        self.nodes = sim.nodes
        self.pars = sim.pars
        self.results = sim.results

        # Add result tracking for SIA
        self.results.add_array_property("sia_vaccinated", shape=(sim.nt, len(sim.nodes)), dtype=np.int32)
        self.results.add_array_property("sia_protected", shape=(sim.nt, len(sim.nodes)), dtype=np.int32)

        # Store vaccination schedule
        self.sia_schedule = sim.pars["sia_schedule"] if sim.pars["sia_schedule"] else []
        # Convert all 'date' values in self.sia_schedule to datetime.date
        for event in self.sia_schedule:
            event["date"] = lp.date(event["date"])

    def step(self):
        t = self.sim.t  # Current timestep

        # Check if there is an SIA event today
        for event in self.sia_schedule:
            if event["date"] == self.sim.datevec[t]:
                if self.pars.vx_prob_sia is None:
                    continue
                # nodes_to_vaccinate = np.array(event["nodes"], dtype=np.int32)  # Convert to NumPy array
                nodes_to_vaccinate = np.zeros(len(self.sim.nodes), np.uint8)
                nodes_to_vaccinate[event["nodes"]] = 1
                vx_prob_sia = np.array(self.pars["vx_prob_sia"], dtype=np.float32)  # Convert to NumPy array
                vaccinetype = event["vaccinetype"]
                vx_eff = self.pars["vx_efficacy"][vaccinetype]
                min_age, max_age = event["age_range"]

                # Suppose we have num_people individuals
                # Pre-allocate thread-local result arrays
                local_vaccinated = np.zeros((nb.get_num_threads(), self.results.sia_vaccinated.shape[1]), dtype=np.int32)
                local_protected = np.zeros((nb.get_num_threads(), self.results.sia_protected.shape[1]), dtype=np.int32)

                fast_sia(
                    self.people.node_id,
                    self.people.disease_state,
                    self.people.date_of_birth,
                    self.sim.t,
                    vx_prob_sia,
                    vx_eff,
                    self.people.count,
                    nodes_to_vaccinate,
                    min_age,
                    max_age,
                    local_vaccinated,
                    local_protected,
                )
                # Aggregate thread-local counts into global result arrays
                self.results.sia_vaccinated[self.sim.t] = local_vaccinated.sum(axis=0)
                self.results.sia_protected[self.sim.t] = local_protected.sum(axis=0)

    # def run_vaccination(self, event):
    #     """
    #     Execute vaccination for the given event.

    #     Args:
    #         event: Dictionary containing 'nodes', 'age_range', and 'coverage'.
    #     """
    #     min_age, max_age = event["age_range"]
    #     nodes_to_vaccinate = event["nodes"]
    #     vaccinetype = event["vaccinetype"]
    #     vx_eff = self.pars["vx_efficacy"][vaccinetype]

    #     node_ids = self.people.node_id[: self.people.count]
    #     disease_states = self.people.disease_state[: self.people.count]
    #     dobs = self.people.date_of_birth[: self.people.count]

    #     for node in nodes_to_vaccinate:
    #         # Find eligible individuals: Alive, susceptible, in the age range
    #         alive_in_node = (node_ids == node) & (disease_states >= 0)
    #         age = self.sim.t - dobs
    #         in_age_range = (age >= min_age) & (age <= max_age)
    #         susceptible = disease_states == 0
    #         eligible = alive_in_node & in_age_range & susceptible

    #         # Apply vaccine coverage probability
    #         prob_vx = self.pars["vx_prob_sia"][node]
    #         rand_vals = np.random.rand(np.sum(eligible))
    #         for i in len(eligible):
    #             if rand_vals[i] < prob_vx:  # Check probability of vaccination
    #                 self.sim.results.sia_vaccinated[self.sim.t, node] += 1  # Increment vaccinated count
    #                 if disease_states[i] == 0:  # If susceptible
    #                     if rand_vals[i] < vx_eff:  # Check probability that vaccine takes/protects
    #                         # Move vaccinated individuals to the Recovered (R) state
    #                         disease_states[i] = 3  # Move to Recovered state
    #                         self.sim.results.n_protected_sia[self.sim.t, node] += 1  # Increment protected count

    def log(self, t):
        pass

    def plot(self, save=False, results_path=None):
        self.plot_cum_vx_sia(save=save, results_path=results_path)

    def plot_cum_vx_sia(self, save=False, results_path=None):
        cum_vx_sia = np.cumsum(self.results.sia_vaccinated, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(cum_vx_sia)
        plt.title("Supplemental Immunization Activity (SIA) Vaccination")
        plt.xlabel("Time (Timesteps)")
        plt.ylabel("Cumulative Vaccinated")
        plt.grid()
        if save:
            plt.savefig(results_path / "cum_sia_vx.png")
        if not save:
            plt.show()
