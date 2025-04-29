import logging
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import h5py
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
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import radiation
from laser_core.migration import row_normalizer
from laser_core.propertyset import PropertySet
from laser_core.random import seed as set_seed
from laser_core.utils import calc_capacity
from tqdm import tqdm

import laser_polio as lp
from laser_polio.laserframeio import LaserFrameIO

__all__ = ["RI_ABM", "SEIR_ABM", "SIA_ABM", "DiseaseState_ABM", "Transmission_ABM", "VitalDynamics_ABM"]

### START WITH LOGGER SETUP


# Let's color-code our log messages based on level.
# Note that this just does the log level and module name, not the whole message
class LogColors:
    RESET = "\033[0m"
    BROWN = "\033[38;5;94m"  # Approximate brown using 256-color mode
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


# Let's add a whole new log level that logging doesn't know about
# We do this in the middle of color-coding since our new level will need a color too.
VALID = 15
logging.addLevelName(VALID, "VALID")


class ColorFormatter(logging.Formatter):
    LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: LogColors.BROWN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.MAGENTA,
        VALID: LogColors.BLUE,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
        record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
        record.name = f"{color}{record.name}{LogColors.RESET}"
        return super().format(record)


def valid(self, message, *args, **kwargs):
    if self.isEnabledFor(VALID):
        self._log(VALID, message, args, **kwargs)


logging.Logger.valid = valid

# Actually get the logger singleton by module-name
logger = logging.getLogger("laser-polio")
# Prevents double/multiple logging
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter("[%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(console_handler)

### DONE WITH LOGGER SETUP

# Configure the logger once
logging.basicConfig(
    filename="simulation_log.txt",  # or use .log or .csv depending on how you want to consume it
    level=logging.INFO,
    format="%(asctime)s [T=%(message)s]",
    filemode="w",  # Overwrite each time you run; use "a" to append
)
logger = logging.getLogger(__name__)


# Logger precision formatter
def fmt(arr, precision=2):
    """Format NumPy arrays as single-line strings with no wrapping."""
    return np.array2string(
        np.asarray(arr),  # Ensures even scalars/lists work
        separator=" ",
        threshold=np.inf,
        max_line_width=np.inf,
        precision=precision,
    )


# SEIR Model
class SEIR_ABM:
    """
    An AGENT-BASED SEIR Model for polio
    Each entry in the population is an agent with a disease state and a node ID
    Disease state codes: 0=S, 1=E, 2=I, 3=R
    """

    def common_init(self, pars, verbose):
        self.component_times = defaultdict(float)  # Initialize component times

        # Load default parameters and optionally override with user-specified ones
        self.pars = deepcopy(lp.default_pars)
        if pars is not None:
            self.pars += pars  # override default values
        pars = self.pars

        self.verbose = pars["verbose"] if "verbose" in pars else 1

        # Set the random seed
        if pars.seed is None:
            now = datetime.now()  # noqa: DTZ005
            pars.seed = now.microsecond ^ int(now.timestamp())
            if self.verbose >= 1:
                sc.printred(f"No seed provided. Using random seed of {pars.seed}.")
        set_seed(pars.seed)

        # Setup time
        self.t = 0  # Current timestep
        self.nt = (
            pars.dur + 1
        )  # Number of timesteps. We add 1 to include step 0 (initial conditions) and then run for pars.dur steps. Individual components can have their own step sizes
        self.datevec = lp.daterange(self.pars["start_date"], days=self.nt)  # Time represented as an array of datetime objects

        # Setup early stopping option - controlled in DiseaseState_ABM component
        self.should_stop = False

        # Initialize the population
        if self.verbose >= 1:
            sc.printcyan("Initializing simulation...")

    def __init__(self, pars: PropertySet = None, verbose=1):
        start_time = time.perf_counter()
        self.common_init(pars, verbose)

        pars.n_ppl = np.atleast_1d(pars.n_ppl).astype(int)  # Ensure pars.n_ppl is an array
        if (pars.cbr is not None) & (len(pars.cbr) == 1):
            capacity = int(1.1 * calc_capacity(np.sum(pars.n_ppl), self.nt, pars.cbr[0]))
        elif (pars.cbr is not None) & (len(pars.cbr) > 1):
            capacity = int(1.1 * calc_capacity(np.sum(pars.n_ppl), self.nt, np.mean(pars.cbr)))
        else:
            capacity = int(np.sum(pars.n_ppl))
        self.people = LaserFrameIO(capacity=capacity, initial_count=int(np.sum(pars.n_ppl)))
        # We initialize disease_state here since it's required for most other components (which facilitates testing)
        self.people.add_scalar_property("disease_state", dtype=np.int32, default=-1)  # -1=Dead/inactive, 0=S, 1=E, 2=I, 3=R
        self.people.disease_state[: self.people.count] = 0  # Set initial population as susceptible
        self.results = LaserFrame(capacity=1)

        # Setup spatial component with node IDs
        self.people.add_scalar_property("node_id", dtype=np.int32, default=0)
        if hasattr(pars, "node_lookup") and pars.node_lookup is not None:
            ordered_node_ids = list(pars.node_lookup.keys())
            self.nodes = np.array(ordered_node_ids)
            node_ids = np.concatenate([np.full(count, node_id) for node_id, count in zip(ordered_node_ids, pars.n_ppl, strict=False)])
            self.people.node_id[0 : np.sum(pars.n_ppl)] = node_ids
        else:
            self.nodes = np.arange(len(np.atleast_1d(pars.n_ppl)))
            node_ids = np.concatenate([np.full(count, i) for i, count in enumerate(pars.n_ppl)])
            self.people.node_id[0 : np.sum(pars.n_ppl)] = node_ids  # Assign node IDs to initial people

        # Components
        self._components = []

        end_time = time.perf_counter()
        self.component_times[self.__class__.__name__ + ".__init__()"] += end_time - start_time

    @classmethod
    def init_from_file(cls, filename: str, pars: PropertySet = None):
        logger.info(f"Initializing SEIR_ABM from file: {filename}")

        # initialize model
        model = cls.__new__(cls)
        #model.pars = deepcopy(lp.default_pars)
        #if pars is not None:
        #    model.pars += pars
        model.common_init(pars, verbose=2)  # TBD: add nasty verbose param
        # model.nt = model.pars["dur"] + 1
        # model.datevec = lp.daterange(model.pars["start_date"], days=model.nt)

        # Use LaserFrameIO to load people
        model.people = LaserFrameIO.load(filename)

        # Setup node list
        model.nodes = np.unique(model.people.node_id[: model.people.count])

        # Results holder
        model.results = LaserFrameIO(capacity=1)

        # Components container
        model.components = []

        return model

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
        self.instances = []
        for cls in ordered_subset:
            start_time = time.perf_counter()
            self.instances.append(cls(self))
            end_time = time.perf_counter()
            self.component_times[cls.__name__ + ".__init__()"] += end_time - start_time

        if self.verbose >= 2:
            print(f"Initialized components: {self.instances}")

    def run(self):
        if self.verbose >= 1:
            sc.printcyan("Initialization complete. Running simulation...")

        with alive_bar(self.nt, title="Simulation progress:", disable=self.verbose < 1) as bar:
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
                        self.component_times[component.__class__.__name__ + ".step()"] += end_time - start_time

                    self.log_results(tick)
                    self.t += 1

                    # Early stopping rule
                    if self.should_stop:
                        if self.verbose >= 1:
                            sc.printyellow(
                                f"[SEIR_ABM] Early stopping at t={self.t}: no E/I and no future seed_schedule events. This stops all components (e.g., no births, deaths, or vaccination)"
                            )
                        break

                bar()  # Update the progress bar

        # logger.info("Simulation complete.") # cyan
        if self.verbose >= 1:
            sc.printcyan("Simulation complete.")

    def log_results(self, t):
        for component in self.instances:
            start_time = time.perf_counter()
            component.log(t)
            end_time = time.perf_counter()
            self.component_times[component.__class__.__name__ + ".log()"] += end_time - start_time

    def plot(self, save=False, results_path=None):
        if save:
            plt.ioff()  # Turn off interactive mode
            if results_path is None:
                raise ValueError("Please provide a results path to save the plots.")
            else:
                results_path = Path(results_path)  # Ensure results_path is a Path object
                results_path.mkdir(parents=True, exist_ok=True)

            # logger.info("Saving plots in " + str(results_path)) # cyan?
            if self.verbose >= 1:
                sc.printcyan("Saving plots in " + str(results_path))

        for component in self.instances:
            component.plot(save=save, results_path=results_path)
        self.plot_node_pop(save=save, results_path=results_path)

        if self.component_times:
            # logger.debug(f"{self.instances=}")
            if self.verbose >= 2:
                print(f"{self.instances=}")
            plt.figure(figsize=(12, 12))

            total_time = sum(self.component_times.values())
            threshold = 1  # 1%
            # Set label to None if the percentage is less than threshold
            labels = list(
                map(
                    lambda k, v: k if (v / total_time) > (threshold / 100) else None,
                    self.component_times.keys(),
                    self.component_times.values(),
                )
            )

            plt.pie(
                x=self.component_times.values(),
                labels=labels,
                autopct=lambda pct: f"{pct:1.1f}%" if pct > threshold else "",  # show percentage only if greater than threshold
                pctdistance=0.85,  # distance of percentage from center (0.6 is the default)
                labeldistance=1.1,  # distance of labels from center (1.1 is the default)
                radius=0.9,  # radius of the pie chart (1.0 is the default)
                # rotatelabels=True,  # rotate labels (False is the default)
            )

            plt.title(f"Time Spent in Each Component ({sum(self.component_times.values()):.2f} seconds)")
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


@nb.njit(parallel=True)
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
    @classmethod
    def init_from_file(cls, sim):
        # Alternate constructor: skip initialization logic
        self = cls.__new__(cls)  # bypass __init__
        self._common_init(sim)
        # Only set up results arrays if needed
        self._initialize_results_arrays()
        return self

    def _common_init(self, sim):
        self.sim = sim
        self.people = sim.people
        self.pars = sim.pars
        self.nodes = sim.nodes
        self.results = sim.results
        self.verbose = self.pars["verbose"] if "verbose" in self.pars else 1
        from collections import defaultdict

        # Schedule additional infections (time → list of (node_id, prevalence))
        self.seed_schedule = defaultdict(list)
        if self.pars.seed_schedule is not None:
            for entry in self.pars.seed_schedule:
                if "date" in entry and "dot_name" in entry:
                    date = lp.date(entry["date"])
                    t = (date - self.pars.start_date).days
                    node_id = next((nid for nid, info in self.pars.node_lookup.items() if info["dot_name"] == entry["dot_name"]), None)
                    if node_id is not None:
                        self.seed_schedule[t].append((node_id, entry["prevalence"]))
                elif "timestep" in entry and "node_id" in entry:
                    self.seed_schedule[entry["timestep"]].append((entry["node_id"], entry["prevalence"]))

    def _initialize_results_arrays(self):
        self.results.add_array_property("S", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.add_array_property("E", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.add_array_property("I", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.add_array_property("R", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.add_array_property("paralyzed", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)

    def __init__(self, sim):
        self._common_init(sim)
        self._initialize_results_arrays()
        self.verbose = sim.pars["verbose"] if "verbose" in sim.pars else 1

        # Setup the SEIR components
        sim.people.add_scalar_property("paralyzed", dtype=np.int32, default=0)
        # Initialize all agents with an exposure_timer & infection_timer
        sim.people.add_scalar_property("exposure_timer", dtype=np.int32, default=0)
        # Subtract 1 to account for the fact that we expose people in transmission component after the disease state component (newly exposed miss their first timer decrement)
        sim.people.exposure_timer[:] = self.pars.dur_exp(self.people.capacity) - 1
        sim.people.add_scalar_property("infection_timer", dtype=np.int32, default=0)
        sim.people.infection_timer[:] = self.pars.dur_inf(self.people.capacity)

        pars = self.pars

        def do_init_imm():
            # logger.debug(f"Before immune initialization, we have {sim.people.count} active agents.")
            if self.verbose >= 2:
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

                # logger.debug(f"After immune initialization and EULA-gizing, we have {sim.people.count} active agents.")
                if self.verbose >= 2:
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
            for node, prev in tqdm(
                enumerate(pars.init_prev), total=len(pars.init_prev), desc="Seeding infections in nodes", disable=self.verbose < 2
            ):
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

        # Seed infections after initialization
        t = self.sim.t
        if t in self.seed_schedule:
            for node_id, prevalence in self.seed_schedule[t]:
                node_mask = (self.people.node_id[: self.people.count] == node_id) & (self.people.disease_state[: self.people.count] >= 0)
                candidates = np.where(node_mask)[0]
                n_seed = int(len(candidates) * prevalence)
                if n_seed > 0:
                    selected = np.random.choice(candidates, size=n_seed, replace=False)
                    self.people.disease_state[selected] = 2  # Set to infectious
                    if self.verbose >= 1:
                        print(f"[DiseaseState_ABM] t={t}: Seeded {n_seed} infections in node {node_id}")

        # Optional early stopping rule if no cases or seed_schedule events remain
        if self.pars["stop_if_no_cases"]:
            any_exposed = np.sum(self.sim.results.E[self.sim.t - 1, :]) > 0
            any_infected = np.sum(self.sim.results.I[self.sim.t - 1, :]) > 0
            future_seeds = any(t > self.sim.t for t in self.seed_schedule)

            if not (any_exposed or any_infected or future_seeds):
                self.sim.should_stop = True

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
        lats = [self.pars.node_lookup[i]["lat"] for i in self.nodes]
        lons = [self.pars.node_lookup[i]["lon"] for i in self.nodes]
        # Scale population for plotting (adjust scale_factor as needed)
        scale_factor = 0.01  # tweak this number to look good visually
        sizes = [pop * scale_factor for pop in self.pars.n_ppl]

        # Get global min and max for consistent color scale
        infection_min = np.min(self.results.I)
        infection_max = np.max(self.results.I)

        for i, ax in enumerate(axs[:n_panels]):  # Ensure we don't go out of bounds
            t = timepoints[i]
            infection_counts = self.results.I[t, :]

            scatter = ax.scatter(
                lons, lats, c=infection_counts, s=sizes, cmap="RdYlBu_r", edgecolors=None, alpha=0.9, vmin=infection_min, vmax=infection_max
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

            # ax.set_facecolor("linen")

        # Add a single colorbar for all plots
        fig.colorbar(scatter, ax=axs, location="right", fraction=0.05, pad=0.05, label="Infection Count")
        # fig.patch.set_facecolor("lightgrey")

        # Add title
        fig.suptitle("Infected Population by Node", fontsize=16)

        if save:
            if results_path is None:
                raise ValueError("Please provide a results path to save the plots.")
            plt.savefig(f"{results_path}/infected_map.png")
        else:
            plt.show()


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


def classic_infect(node_ids, exposure_probs, disease_state, new_infections):
    """
    Classic agent-level infection: for each susceptible agent, draw a random number and expose if r < p.

    This function is compatible with the fast_infect() signature and can be swapped in via 'infection_method' param.

    Parameters:
        node_ids (np.ndarray): Array of node IDs for each person (int32).
        exposure_probs (np.ndarray): Per-agent exposure probability (float32, [0, 1]).
        disease_state (np.ndarray): Array of disease states (0 = S, 1 = E, 2 = I, 3 = R).
        new_infections (np.ndarray): Ignored in this function; included for API compatibility.

    Returns:
        new_exposures_by_node (np.ndarray): Number of new exposures assigned per node.
    """
    susceptible = disease_state == 0
    rand_vals = np.random.rand(len(disease_state))
    will_be_exposed = (rand_vals < exposure_probs) & susceptible

    # Expose the selected individuals
    disease_state[will_be_exposed] = 1

    # Tally new exposures per node
    n_nodes = new_infections.shape[0]
    exposed_nodes = node_ids[will_be_exposed]
    new_exposures_by_node = np.bincount(exposed_nodes, minlength=n_nodes)

    return new_exposures_by_node


@nb.njit(parallel=True)
def fast_infect(node_ids, exposure_probs, disease_state, new_infections):
    """
    A Numba-accelerated version of faster_infect.
    Parallelizes over nodes, computing a CDF for each node's susceptible population.
    Selects 'n_to_draw' indices via binary search of random values, and marks them as exposed.

    NOTE: This version does NOT enforce uniqueness of selected indices within the same node.
    """
    num_nodes = len(new_infections)
    n_people = len(node_ids)
    n_new_exposures = np.zeros(num_nodes, dtype=np.int32)

    # 1A) Calculate the number of susceptible individuals in each node
    # This is done in parallel to speed up the process

    # Thread-local storage
    local_sums = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)
    # Parallel loop
    for i in nb.prange(n_people):
        if disease_state[i] == 0:  # susceptible
            nd = node_ids[i]
            local_sums[nb.get_thread_id(), nd] += 1
    # Merge
    susceptible_sums = local_sums.sum(axis=0)  # Sum across threads

    for node in nb.prange(num_nodes):
        n_to_draw = new_infections[node]
        if n_to_draw <= 0:
            continue

        # 1B) Get and check count of susceptible agents in _this_ node
        sus_count = susceptible_sums[node]
        if sus_count == 0:
            continue

        # Step 2: Collect indices of susceptible individuals in this node
        # and copy their exposure probabilities into a new array
        # We need the copy in case we need to retry sampling to get unique indices
        sus_indices = np.empty(sus_count, dtype=np.int32)
        sus_probs = np.empty(sus_count, dtype=np.float32)
        idx = 0
        for i in range(n_people):
            if (node_ids[i] == node) and (disease_state[i] == 0):
                sus_indices[idx] = i
                sus_probs[idx] = exposure_probs[i]
                idx += 1

        # Step 3: Choose unique indices from susceptible population
        # using variation of NumPy random.choice()
        n_uniq = 0  # How many unique indices have we selected so far
        p = sus_probs  # alias sus_probs because the original algorithm uses p
        size = n_to_draw  # alias n_to_draw because the original algorithm uses size
        while n_uniq < size:
            # The magic is here with the random probes, the cumulative sum of the weights,
            # which effectively makes each index scaled by its weight,
            # and the binary search to find the indices.

            # Easy example, imagine two susceptible individuals, one with p=0.1 and one with p=0.9
            # If we draw a random number x in [0..1), we can find the index of the individual
            # that will be exposed by searching for the index of the first element in the cumulative
            # sum of the weights that is greater than x, which is much more likely to be the second
            # individual than the first.

            x = np.random.rand(size - n_uniq)  # Random values for sampling [0..1)
            cdf = np.cumsum(p)  # cumsum of weights for searching
            if cdf[-1] == 0:  # exit early if no susceptibles remaining
                break
            # Binary search for indices, modify x to be in [0..cdf[-1])
            # One multiply vs thousands of divides for cdf /= cdf[-1]
            indices = np.searchsorted(cdf, x * cdf[-1], side="right")
            indices = np.unique(indices)  # unique indices only
            disease_state[sus_indices[indices]] = 1  # expose the chosen individuals
            n_new_exposures[node] += indices.size  # update the count with new exposures
            n_uniq += indices.size  # update the number of unique indices selected
            if n_uniq < size:  # if we haven't selected enough unique indices, we need to retry
                p[indices] = 0.0  # set the probabilities for the selected indices to zero

    return n_new_exposures


def efsp_infect(node_ids, exposure_probs, disease_state, new_infections):
    """
    Infect agents using precomputed number of infections per node, using Efraimidis-Spirakis
    weighted sampling without replacement.

    Parameters:
        node_ids (np.ndarray): Array of node IDs per agent (int32).
        exposure_probs (np.ndarray): Exposure probabilities (float32).
        disease_state (np.ndarray): Disease states (0 = S, 1 = E, 2 = I, 3 = R).
        new_infections (np.ndarray): Number of infections to assign per node (int32).

    Returns:
        np.ndarray: Number of new exposures assigned per node (same shape as new_infections).
    """
    n_nodes = new_infections.shape[0]
    new_exposures_by_node = np.zeros(n_nodes, dtype=np.int32)

    # Loop over each node with infections to assign
    for node in np.flatnonzero(new_infections):
        # Get agent indices for this node
        in_node = np.flatnonzero((node_ids == node) & (disease_state == 0))
        if in_node.size == 0:
            continue

        # Weights for these susceptible individuals
        weights = exposure_probs[in_node]
        weights = np.clip(weights, 1e-8, 1.0)  # Avoid division by 0 or unstable keys

        k = min(new_infections[node], in_node.size)  # Can't infect more than available
        if k == 0:
            continue

        # Efraimidis-Spirakis: sample k indices based on weights
        keys = np.log(np.random.rand(in_node.size)) / weights
        top_k_indices = np.argpartition(keys, k)[:k]
        selected_agents = in_node[top_k_indices]

        # Infect them
        disease_state[selected_agents] = 1
        new_exposures_by_node[node] = k

    return new_exposures_by_node


def chunk_infect(node_ids, exposure_probs, disease_state, new_infections, chunk_size=1000):
    """
    Efficiently sample exactly the specified number of new infections from weighted susceptibles in each node.

    Parameters:
        node_ids (np.ndarray): Array of node IDs for each person.
        exposure_probs (np.ndarray): Per-person exposure probability (not necessarily normalized).
        disease_state (np.ndarray): Array of disease states (0 = susceptible).
        new_infections (np.ndarray): Number of infections to assign per node.
        chunk_size (int): How many agents to process at once per chunk.

    Returns:
        infected_by_node (np.ndarray): Array of actual infections performed per node.
        And updates disease_state in place (sets selected susceptibles to 1).

    """
    num_nodes = len(new_infections)
    infected_by_node = np.zeros(num_nodes, dtype=np.int32)

    for node in range(num_nodes):
        n_draw = new_infections[node]
        if n_draw <= 0:
            continue
        selected = set()  # Store indices to infect

        # Step 1: Get susceptible individuals in this node
        susceptible = (node_ids == node) & (disease_state == 0)
        sus_indices = np.where(susceptible)[0]
        if len(sus_indices) == 0:
            continue

        # Step 2: Chunking loop
        for chunk_start in range(0, len(sus_indices), chunk_size):
            chunk_sus_inds = sus_indices[chunk_start : chunk_start + chunk_size]  # Get the sus indices in the chunk
            chunk_probs = exposure_probs[chunk_sus_inds]  # Get the exposure probabilities for sus in the chunk

            # Calc the number of infections to make in this chunk
            prob_sum = np.sum(chunk_probs)  # Sum the exposure probabilities in the chunk
            if prob_sum == 0:
                continue

            weights = chunk_probs / prob_sum  # Normalize the probabilities since np.random.choice expects p to sum to 1
            k = min(n_draw - len(selected), len(chunk_sus_inds))  # Number of draws to make in this chunk
            if k <= 0:
                break

            # Select individuals to infect
            draws = np.random.choice(chunk_sus_inds, size=k, replace=False, p=weights)
            selected.update(draws)

            if len(selected) >= n_draw:
                break

        # Optional fallback if we still haven't filled the target
        if len(selected) < n_draw:
            missing = n_draw - len(selected)
            fallback_pool = np.setdiff1d(sus_indices, list(selected), assume_unique=True)
            fill = fallback_pool[:missing]  # take as many as we can
            selected.update(fill)

        # Step 3: Infect selected individuals
        selected = list(selected)[:n_draw]  # clip if over-selected
        disease_state[selected] = 1
        infected_by_node[node] = len(selected)

    return infected_by_node


@nb.njit((nb.int32[:], nb.int32[:], nb.int32[:], nb.int32, nb.int32), nogil=True, cache=True)
def count_SEIRP(node_id, disease_state, paralyzed, n_nodes, n_people):
    """
    Go through each person exactly once and increment counters for their node.

    node_id:        array of node IDs for each individual
    disease_state:  array storing each person's disease state (-1=dead/inactive, 0=S, 1=E, 2=I, 3=R)
    paralyzed:      array (0 or 1) if the person is paralyzed
    n_nodes:        total number of nodes

    Returns: S, E, I, R, P arrays, each length n_nodes
    """

    S = np.zeros(n_nodes, dtype=np.int64)
    E = np.zeros(n_nodes, dtype=np.int64)
    I = np.zeros(n_nodes, dtype=np.int64)
    R = np.zeros(n_nodes, dtype=np.int64)
    P = np.zeros(n_nodes, dtype=np.int64)

    # Single pass over the entire population
    for i in nb.prange(n_people):
        if disease_state[i] >= 0:  # Only count those who are alive
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
        self.verbose = sim.pars["verbose"] if "verbose" in sim.pars else 1

        # Stash the R0 scaling factor
        self.r0_scalars = self.pars.r0_scalars

        self._initialize_people_fields()
        self._initialize_common()

    @classmethod
    def init_from_file(cls, sim):
        """Alternative constructor for loading from file without resetting people."""
        instance = cls.__new__(cls)
        instance.sim = sim
        instance.people = sim.people
        instance.nodes = np.arange(len(sim.pars.n_ppl))
        instance.pars = sim.pars
        instance.results = sim.results
        instance.r0_scalars = instance.pars.r0_scalars
        instance.verbose = sim.pars["verbose"] if "verbose" in sim.pars else 1

        # Skip sampling & property setting
        instance._initialize_common()
        return instance

    def _initialize_people_fields(self):
        """Initialize individual-level transmission properties."""
        self.people.add_scalar_property("acq_risk_multiplier", dtype=np.float32, default=1.0)
        self.people.add_scalar_property("daily_infectivity", dtype=np.float32, default=1.0)

        mean_ln = 1
        var_ln = self.pars.risk_mult_var
        mu_ln = np.log(mean_ln**2 / np.sqrt(var_ln + mean_ln**2))
        sigma_ln = np.sqrt(np.log(var_ln / mean_ln**2 + 1))

        mean_gamma = self.pars.r0 / np.mean(self.pars.dur_inf(1000))
        scale_gamma = max(mean_gamma / 1, 1e-10)

        rho = 0.8
        L = np.linalg.cholesky([[1, rho], [rho, 1]])
        if not hasattr(self.people, "true_capacity"):
            self.people.true_capacity = self.people.capacity
        n = self.people.true_capacity

        # Record new exposure counts aka incidence
        # Pretty sure this code from after merge belongs somewhere else. This is NOT for init_from_file. Think...
        # self.sim.results.add_array_property("new_exposed", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)

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
        # Set individual heterogeneity properties
        if self.pars.individual_heterogeneity:
            self.people.acq_risk_multiplier[: self.people.true_capacity] = acq_risk_multiplier
            self.people.daily_infectivity[: self.people.true_capacity] = daily_infectivity
        else:
            sc.printyellow("Warning: manually resetting acq_risk_multiplier and daily_infectivity to 1.0 for testing")
            self.people.acq_risk_multiplier[: self.people.true_capacity] = 1.0
            self.people.daily_infectivity[: self.people.true_capacity] = mean_gamma

        z = np.random.normal(size=(n, 2)) @ L.T
        self.people.acq_risk_multiplier[:n] = np.exp(mu_ln + sigma_ln * z[:, 0])
        self.people.daily_infectivity[:n] = stats.gamma.ppf(stats.norm.cdf(z[:, 1]), a=1, scale=scale_gamma)

    def _initialize_common(self):
        """Initialize shared network and timers."""
        # Compute the infection migration network
        self.sim.results.add_vector_property("network", length=len(self.sim.nodes), dtype=np.float32)
        self.network = self.sim.results.network
        init_pops = self.sim.pars.n_ppl
        # Get the distance matrix
        if self.sim.pars.distances is not None:
            dist_matrix = self.sim.pars.distances
        else:
            # Calculate the distance matrix based on the Haversine formula
            node_lookup = self.sim.pars.node_lookup
            n_nodes = len(self.sim.nodes)
            node_ids = sorted(node_lookup.keys())
            node_lookup = self.sim.pars.node_lookup
            lats = np.array([node_lookup[i]["lat"] for i in node_ids])
            lons = np.array([node_lookup[i]["lon"] for i in node_ids])
            dist_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    dist_matrix[i, j] = distance(lats[i], lons[i], lats[j], lons[j])
        # Setup the network
        if self.pars.migration_method.lower() == "gravity":
            k, a, b, c = self.pars.gravity_k, self.pars.gravity_a, self.pars.gravity_b, self.pars.gravity_c
            self.network = gravity(init_pops, dist_matrix, k, a, b, c)
            self.network /= np.power(init_pops.sum(), c)  # Normalize
        elif self.pars.migration_method.lower() == "radiation":
            k = self.pars.radiation_k
            self.network = radiation(init_pops, dist_matrix, k, include_home=False)
        else:
            raise ValueError(f"Unknown migration method: {self.pars.migration_method}")
        # Normalize so that each row sums to a max of max_migr_frac, else uses the unnormalized values
        self.network = row_normalizer(self.network, self.pars.max_migr_frac)

        self.beta_sum_time = 0
        self.spatial_beta_time = 0
        self.seasonal_beta_time = 0
        self.probs_time = 0
        self.calc_ni_time = 0
        self.do_ni_time = 0

        self.sim.results.add_array_property("new_exposed", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)

        # Map infection method to function
        method = self.pars["infection_method"].lower()
        if method == "fast":
            self.infect_fn = fast_infect
        elif method == "classic":
            self.infect_fn = classic_infect
        elif method == "efsp":
            self.infect_fn = efsp_infect
        else:
            raise ValueError(f"Unknown infection method: {method}")

    def step(self):
        # Manual debugging of transmission
        if self.verbose >= 3:
            # Log timestep
            logger.info(f"TIMESTEP: {self.sim.t}")

            # # Go node by node
            # ds = self.people.disease_state[: self.people.count]
            # node_id = self.people.node_id[: self.people.count]
            # daily_infectivity = self.people.daily_infectivity[: self.people.count]
            # risk = self.people.acq_risk_multiplier[: self.people.count]

            # infecteds = np.where(ds == 2)
            # infected_nodes = np.unique(node_id[infecteds])
            # for node in infected_nodes:
            #     # Number of infecteds
            #     num_alive = np.sum((node_id == node) & (ds >= 0)) + self.sim.results.R[self.sim.t][node]
            #     num_susceptibles = np.sum((node_id == node) & (ds == 0))
            #     num_infecteds = np.sum((node_id == node) & (ds == 2))

            #     # Calc beta for this node
            #     node_beta_sum = np.sum(daily_infectivity[(node_id == node) & (ds == 2)])
            #     beta_seasonality = lp.get_seasonality(self.sim)
            #     r0_scalar = self.r0_scalars[node]
            #     beta = node_beta_sum * beta_seasonality * r0_scalar
            #     per_agent_infection_rate = beta / np.clip(num_alive, 1, None)
            #     base_prob_infection = 1 - np.exp(-per_agent_infection_rate)
            #     mean_risk = np.mean(risk[(node_id == node) & (ds == 0)])
            #     exp_infections_using_infectivity = base_prob_infection * num_susceptibles

            #     # Back of the envelope calculation
            #     R0 = self.sim.pars.r0
            #     infectious_period = 24
            #     beta_manual = R0 / infectious_period
            #     lambda_ = beta_manual * num_infecteds / num_alive * beta_seasonality * r0_scalar
            #     p_infection = 1 - np.exp(-lambda_)
            #     exp_infections_manual = p_infection * num_susceptibles

            #     # Log detailed node-level stats
            #     logger.info(
            #         f"MANUAL CALCS: "
            #         f"Node {node}: S={num_susceptibles}, I={num_infecteds}, Alive={num_alive}, frac_I={num_infecteds / num_alive:.2f}, frac_S={num_susceptibles / num_alive:.2f}, "
            #     )
            #     logger.info(
            #         f"Exp infections (back of the envelope) (lambda = R0 / inf_period * I / N * scalars; new I = S * (1-exp(-lambda)))={exp_infections_manual:.2f}, "  # Obs infectivity={obs_infectivity:.2f}, Exp infectivity={exp_infectivity:.2f},
            #     )
            #     # logger.info(
            #     #     f"r0={self.sim.pars.r0}, beta_seasonality={beta_seasonality:.4f}, r0_scalar={r0_scalar:.4f}, beta={beta:.4f}, "
            #     #     f"base_prob_infection={base_prob_infection:.4f}, mean_risk={mean_risk:.4f}, "
            #     # )
            #     logger.info(f"Exp infections (using individual infectivity) ={exp_infections_using_infectivity:.2f}, ")

        # 1) Sum up the total amount of infectivity shed by all infectious agents within a node.
        # This is the daily number of infections that these individuals would be expected to generate
        # in a fully susceptible population sans spatial and seasonal factors.
        # check_time = time.perf_counter()
        disease_state = self.people.disease_state[: self.people.count]
        node_ids = self.people.node_id[: self.people.count]
        infectivity = self.people.daily_infectivity[: self.people.count]
        risk = self.people.acq_risk_multiplier[: self.people.count]
        # node_beta_sums = fast_beta()
        # logger.valid(f"node_beta_sums={np.array2string(node_beta_sums, separator=',', max_line_width=9999)}")
        node_beta_sums = compute_beta_ind_sums(node_ids, infectivity, disease_state, len(self.nodes))
        if self.verbose >= 3:
            n_infected = []
            for node in self.sim.nodes:
                # num_alive = np.sum((node_ids == node) & (disease_state >= 0)) + self.sim.results.R[self.sim.t][node]
                num_susceptibles = np.sum((node_ids == node) & (disease_state == 0))
                n_I_node = np.sum((node_ids == node) & (disease_state == 2))
                n_infected.append(n_I_node)
            n_infected = np.array(n_infected)
            exp_node_beta_sums = n_infected * self.sim.pars.r0 / np.mean(self.sim.pars.dur_inf(1000))
            logger.info(f"Expected node beta sums: {fmt(exp_node_beta_sums, 2)}")
            node_beta_sums_pre = node_beta_sums.copy()
            logger.info(f"Node beta sums (pre-transfer): {fmt(node_beta_sums_pre, 2)}")
            logger.info(f"Total node beta sums (pre-transfer): {fmt(node_beta_sums_pre.sum(), 2)}")

        # 2) Spatially redistribute infectivity among nodes
        transfer = (node_beta_sums * self.network).astype(np.float64)  # Don't round here, we'll handle fractional infections later
        # Ensure net contagion remains positive after movement
        node_beta_sums += transfer.sum(axis=1) - transfer.sum(axis=0)
        node_beta_sums = np.maximum(node_beta_sums, 0)  # Prevent negative contagion

        # logger.valid(f"node_beta_sums [post-migration]={np.array2string(node_beta_sums, separator=',', max_line_width=9999)}")
        if self.verbose >= 3:
            logger.info(f"Node beta sums (post-transfer): {fmt(node_beta_sums, 2)}")
            logger.info(f"Total Node beta sums (post-transfer): {fmt(node_beta_sums.sum(), 2)}")

        # 3) Apply seasonal & geographic modifiers
        beta_seasonality = lp.get_seasonality(self.sim)
        beta = node_beta_sums * beta_seasonality * self.r0_scalars  # Total node infection rate
        # logger.valid( f"{beta=}" )
        if self.verbose >= 3:
            logger.info(f"beta_seasonality: {fmt(beta_seasonality, 2)}")
            logger.info(f"R0 scalars: {fmt(self.r0_scalars, 2)}")
            logger.info(f"beta: {fmt(beta, 2)}")
            logger.info(f"Total beta: {fmt(beta.sum(), 2)}")

        # 4) Calculate base probability for each agent to become exposed
        alive_counts = (
            self.sim.results.S[self.sim.t - 1]
            + self.sim.results.E[self.sim.t - 1]
            + self.sim.results.I[self.sim.t - 1]
            + self.sim.results.R[self.sim.t - 1]
            + self.sim.results.births[self.sim.t]
            - self.sim.results.deaths[self.sim.t]
        )
        per_agent_infection_rate = beta / np.clip(alive_counts, 1, None)
        base_prob_infection = 1 - np.exp(-per_agent_infection_rate)

        # logger.valid( f"{base_prob_infection=}" )
        if self.verbose >= 3:
            logger.info(f"Alive counts: {fmt(alive_counts, 2)}")
            logger.info(f"Per agent infection rate: {fmt(per_agent_infection_rate, 2)}")
            logger.info(f"Base prob infection: {fmt(base_prob_infection, 2)}")
            logger.info(f"Exp inf (sans acq risk): {fmt(num_susceptibles * base_prob_infection, 2)}")

        # 5) Calculate infections
        exposure_sums = compute_infections_nb(disease_state, node_ids, risk, base_prob_infection)
        new_infections = np.random.poisson(exposure_sums).astype(np.int32)
        # logger.valid( f"{new_infections=}" )
        if self.verbose >= 3:
            logger.info(f"exposure_sums: {fmt(exposure_sums, 2)}")
            logger.info(f"Expected new exposures: {new_infections}")

        # 6) Draw n_expected_exposures for each node according to their exposure_probs
        exposure_probs = base_prob_infection[node_ids] * risk  # Try adding in node-level force & personal risk
        if self.verbose >= 3:
            disease_state_pre_infect = disease_state.copy()
        new_exposed = self.infect_fn(node_ids, exposure_probs, disease_state, new_infections)
        self.sim.results.new_exposed[self.sim.t, :] = new_exposed
        if self.verbose >= 3:
            logger.info(f"Observed new exposures: {new_exposed}")
            total_expected = np.sum(exposure_sums)
            tot_poisson_draw = np.sum(new_infections)
            # Check the number of people that are newly exposed
            num_new_exposed = np.sum(disease_state == 1) - np.sum(disease_state_pre_infect == 1)
            logger.info(
                f"Tot exp infections: {total_expected:.2f}, Total pois draw: {tot_poisson_draw}, Tot realized infections: {num_new_exposed}"
            )

    def log(self, t):
        # Get the counts for each node in one pass
        S_counts, E_counts, I_counts, R_counts, P_counts = count_SEIRP(
            self.people.node_id,
            self.people.disease_state,
            self.people.paralyzed,
            np.int32(len(self.nodes)),
            np.int32(self.people.count),
        )

        # Store them in results
        self.results.S[t, :] = S_counts
        self.results.E[t, :] = E_counts
        self.results.I[t, :] = I_counts
        # Note that we add to existing non-zero EULA values for R
        self.results.R[t, :] += R_counts
        self.results.paralyzed[t, :] = P_counts

        if self.verbose >= 3:
            logger.info(f"Exposed logged at end of timestep: {self.results.E[t, :]}")
            logger.info("")

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
        self._common_init(sim)
        self._initialize_ages_and_births()
        self._initialize_deaths()
        self._initialize_birth_rates()

    @classmethod
    def init_from_file(cls, sim):
        """Minimal constructor for bootstrapped model state."""
        self = cls.__new__(cls)
        self._common_init(sim)
        self._initialize_birth_results_if_needed()
        self._initialize_birth_rates()
        cumulative_deaths = lp.create_cumulative_deaths(np.sum(self.pars.n_ppl), max_age_years=100)
        self.death_estimator = KaplanMeierEstimator(cumulative_deaths)
        return self

    def _common_init(self, sim):
        self.sim = sim
        self.people = sim.people
        self.nodes = sim.nodes
        self.results = sim.results
        self.pars = sim.pars
        self.step_size = self.pars.step_size_VitalDynamics_ABM
        self.verbose = sim.pars["verbose"] if "verbose" in sim.pars else 1

    def _initialize_ages_and_births(self):
        pars = self.pars
        if pars.age_pyramid_path is not None:
            self.people.add_scalar_property("date_of_birth", dtype=np.int32, default=-1)
            pyramid = load_pyramid_csv(pars.age_pyramid_path)
            sampler = AliasedDistribution(pyramid[:, 2] + pyramid[:, 3])  # male + female

            samples = sampler.sample(self.people.count)
            bin_min = pyramid[:, 0] * 365
            bin_max = (pyramid[:, 1] + 1) * 365
            mask = np.zeros(self.people.count, dtype=bool)
            ages = np.zeros(self.people.count, dtype=np.int32)

            for i in range(len(pyramid)):
                mask[:] = samples == i
                ages[mask] = np.random.randint(bin_min[i], bin_max[i], mask.sum())

            ages[ages == 0] = 1
            self.people.date_of_birth[: self.people.count] = -ages

    def _initialize_deaths(self):
        pars = self.pars
        if pars.cbr is not None:
            self.results.add_array_property("births", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
            self.results.add_array_property("deaths", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
            self.people.add_scalar_property("date_of_death", dtype=np.int32, default=0)

            cumulative_deaths = lp.create_cumulative_deaths(np.sum(pars.n_ppl), max_age_years=100)
            self.death_estimator = KaplanMeierEstimator(cumulative_deaths)

            # Only compute lifespans if date_of_birth was initialized
            if "date_of_birth" in self.people.__dict__:
                ages = -self.people.date_of_birth[: self.people.count]
                lifespans = self.death_estimator.predict_age_at_death(ages, max_year=100)
                dods = lifespans - ages
                self.people.date_of_death[: self.people.count] = dods

                # Compute life expectancies per node
                node_ids = self.people.node_id[: self.people.count]
                _, indices = np.unique(node_ids, return_inverse=True)
                weighted = np.bincount(indices, weights=lifespans / 365)
                counts = np.bincount(indices)

                n_nodes = len(self.nodes)
                life_expectancies = np.zeros(n_nodes)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean_lifespans = np.divide(weighted, counts, out=np.zeros_like(weighted), where=counts > 0)
                life_expectancies[: len(mean_lifespans)] = mean_lifespans
                pars.life_expectancies = life_expectancies

    def _initialize_birth_rates(self):
        pars = self.pars
        self.birth_rate = np.zeros(len(self.nodes))
        if pars.cbr is not None:
            if isinstance(pars.cbr, (float, int)) or len(pars.cbr) == 1:
                self.birth_rate[:] = pars.cbr[0] / (365 * 1000)
            else:
                self.birth_rate[:] = np.array(pars.cbr) / (365 * 1000)

    def _initialize_birth_results_if_needed(self):
        """For bootstrapped sims, add result arrays if not already present."""
        if "births" not in self.results.__dict__:
            self.results.add_array_property("births", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        if "deaths" not in self.results.__dict__:
            self.results.add_array_property("deaths", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)

    def step(self):
        t = self.sim.t
        if t % self.step_size != 0:
            return

        # 1) Get vital statistics - alive and newly deceased
        num_nodes = len(self.nodes)
        tl_alive = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)
        tl_dying = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)
        alive_count_by_node = np.zeros(num_nodes, dtype=np.int32)
        deaths_count_by_node = np.zeros(num_nodes, dtype=np.int32)
        get_vital_statistics(
            num_nodes,
            self.people.count,
            self.people.disease_state,
            self.people.node_id,
            self.people.date_of_death,
            t,
            tl_alive,
            tl_dying,
            alive_count_by_node,
            deaths_count_by_node,
        )

        # 2) Compute births
        expected_births = self.step_size * self.birth_rate * alive_count_by_node
        birth_integer = expected_births.astype(np.int32)
        birth_fraction = expected_births - birth_integer
        birth_rand = np.random.binomial(1, birth_fraction)  # Bernoulli draw
        births = birth_integer + birth_rand

        if (total_births := births.sum()) > 0:
            start, end = self.people.add(total_births)

            dobs = self.people.date_of_birth[start:end]
            dods = self.people.date_of_death[start:end]

            dobs[:] = 0  # temporarily
            dods[:] = self.death_estimator.predict_age_at_death(dobs, max_year=100)
            dobs[:] = t  # now set to current time
            self.people.disease_state[start:end] = 0
            dods[:] += t  # offset by current time
            # assign node IDs to newborns
            self.people.node_id[start:end] = np.repeat(np.arange(num_nodes), births)
            if any(isinstance(component, RI_ABM) for component in self.sim.components):
                self.people.ri_timer[start:end] = 182

            self.results.births[t] = births

        # 3) Store the death counts
        # Actual "death" handled in get_vital_statistics() as we count newly deceased
        self.results.deaths[t] = deaths_count_by_node

        return

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


@nb.njit(
    (nb.int32, nb.int32, nb.int32[:], nb.int32[:], nb.int32[:], nb.int32, nb.int32[:, :], nb.int32[:, :], nb.int32[:], nb.int32[:]),
    parallel=True,
    cache=True,
)
def get_vital_statistics(num_nodes, num_people, disease_state, node_id, date_of_death, t, tl_alive, tl_dying, num_alive, num_dying):
    # Iterate in parallel over all people
    for i in nb.prange(num_people):
        if disease_state[i] >= 0:  # If they're alive ...
            tl_alive[nb.get_thread_id(), node_id[i]] += 1  # Count 'em
            if date_of_death[i] <= t:  # If they're past their due date ...
                disease_state[i] = -1  # Mark them as deceased
                tl_dying[nb.get_thread_id(), node_id[i]] += 1  # Count 'em as deceased

    num_alive[:] = tl_alive.sum(axis=0)  # Merge per-thread results
    num_dying[:] = tl_dying.sum(axis=0)  # Merge per-thread results

    return


@nb.njit(parallel=True)
def fast_ri(
    step_size,
    node_id,
    disease_state,
    ri_timer,
    sim_t,
    vx_prob_ri,
    results_ri_vaccinated,
    rand_vals,
    count,
):
    """
    Optimized vaccination step with thread-local storage and parallel execution.
    """
    if sim_t % step_size != 0:  # Run only every 14th timestep
        return

    num_people = count
    num_nodes = results_ri_vaccinated.shape[1]  # Assuming shape (timesteps, nodes)
    num_threads = nb.get_num_threads()

    # Allocate per-thread local arrays
    local_vaccinated = np.zeros((num_threads, num_nodes), dtype=np.int32)

    # for i in np.arange(num_people):
    for i in nb.prange(num_people):
        thread_id = nb.get_thread_id()
        node = node_id[i]
        if disease_state[i] < 0:  # Skip dead or inactive agents
            continue

        prob_vx = vx_prob_ri[node]

        # if self.verbose >= 2:
        #     print(f"Agent {i} in disease state {disease_state[i]}")
        #     print("prob_vx=", prob_vx)

        ri_timer[i] -= step_size
        eligible = False
        # If first vx, account for the fact that no components are run on day 0
        if sim_t == step_size:
            eligible = ri_timer[i] <= 0 and ri_timer[i] >= -step_size
        elif sim_t > step_size:
            eligible = ri_timer[i] <= 0 and ri_timer[i] > -step_size

        if eligible:
            if rand_vals[i] < prob_vx:  # Check probability of vaccination
                local_vaccinated[thread_id, node] += 1  # Increment vaccinated count
                if disease_state[i] == 0:  # If susceptible
                    # We don't check for vx_eff here, since that is already accounted for in the prob_vx file
                    disease_state[i] = 3  # Move to Recovered state

    # Merge per-thread results
    for thread_id in range(num_threads):
        for j in range(num_nodes):
            results_ri_vaccinated[sim_t, j] += local_vaccinated[thread_id, j]


class RI_ABM:
    def __init__(self, sim):
        self.sim = sim
        self.step_size = sim.pars.step_size_RI_ABM
        self.people = sim.people
        self.nodes = sim.nodes
        self.pars = sim.pars
        self.results = sim.results
        self.verbose = self.pars["verbose"] if "verbose" in self.pars else 1

        # Only initialize people-based values if not loading from file
        self._initialize_people_fields()
        self._initialize_common()

    @classmethod
    def init_from_file(cls, sim):
        """Alternative constructor when loading people from disk."""
        instance = cls.__new__(cls)
        instance.sim = sim
        instance.step_size = sim.pars.step_size_RI_ABM
        instance.people = sim.people  # Already loaded from disk
        instance.nodes = sim.nodes
        instance.pars = sim.pars
        instance.results = sim.results

        # Skip setting `ri_timer`, just initialize shared parts
        instance._initialize_common()
        return instance

    def _initialize_people_fields(self):
        """Set RI timers and other properties from scratch."""

        # Calc date of RI (assume single point in time between 1st and 3rd dose)
        self.people.add_scalar_property("ri_timer", dtype=np.int32, default=-1)
        dob = self.people.date_of_birth
        days_from_birth_to_ri = np.random.uniform(42, 98, len(self.people.ri_timer))
        self.people.ri_timer = dob + days_from_birth_to_ri

    def _initialize_common(self):
        """Initialize common result arrays."""
        self.sim.results.add_array_property(
            "ri_vaccinated",
            shape=(self.sim.nt, len(self.sim.nodes)),
            dtype=np.int32,
        )
        self.results = self.sim.results

    def step(self):
        if self.pars["vx_prob_ri"] is None:
            return

        vx_prob_ri = self.pars["vx_prob_ri"]  # Includes coverage & efficacy
        num_nodes = len(self.sim.nodes)
        # Promote to 1D arrays if needed
        if np.isscalar(vx_prob_ri):
            vx_prob_ri = np.full(num_nodes, vx_prob_ri, dtype=np.float64)

        # Suppose we have num_people individuals
        rand_vals = np.random.rand(self.people.count)  # this could be done clevererly

        fast_ri(
            self.step_size,
            self.people.node_id,
            self.people.disease_state,
            self.people.ri_timer,
            self.sim.t,
            vx_prob_ri,
            self.results.ri_vaccinated,
            rand_vals,
            self.people.count,
        )

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


@nb.njit(parallel=True)
def fast_sia(
    node_ids,
    disease_states,
    dobs,
    sim_t,
    vx_prob,
    vx_eff,
    results_vaccinated,
    results_protected,
    rand_vals,
    count,
    nodes_to_vaccinate,
    min_age,
    max_age,
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
    num_nodes = results_vaccinated.shape[1]
    num_threads = nb.get_num_threads()

    # Pre-allocate thread-local result arrays
    local_vaccinated = np.zeros((num_threads, num_nodes), dtype=np.int32)
    local_protected = np.zeros((num_threads, num_nodes), dtype=np.int32)

    for i in nb.prange(num_people):
        thread_id = nb.get_thread_id()
        node = node_ids[i]

        # Skip if agent is not alive, not in targeted node, or not in age range
        if disease_states[i] < 0:
            continue
        if node not in nodes_to_vaccinate:
            continue
        age = sim_t - dobs[i]
        if not (min_age <= age <= max_age):
            continue

        prob_vx = vx_prob[node]
        r = rand_vals[i]

        if r < prob_vx:  # Check probability of vaccination
            local_vaccinated[thread_id, node] += 1  # Increment vaccinated count
            if disease_states[i] == 0:  # If susceptible
                if r < prob_vx * vx_eff:  # Check probability that vaccine takes/protects
                    disease_states[i] = 3  # Move to Recovered state
                    local_protected[thread_id, node] += 1  # Increment protected count

    # Aggregate thread-local counts into global result arrays
    results_vaccinated[sim_t] = local_vaccinated.sum(axis=0)
    results_protected[sim_t] = local_protected.sum(axis=0)


class SIA_ABM:
    def __init__(self, sim):
        self._common_init(sim)
        self._initialize_results()
        self._load_schedule()

    @classmethod
    def init_from_file(cls, sim):
        self = cls.__new__(cls)
        self._common_init(sim)
        self._initialize_results()
        self._load_schedule()
        return self

    def _common_init(self, sim):
        self.sim = sim
        self.people = sim.people
        self.nodes = sim.nodes
        self.pars = sim.pars
        self.results = sim.results
        self.verbose = sim.pars["verbose"] if "verbose" in sim.pars else 1

    def _initialize_results(self):
        self.results.add_array_property("sia_vaccinated", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.add_array_property("sia_protected", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)

    def _load_schedule(self):
        self.sia_schedule = [] if "sia_schedule" not in self.pars or self.pars["sia_schedule"] is None else self.pars["sia_schedule"]
        for event in self.sia_schedule:
            event["date"] = lp.date(event["date"])

    def step(self):
        t = self.sim.t  # Current timestep

        # Check if there is an SIA event today
        for event in self.sia_schedule:
            if event["date"] == self.sim.datevec[t]:
                if self.pars.vx_prob_sia is None:
                    continue
                nodes_to_vaccinate = np.array(event["nodes"], dtype=np.int32)  # Convert to NumPy array
                vx_prob_sia = np.array(self.pars["vx_prob_sia"], dtype=np.float32)  # Convert to NumPy array
                vaccinetype = event["vaccinetype"]
                vx_eff = self.pars["vx_efficacy"][vaccinetype]
                min_age, max_age = event["age_range"]

                # Suppose we have num_people individuals
                rand_vals = np.random.rand(self.people.count)  # this could be done clevererly
                fast_sia(
                    self.people.node_id,
                    self.people.disease_state,
                    self.people.date_of_birth,
                    self.sim.t,
                    vx_prob_sia,
                    vx_eff,
                    self.results.sia_vaccinated,
                    self.results.sia_protected,
                    rand_vals,
                    self.people.count,
                    nodes_to_vaccinate,
                    min_age,
                    max_age,
                )

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
