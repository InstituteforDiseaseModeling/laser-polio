import numbers
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
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
from laser_polio.logger import fmt
from laser_polio.logger import logger
from laser_polio.plots import plot_age_pyramid
from laser_polio.plots import plot_animated_strain_map
from laser_polio.plots import plot_animated_strain_map_panels
from laser_polio.plots import plot_cum_new_exposed_paralyzed
from laser_polio.plots import plot_cum_ri_vx
from laser_polio.plots import plot_cum_vx_sia
from laser_polio.plots import plot_infected_by_node
from laser_polio.plots import plot_infected_by_node_strain
from laser_polio.plots import plot_infected_choropleth
from laser_polio.plots import plot_infected_choropleth_by_strain
from laser_polio.plots import plot_infected_dot_map
from laser_polio.plots import plot_network
from laser_polio.plots import plot_new_exposed_by_strain
from laser_polio.plots import plot_node_pop
from laser_polio.plots import plot_total_seir_counts
from laser_polio.plots import plot_vital_dynamics
from laser_polio.utils import TimingStats

__all__ = ["RI_ABM", "SEIR_ABM", "SIA_ABM", "DiseaseState_ABM", "Transmission_ABM", "VitalDynamics_ABM"]


# SEIR Model
class SEIR_ABM:
    """
    An AGENT-BASED SEIR Model for polio
    Each entry in the population is an agent with a disease state and a node ID
    Disease state codes: 0=S, 1=E, 2=I, 3=R
    """

    def common_init(self, pars, verbose):
        self.perf_stats = TimingStats()
        with self.perf_stats.start(self.__class__.__name__ + ".__init__()"):
            # Load default parameters and optionally override with user-specified ones
            self.pars = deepcopy(lp.default_pars)
        if pars is not None:
            unexpected_keys = set(pars.to_dict().keys()) - set(self.pars.to_dict().keys())
            if unexpected_keys:
                sc.printred(f"Warning: ignoring unexpected parameters: {list(unexpected_keys)}")
                # Filter out unexpected keys before override
                filtered_pars = {k: v for k, v in pars.items() if k in self.pars}
                self.pars <<= filtered_pars
            else:
                self.pars <<= pars  # override existing parameters
            pars = self.pars

            self.verbose = pars["verbose"] if "verbose" in pars else 1

            # Set the random seed
            if pars.seed is None:
                now = datetime.now()  # noqa: DTZ005
                pars.seed = now.microsecond ^ int(now.timestamp())
                if self.verbose >= 1:
                    sc.printgreen(f"No seed provided. Using random seed of {pars.seed}.")
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

        # Setup early stopping option - controlled in DiseaseState_ABM component
        self.should_stop = False

    def __init__(self, pars: PropertySet = None, verbose=1):
        """
        This is the regular constructor. It is not called when initializing from file.
        add_scalar_property calls should only be here, not in common_init, or init_from_file.
        Same goes for assignments to values in sim.people.xxx
        """
        self.perf_stats = TimingStats()
        with self.perf_stats.start(self.__class__.__name__ + ".__init__()"):
            self.common_init(pars, verbose)
            pars = self.pars

            # --- Initialize the LaserFrame ---
            # Start by setting the number of initial agents per node
            pars.init_pop = np.atleast_1d(pars.init_pop).astype(int)  # Ensure pars.init_pop is an array
            if pars.init_sus_by_age is not None:
                # If init_sus_by_age is provided, we'll only allocate the laserframe for the susceptible population (saves on memory)
                init_sus = (
                    pars.init_sus_by_age.groupby("node_id")["n_susceptible"].sum().astype(int)
                )  # Sum susceptibles by node (i.e., sum across age bins)
                pars.init_sus = np.atleast_1d(init_sus)
                total_pop = init_sus.sum()
            else:
                total_pop = np.sum(pars.init_pop)
            # Next, calculate capacity aka the total number of people expected over the course of the simulation
            # +100 is another fudge factor to account for calc_cap and other math tending to be too low.
            ADAPTIVE_SAFETY_MARGIN_NUMERATOR = 4
            expected_births = 0
            if (pars.cbr is not None) and (len(pars.cbr) == 1):
                expected_births = calc_capacity(pars.init_pop.sum(), pars.dur + 100, pars.cbr[0]) - pars.init_pop.sum()
            elif (pars.cbr is not None) and (len(pars.cbr) > 1):
                expected_births = calc_capacity(pars.init_pop.sum(), pars.dur + 100, np.mean(pars.cbr)) - pars.init_pop.sum()
            if expected_births > 0:
                fudge_factor = 1 + ADAPTIVE_SAFETY_MARGIN_NUMERATOR / np.sqrt(expected_births)
            else:
                fudge_factor = 1
            # Note that capacity = total_pop if there are no births.
            capacity = int(fudge_factor * (total_pop + expected_births))
            # Finally, initialize the LaserFrame
            self.people = LaserFrame(capacity=capacity, initial_count=int(total_pop))
            logger.info(f"count={self.people.count}, capacity={capacity}")

            # --- Initializes any essential agent properties that are required across multiple components ---
            # Initialize disease_state, ipv_protected, paralyzed, and potentially_paralyzed here since they're required for most other components
            self.people.add_scalar_property("disease_state", dtype=np.int8, default=-1)  # -1=Dead/inactive, 0=S, 1=E, 2=I, 3=R
            self.people.disease_state[: self.people.count] = 0  # Set initial population as susceptible
            self.people.add_scalar_property(
                "potentially_paralyzed", dtype=np.int8, default=-1
            )  # Set default to -1 as a way to check if they've been potentially paralyzed
            self.people.add_scalar_property("paralyzed", dtype=np.int8, default=0)
            self.people.add_scalar_property("ipv_protected", dtype=np.int8, default=0)
            self.results = LaserFrame(capacity=1)
            self.people.add_scalar_property("strain", dtype=np.int8, default=0)  # 0 = VDPV, 1 = Sabin, 2 = nOPV2
            # Setup the chronically missed population for vaccination: 0 = missed/inaccessible to vx, 1 = accessible for vaccination
            self.people.add_scalar_property("chronically_missed", dtype=np.uint8, default=0)
            missed_frac = pars.missed_frac
            n = self.people.count
            n_missed = int(missed_frac * n)
            missed_ids = np.random.choice(n, size=n_missed, replace=False)
            self.people.chronically_missed[missed_ids] = 1  # Set the missed population to 1 (missed/inaccessible)

            # --- Initialize nodes & node IDs ---
            self.people.add_scalar_property("node_id", dtype=np.int16, default=-1)
            self.nodes = np.arange(len(pars.init_pop))
            if pars.init_sus_by_age is not None:
                pop_by_node = pars.init_sus
            else:
                pop_by_node = pars.init_pop
            # Check that pop_by_node is positive & non-zero
            if np.any(pop_by_node <= 0) or np.any(np.isnan(pop_by_node)):
                raise ValueError("pop_by_node must be positive & non-nan")
            node_ids = np.concatenate([np.full(count, i) for i, count in enumerate(pop_by_node)])
            self.people.node_id[0 : np.sum(pop_by_node)] = node_ids  # Assign node IDs to initial people

            # Components
            self._components = []

    @classmethod
    def init_from_file(cls, people: LaserFrame, pars: PropertySet = None):
        # initialize model
        model = cls.__new__(cls)
        model.common_init(pars, verbose=2)  # TBD: add nasty verbose param

        model.people = people
        print(f"Capacity of reload = {model.people.capacity}.")

        # Setup node list
        model.nodes = np.unique(model.people.node_id[: model.people.count])

        # Results holder
        model.results = LaserFrame(capacity=1)

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
            with self.perf_stats.start(cls.__name__ + ".__init__()"):
                self.instances.append(cls(self))

        if self.verbose >= 2:
            print(f"Initialized components: {self.instances}")

    def run(self):
        if self.verbose >= 1:
            sc.printcyan("Initialization complete. Running simulation...")
        step_stats = TimingStats()
        with alive_bar(self.nt, title="Simulation progress:", disable=self.verbose < 1) as bar:
            for tick in range(self.nt):
                with step_stats.start(f"t={tick}"):
                    if tick == 0:
                        # Just record the initial state on t=0 & don't run any components
                        self.log_results(tick)
                        self.t += 1
                    else:
                        for component in self.instances:
                            with self.perf_stats.start(component.__class__.__name__ + ".step()"):
                                component.step()

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

        self.perf_stats.log(logger)
        step_stats.log(logger)

        return

    def log_results(self, t):
        for component in self.instances:
            with self.perf_stats.start(component.__class__.__name__ + ".log()"):
                component.log(t)

        return

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
        plot_node_pop(self.results, self.nodes, save=save, results_path=results_path)

        if self.perf_stats and self.perf_stats.stats:
            # logger.debug(f"{self.instances=}")
            if self.verbose >= 2:
                print(f"{self.instances=}")
            plt.figure(figsize=(12, 12))

        total_time = sum(self.perf_stats.stats.values())
        threshold = 1  # 1%
        # Set label to None if the percentage is less than threshold
        labels = list(
            map(
                lambda k, v: k if (v / total_time) > (threshold / 100) else None,
                self.perf_stats.stats.keys(),
                self.perf_stats.stats.values(),
            )
        )

        plt.pie(
            x=self.perf_stats.stats.values(),
            labels=labels,
            autopct=lambda pct: f"{pct:1.1f}%" if pct > threshold else "",  # show percentage only if greater than threshold
            pctdistance=0.85,  # distance of percentage from center (0.6 is the default)
            labeldistance=1.1,  # distance of labels from center (1.1 is the default)
            radius=0.9,  # radius of the pie chart (1.0 is the default)
            # rotatelabels=True,  # rotate labels (False is the default)
        )

        plt.title(f"Time Spent in Each Component ({sum(self.perf_stats.stats.values()) / 1e9:.2f} seconds)")
        if save:
            plt.savefig(results_path / "perfpie.png")
        if not save:
            plt.show()

        return


def disease_state_step(
    node_id,
    n_nodes,
    disease_state,
    active_count,
    exposure_timer,
    infection_timer,
    potentially_paralyzed,
    paralyzed,
    ipv_protected,
    paralysis_timer,
    p_paralysis,
    new_potential,
    new_paralyzed,
):
    """
    Python wrapper:
      - allocates thread-local buffers once
      - calls the jitted kernel to fill them
      - reduces across threads into the provided output arrays in-place
    """
    n_threads = nb.get_num_threads()  # OK to call here (outside jitted code)
    local_new_potential = np.zeros((n_threads, n_nodes), dtype=np.int32)
    local_new_paralyzed = np.zeros((n_threads, n_nodes), dtype=np.int32)

    # kernel mutates state arrays in-place + accumulates thread-local counts
    disease_state_step_kernel(
        node_id,
        n_nodes,
        disease_state,
        active_count,
        exposure_timer,
        infection_timer,
        potentially_paralyzed,
        paralyzed,
        ipv_protected,
        paralysis_timer,
        p_paralysis,
        local_new_potential,
        local_new_paralyzed,
    )

    # Parallel-safe reduction (NumPy on host is fine)
    new_potential[:] += local_new_potential.sum(axis=0)
    new_paralyzed[:] += local_new_paralyzed.sum(axis=0)


@nb.njit(parallel=True, nogil=True, cache=True)  # no explicit signature -> no import-time compile
def disease_state_step_kernel(
    node_id,
    n_nodes,
    disease_state,
    active_count,
    exposure_timer,
    infection_timer,
    potentially_paralyzed,
    paralyzed,
    ipv_protected,
    paralysis_timer,
    p_paralysis,
    local_new_potential,
    local_new_paralyzed,
):
    # NOTE: no large allocations, no nb.get_num_threads() here
    for i in nb.prange(active_count):
        tid = nb.get_thread_id()
        nid = node_id[i]

        was_potentially_paralyzed = False
        was_paralyzed = False

        # ---- Exposed to Infected Transition ----
        if disease_state[i] == 1:  # Exposed
            if exposure_timer[i] <= 0:
                disease_state[i] = 2  # Become infected
            exposure_timer[i] -= 1

        # ---- Infected to Recovered Transition ----
        if disease_state[i] == 2:  # Infected
            if infection_timer[i] <= 0:
                disease_state[i] = 3  # Become recovered
            infection_timer[i] -= 1

        # ---- Paralysis ----
        if disease_state[i] in (1, 2, 3):
            # NOTE: Currently we don't have strain tracking, so I had to set potentially_paralyzed to 0 in SIA_ABM & RI_ABM, otherwise those interventions would cause potential paralysis cases.
            # TODO: revise when we have strain stracking
            # TODO: remove the potential_paralysis attributes from RI & SIAs after we have strain tracking
            # Any time after exposure, but not yet potentially paralyzed
            if potentially_paralyzed[i] == -1:  # after exposure, but not yet flagged
                if paralysis_timer[i] <= 0:
                    if ipv_protected[i] == 0:
                        potentially_paralyzed[i] = 1
                        was_potentially_paralyzed = True
                        # Numba supports np.random.random() in nopython mode
                        if np.random.random() < p_paralysis:
                            paralyzed[i] = 1
                            was_paralyzed = True
                    else:
                        # Explicitly set to 0 if protected
                        potentially_paralyzed[i] = 0
                paralysis_timer[i] -= 1

        if was_potentially_paralyzed:
            local_new_potential[tid, nid] += 1
        if was_paralyzed:
            local_new_paralyzed[tid, nid] += 1


@nb.njit(parallel=True, cache=True)
def set_recovered_by_dob(num_people, dob, disease_state, threshold_dob):
    for i in nb.prange(num_people):
        if dob[i] < threshold_dob:
            disease_state[i] = 3  # Set as recovered

    return


@nb.njit([(nb.int32, nb.int8[:], nb.boolean[:]), (nb.int64, nb.int32[:], nb.boolean[:])], parallel=True, cache=True)
def set_filter_mask(num_people, disease_state, filter_mask):
    for i in nb.prange(num_people):
        select = (disease_state[i] >= 0) and (disease_state[i] < 3)
        filter_mask[i] = select

    return


@nb.njit(parallel=True, cache=True)
def get_eligible_by_node(num_nodes, num_people, eligible, node_ids):
    tls_counts = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)  # Adjust size as needed

    for i in nb.prange(num_people):
        if eligible[i]:
            tls_counts[nb.get_thread_id(), node_ids[i]] += 1

    return tls_counts.sum(axis=0)  # Sum across threads to get the final counts


@nb.njit(parallel=True, cache=True)
def set_recovered_by_probability(num_people, eligible, recovery_probs, node_ids, disease_state):
    for i in nb.prange(num_people):
        if eligible[i]:
            recovered = np.random.binomial(1, recovery_probs[node_ids[i]])
            if recovered > 0:
                disease_state[i] = 3

    return


@nb.njit(parallel=True, cache=True)
def set_eligible_mask(num_people, alive_mask, age, age_min, age_max, eligible_mask):
    for i in nb.prange(num_people):
        eligible_mask[i] = alive_mask[i] and (age[i] >= age_min) and (age[i] < age_max)

    return


class DiseaseState_ABM:
    @classmethod
    def init_from_file(cls, sim):
        # Alternate constructor: skip initialization logic
        self = cls.__new__(cls)  # bypass __init__
        self._common_init(sim)
        # Only set up results arrays if needed
        self._initialize_results_arrays()

        cap = getattr(self.people, "true_capacity", self.people.capacity)
        count = self.people.count
        # We need to set daily_infectivity and acq_risk_multiplier for count:capacity
        populate_heterogeneous_values(count, cap, self.people.acq_risk_multiplier, self.people.daily_infectivity, self.pars)
        sim.people.exposure_timer[count:cap] = self.pars.dur_exp(cap - count)
        sim.people.infection_timer[count:cap] = self.pars.dur_inf(cap - count)
        sim.people.paralysis_timer[count:cap] = self.pars.t_to_paralysis(cap - count)
        sim.people.potentially_paralyzed[count:cap] = -1
        sim.people.paralyzed[count:cap] = -1
        sim.people.ipv_protected[count:cap] = -1
        return self

    def _common_init(self, sim):
        self.sim = sim
        self.people = sim.people
        self.pars = sim.pars
        self.nodes = sim.nodes
        self.results = sim.results
        self.verbose = self.pars["verbose"] if "verbose" in self.pars else 1

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
        self.results.add_array_property("E_by_strain", shape=(self.sim.nt, len(self.nodes), len(self.sim.pars.strain_ids)), dtype=np.int32)
        self.results.add_array_property("I_by_strain", shape=(self.sim.nt, len(self.nodes), len(self.sim.pars.strain_ids)), dtype=np.int32)
        self.results.add_array_property("potentially_paralyzed", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.add_array_property("paralyzed", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.add_array_property("new_potentially_paralyzed", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.add_array_property("new_paralyzed", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.add_array_property("pop", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
        self.results.pop[0] = self.sim.pars.init_pop

    def __init__(self, sim):
        self._common_init(sim)
        self._initialize_results_arrays()
        self.verbose = sim.pars["verbose"] if "verbose" in sim.pars else 1

        # Initialize all agents with an exposure_timer, infection_timer, and paralysis_timer
        sim.people.add_scalar_property("exposure_timer", dtype=np.uint8, default=0)
        sim.people.exposure_timer[:] = self.pars.dur_exp(self.people.capacity)
        sim.people.add_scalar_property("infection_timer", dtype=np.uint8, default=0)
        sim.people.infection_timer[:] = self.pars.dur_inf(self.people.capacity)
        sim.people.add_scalar_property("paralysis_timer", dtype=np.uint8, default=0)
        sim.people.paralysis_timer[:] = self.pars.t_to_paralysis(self.people.capacity)

        pars = self.pars

        # logger.debug(f"Before immune initialization, we have {sim.people.count} active agents.")
        if self.verbose >= 2:
            print(f"Before immune initialization, we have {sim.people.count} active agents.")

        # -- Initialize immunity --
        if pars.init_sus_by_age is None:
            # Normalize init_immun into per-node immunity fractions
            if isinstance(pars.init_immun, float):
                immun_fracs = np.full_like(pars.init_pop, fill_value=pars.init_immun, dtype=np.float32)
            elif isinstance(pars.init_immun, list):
                immun_fracs = np.asarray(pars.init_immun, dtype=np.float32)
            elif isinstance(pars.init_immun, np.ndarray):
                immun_fracs = pars.init_immun
                assert immun_fracs.shape == pars.init_pop.shape, "init_immun must match init_pop shape"
            else:
                raise ValueError(f"Unsupported init_immun type: {type(pars.init_immun)}")

            # Loop over nodes to apply immunity
            for nid, (immun_frac, node_pop) in enumerate(zip(immun_fracs, pars.init_pop, strict=False)):
                assert 0.0 <= immun_frac <= 1.0, f"Invalid immunity fraction: {immun_frac} for node {nid}"
                num_recovered = int(immun_frac * node_pop)
                if num_recovered > 0:
                    node_indices = np.where(sim.people.node_id == nid)[0]
                    recovered_indices = np.random.choice(node_indices, size=num_recovered, replace=False)
                    sim.people.disease_state[recovered_indices] = 3
                    # Don't add to results here, since anyone with a disease_state will get counted during logging

        elif pars.init_sus_by_age is not None:
            # Initialize nodes with only susceptible population. We'll handle mortality in the VitalDynamics_ABM component
            sim.people.disease_state[:] = 0
            init_recovered = pars.init_pop - pars.init_sus
            sim.results.R[:, :] += init_recovered  # Tally here since they don't have a disease_state and won't get counted during logging

            # Account for IPV protection which prevents paralysis but not transmission
            # The subset of susceptible people who are IPV-protected is stored in the init_sus_by_age table as n_ipv_protected

            # Initialize all agents as not IPV protected
            sim.people.ipv_protected[: sim.people.count] = 0

            # Filter for age groups that actually have IPV protection (n_ipv_protected > 0)
            ipv_data = pars.init_sus_by_age[pars.init_sus_by_age["n_ipv_protected"] > 0]

            if len(ipv_data) > 0:
                # Group by node for more efficient processing
                for node in ipv_data["node_id"].unique():
                    # Get all agents in this node (filter by node once per node)
                    node_mask = sim.people.node_id[: sim.people.count] == node
                    node_agent_indices = np.where(node_mask)[0]

                    if len(node_agent_indices) == 0:
                        continue

                    # Get ages in years for all agents in this node
                    agent_ages_days = -sim.people.date_of_birth[node_agent_indices]
                    agent_ages_years = agent_ages_days / 365.0

                    # Get IPV data for this specific node
                    node_ipv_data = ipv_data[ipv_data["node_id"] == node]

                    # Apply IPV protection for each age group in this node
                    for _, row in node_ipv_data.iterrows():
                        n_ipv_protected = int(row["n_ipv_protected"])
                        age_min_yr = row["age_min_yr"]
                        age_max_yr = row["age_max_yr"]

                        # Find agents in this age group
                        age_mask = (agent_ages_years >= age_min_yr) & (agent_ages_years < age_max_yr)
                        eligible_agents = node_agent_indices[age_mask]

                        if len(eligible_agents) > 0:
                            # Sample agents to protect with IPV (without replacement)
                            n_to_protect = min(n_ipv_protected, len(eligible_agents))
                            if n_to_protect > 0:
                                protected_agents = np.random.choice(eligible_agents, size=n_to_protect, replace=False)
                                sim.people.ipv_protected[protected_agents] = 1

            # Account for mortality in the immune/recovered population which is pre-calculated in the VitalDynamics_ABM component
            if hasattr(self.results, "deaths"):
                deaths = self.results.deaths
                cum_deaths = np.cumsum(deaths, axis=0)
                self.results.R[:, :] -= cum_deaths

        else:
            raise ValueError(f"Unsupported init_immun type: {type(pars.init_immun)}")

        # --- Seed infections ---
        # This (potentially) overwrites immunity, e.g., if an individual is drawn as both immune (during immunity initialization above) and infected (below), they will be infected)
        # The specification is flexible and can handle a fixed number OR fraction
        infected_indices = []
        if isinstance(pars.init_prev, float):
            # Interpret as fraction of total population
            num_infected = int(sum(pars.init_pop) * pars.init_prev)
            infected_indices = np.random.choice(sum(pars.init_pop), size=num_infected, replace=False)
        elif isinstance(pars.init_prev, int):
            # Interpret as absolute number
            num_infected = min(pars.init_prev, sum(pars.init_pop))  # Don't exceed population
            infected_indices = np.random.choice(sum(pars.init_pop), size=num_infected, replace=False)
        elif isinstance(pars.init_prev, (list, np.ndarray)):
            # Ensure that the length of init_prev matches the number of nodes
            if len(pars.init_prev) != len(pars.init_pop):
                raise ValueError(f"Length mismatch: init_prev has {len(pars.init_prev)} entries, expected {len(pars.init_pop)} nodes.")
            # Interpret as per-node infection seeding
            node_ids = self.people.node_id[: self.people.count]
            disease_states = self.people.disease_state[: self.people.count]
            for node, prev in tqdm(
                enumerate(pars.init_prev), total=len(pars.init_prev), desc="Seeding infections in nodes", disable=self.verbose < 2
            ):
                if isinstance(prev, numbers.Real):
                    if 0 < prev < 1:
                        # interpret as a fraction
                        num_infected = int(pars.init_pop[node] * prev)
                    else:
                        # interpret as an integer count
                        num_infected = min(int(prev), pars.init_pop[node])
                else:
                    raise ValueError(f"Unsupported value in init_prev list at node {node}: {prev}")

                alive_in_node = (node_ids == node) & (disease_states >= 0)
                alive_in_node_indices = np.where(alive_in_node)[0]
                num_infections_to_draw = min(num_infected, len(alive_in_node_indices))
                infected_indices_node = np.random.choice(alive_in_node_indices, size=num_infections_to_draw, replace=False)
                infected_indices.extend(infected_indices_node)
        else:
            raise ValueError(f"Unsupported init_prev type: {type(pars.init_prev)}")
        # Create the infections
        num_infected = len(infected_indices)
        sim.people.disease_state[infected_indices] = 2

    def step(self):
        t = self.sim.t
        n_nodes = len(self.nodes)

        # Progress disease state & check for paralysis
        new_potential = np.zeros(n_nodes, dtype=np.int32)
        new_paralyzed = np.zeros(n_nodes, dtype=np.int32)
        disease_state_step(
            node_id=self.people.node_id,
            n_nodes=n_nodes,
            disease_state=self.people.disease_state,
            active_count=self.people.count,
            exposure_timer=self.people.exposure_timer,
            infection_timer=self.people.infection_timer,
            potentially_paralyzed=self.people.potentially_paralyzed,
            paralyzed=self.people.paralyzed,
            ipv_protected=self.people.ipv_protected,
            paralysis_timer=self.people.paralysis_timer,
            p_paralysis=nb.float32(self.pars.p_paralysis),
            new_potential=new_potential,
            new_paralyzed=new_paralyzed,
        )
        self.results.new_potentially_paralyzed[t, :] = new_potential
        self.results.new_paralyzed[t, :] = new_paralyzed

        # Debugging
        if self.verbose >= 3:
            logger.info(f"DISEASE STATE STEP at time: {self.sim.t}")

            # Who is exposed?
            exposed = self.people.disease_state == 1
            logger.info(f"Exposed IDs: {np.where(exposed)[0]}")
            logger.info(f"Exposure timers: {self.people.exposure_timer[np.where(exposed)[0]]}")
            unique_exposure_timers, counts_exposure_timers = np.unique(self.people.exposure_timer, return_counts=True)
            logger.info(f"Unique exposure timers: {dict(zip(unique_exposure_timers, counts_exposure_timers, strict=False))}")

            # Who is infected?
            infected = self.people.disease_state == 2
            logger.info(f"Infected IDs: {np.where(infected)[0]}")
            logger.info(f"Infection timers: {self.people.infection_timer[np.where(infected)[0]]}")
            unique_infection_timers, counts_infection_timers = np.unique(self.people.infection_timer, return_counts=True)
            logger.info(f"Unique infection timers: {dict(zip(unique_infection_timers, counts_infection_timers, strict=False))}")

        # --- Seed infections from seed_schedule ---
        if t in self.seed_schedule:
            for node_id, value in self.seed_schedule[t]:
                node_mask = (self.people.node_id[: self.people.count] == node_id) & (self.people.disease_state[: self.people.count] >= 0)
                candidates = np.where(node_mask)[0]
                # Handle prevalence (float) or fixed count (int)
                if isinstance(value, float):
                    n_seed = int(len(candidates) * value)
                elif isinstance(value, int):
                    n_seed = min(value, len(candidates))  # Avoid oversampling
                else:
                    raise ValueError(f"Unsupported seed value type: {type(value)}")
                if n_seed > 0:
                    selected = np.random.choice(candidates, size=n_seed, replace=False)
                    self.people.disease_state[selected] = 2  # Set to infectious regardless of current state
                    # If people were previously infected, we'll need to give them an infection timer again
                    inf_timer = self.people.infection_timer[selected]
                    inds_zero_timers = selected[np.where(inf_timer <= 0)]
                    self.sim.people.infection_timer[inds_zero_timers] = self.pars.dur_inf(len(inds_zero_timers))
                    if self.verbose >= 1:
                        print(f"[DiseaseState_ABM] t={t}: Seeded {n_seed} infections in node {node_id}")
                        # daily_infectivity = self.people.daily_infectivity[selected]
                        # inf_timer = self.people.infection_timer[selected]
                        # len(selected)
                        # daily_infectivity.min()
                        # daily_infectivity.mean()
                        # inf_timer.min()
                        # inf_timer.mean()

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
        plot_total_seir_counts(self.results, save=save, results_path=results_path)
        plot_infected_by_node(self.results, self.nodes, save=save, results_path=results_path)
        plot_infected_by_node_strain(self.results, self.pars, save=save, results_path=results_path)
        plot_infected_dot_map(self.results, self.pars, self.nodes, save=save, results_path=results_path)
        plot_cum_new_exposed_paralyzed(self.results, save=save, results_path=results_path)
        plot_new_exposed_by_strain(self.results, self.pars, save=save, results_path=results_path)
        if self.pars.shp is not None:
            plot_infected_choropleth(self.results, self.pars, save=save, results_path=results_path)
            plot_infected_choropleth_by_strain(self.results, self.pars, save=save, results_path=results_path)
            plot_animated_strain_map(self.results, self.pars, self.sim.datevec, save=save, results_path=results_path)
            plot_animated_strain_map_panels(self.results, self.pars, self.sim.datevec, save=save, results_path=results_path)


# This utility function is called from two different places; doesn't need to be member of
# a class
def populate_heterogeneous_values(start, end, acq_risk_out, infectivity_out, pars):
    """
    Populates acq_risk_out and infectivity_out arrays in-place using the specified
    correlation structure and parameter set.

    Parameters
    ----------
    start : int
        Start index (inclusive).
    end : int
        End index (exclusive).
    acq_risk_out : np.ndarray
        Pre-allocated array to store acquisition risk multipliers.
    infectivity_out : np.ndarray
        Pre-allocated array to store daily infectivity values.
    pars : PropertySet
        LASER parameter set with keys:
            - risk_mult_var
            - r0
            - dur_inf
            - corr_risk_inf
    """

    mean_ln = 1
    var_ln = pars.risk_mult_var
    mu_ln = np.log(mean_ln**2 / np.sqrt(var_ln + mean_ln**2))
    sigma_ln = np.sqrt(np.log(var_ln / mean_ln**2 + 1))
    mean_gamma = pars.r0 / np.mean(pars.dur_inf(1000))
    shape_gamma = 1
    scale_gamma = max(mean_gamma / shape_gamma, 1e-10)

    rho = pars.corr_risk_inf
    cov_matrix = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov_matrix)

    logger.info("FIXME: This chunk of code to initialize acq_risk_out and infectivity_out is known to be slow right now.")
    BATCH_SIZE = 1_000_000
    for batch_start in range(start, end, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, end)
        b_n = batch_end - batch_start

        z = np.random.normal(size=(b_n, 2))
        z_corr = z @ L.T

        if pars.individual_heterogeneity:
            acq_risk_out[batch_start:batch_end] = np.exp(mu_ln + sigma_ln * z_corr[:, 0])
            infectivity_out[batch_start:batch_end] = stats.gamma.ppf(stats.norm.cdf(z_corr[:, 1]), a=shape_gamma, scale=scale_gamma)
        else:
            acq_risk_out[batch_start:batch_end] = 1.0
            infectivity_out[batch_start:batch_end] = mean_gamma


def count_SEIRP(node_id, disease_state, strain, potentially_paralyzed, paralyzed, n_nodes: int, n_strains: int, n_people: int):
    """
    Python wrapper:
      - allocates thread-local buffers once
      - calls the jitted kernel to fill them
      - reduces across threads & strains to produce outputs
    """
    # Thread-local buffers, allocated outside jitted code
    n_threads = nb.get_num_threads()
    S = np.zeros((n_threads, n_nodes), dtype=np.int32)
    R = np.zeros((n_threads, n_nodes), dtype=np.int32)
    POTP = np.zeros((n_threads, n_nodes), dtype=np.int32)
    P = np.zeros((n_threads, n_nodes), dtype=np.int32)
    Ebs = np.zeros((n_threads, n_nodes, n_strains), dtype=np.int32)  # E_by_strain
    Ibs = np.zeros((n_threads, n_nodes, n_strains), dtype=np.int32)  # I_by_strain

    # Fill in thread-local buffers
    count_SEIRP_kernel(node_id, disease_state, strain, potentially_paralyzed, paralyzed, n_people, S, R, POTP, P, Ebs, Ibs)

    # Reductions (can be in Python/Numpy — no need to jittify)
    S_final = S.sum(axis=0)
    R_final = R.sum(axis=0)
    POTP_final = POTP.sum(axis=0)
    P_final = P.sum(axis=0)

    E_by_strain_final = Ebs.sum(axis=0)
    I_by_strain_final = Ibs.sum(axis=0)
    E_final = E_by_strain_final.sum(axis=1)
    I_final = I_by_strain_final.sum(axis=1)

    return (S_final, E_final, I_final, R_final, E_by_strain_final, I_by_strain_final, POTP_final, P_final)


@nb.njit(parallel=True, nogil=True, cache=True)  # <-- no explicit signature here
def count_SEIRP_kernel(node_id, disease_state, strain, potentially_paralyzed, paralyzed, n_people, S, R, POTP, P, Ebs, Ibs):
    """
    Jitted kernel:
      - no big dynamic allocations inside
      - only writes into provided thread-local buffers
      - safe to cache
    """
    for i in nb.prange(n_people):
        if disease_state[i] >= 0:
            nd = node_id[i]
            ds = disease_state[i]
            st = strain[i]
            tid = nb.get_thread_id()

            if ds == 0:  # Susceptible
                S[tid, nd] += 1
            elif ds == 1:  # Exposed
                Ebs[tid, nd, st] += 1
            elif ds == 2:  # Infected
                Ibs[tid, nd, st] += 1
            elif ds == 3:  # Recovered
                R[tid, nd] += 1

            if potentially_paralyzed[i] == 1:
                POTP[tid, nd] += 1
            if paralyzed[i] == 1:
                P[tid, nd] += 1


def tx_step_prep(
    num_nodes: int,
    num_people: int,
    n_strains: int,
    strains,
    strain_r0_scalars,
    disease_states,
    node_ids,
    daily_infectivity,  # per-agent infectivity (heterogeneous)
    risks,  # per-agent susceptibility (heterogeneous)
):
    """
    Python wrapper:
      - allocates thread-local buffers once
      - calls the jitted kernel to fill them
      - reduces across threads and returns arrays matching the original API
    """
    n_threads = nb.get_num_threads()

    # Thread-local buffers allocated outside jitted code (cache-friendly)
    tl_beta_by_node_strain = np.zeros((n_threads, num_nodes, n_strains), dtype=np.float32)
    tl_exposure_by_node = np.zeros((n_threads, num_nodes), dtype=np.float32)
    tl_sus_by_node = np.zeros((n_threads, num_nodes), dtype=np.int32)

    # Fill thread-local buffers in parallel
    tx_step_prep_kernel(
        num_people,
        n_strains,
        strains,
        strain_r0_scalars,
        disease_states,
        node_ids,
        daily_infectivity,
        risks,
        tl_beta_by_node_strain,
        tl_exposure_by_node,
        tl_sus_by_node,
    )

    # Reductions across threads (NumPy host reductions are fine)
    exposure_by_node = tl_exposure_by_node.sum(axis=0)
    sus_by_node = tl_sus_by_node.sum(axis=0)
    beta_by_node_strain_pre = tl_beta_by_node_strain.sum(axis=0)
    beta_by_node_strain = beta_by_node_strain_pre.copy()  # preserve your original copy semantics

    return beta_by_node_strain, exposure_by_node, sus_by_node


@nb.njit(parallel=True, nogil=True, cache=True)  # no large allocations inside
def tx_step_prep_kernel(
    num_people,
    n_strains,
    strains,
    strain_r0_scalars,
    disease_states,
    node_ids,
    daily_infectivity,
    risks,
    tl_beta_by_node_strain,
    tl_exposure_by_node,
    tl_sus_by_node,
):
    # NOTE: no nb.get_num_threads() here; buffers provided by the wrapper
    for i in nb.prange(num_people):
        state = disease_states[i]
        tid = nb.get_thread_id()
        nid = node_ids[i]

        if state == 0:  # susceptible
            tl_exposure_by_node[tid, nid] += risks[i]
            tl_sus_by_node[tid, nid] += 1

        elif state == 2:  # infected
            s = strains[i]
            # (Assumes 0 <= s < n_strains; if not, validate upstream)
            tl_beta_by_node_strain[tid, nid, s] += daily_infectivity[i] * strain_r0_scalars[s]


@nb.njit(parallel=True, cache=True)
def tx_infect_nb(
    num_nodes,
    num_people,
    num_strains,
    sus_by_node,
    node_ids,
    strain,
    disease_state,
    sus_indices_storage,
    sus_probs_storage,
    risks,
    prob_exp_by_node_strain,
    n_exposures_to_create_by_node_strain,  # shape: (num_nodes, num_strains)
):
    """
    Parallelizes over nodes, computing a CDF for each node's susceptible population.
    Selects unique indices via weighted sampling, and allocates them to strains based on relative FOI.
    """

    # Susceptible agents in a node are _not_ necessarily contiguous in the array because
    #   a) disease dynamics (some E, I, and R) and
    #   b) vital dynamics (some dead/inactive)
    # So we need to create a mapping, in contiguous memory, of all the susceptible agents for a (each) node
    # We also need the heterogeneous susceptibility (risks) for each susceptible agent

    # If there are, e.g., [50, 20, 18, 91] susceptibles in each node, we "reserve" that many slots in sus_indices
    # and sus_probs by setting the offsets for each node to [0, 50, 70, 88]. I.e., the indices of susceptible agents
    # for node 0 start at sus_indices[0], node 1 at sus_indices[50], etc.
    # Then we track how many slots we have used with next_index which we increment as we fill in the slots.

    # At the end, the first 50 entries of sus_indices will have the indices of the susceptible agents in node 0,
    # the next 20 will have the indices of the susceptible agents in node 1, etc.
    # The values in sus_probs will be the susceptibility of each of those agents.

    # If later we want to access or process the susceptible agents in node 2 (and we do want), for example,
    #  we can do so by using
    #   sus_indices[offsets[2]:offsets[2] + sus_by_node[2]] (sus_indices[70:88]) and
    #   sus_probs[offsets[2]:offsets[2] + sus_by_node[2]] (sus_probs[70:88]).

    offsets = np.zeros(num_nodes, dtype=np.int32)
    offsets[1:] = sus_by_node[:-1].cumsum()
    next_index = np.empty(num_nodes, dtype=np.int32)
    next_index[:] = offsets
    total_exposures_by_node = n_exposures_to_create_by_node_strain.sum(axis=1)  # shape: [num_nodes]
    for i in range(num_people):
        nid = node_ids[i]
        if (total_exposures_by_node[nid] > 0) and (disease_state[i] == 0):
            idx = next_index[nid]
            sus_indices_storage[idx] = i
            sus_probs_storage[idx] = risks[i]  # base susceptibility, will be scaled by strain FOI later
            next_index[nid] = idx + 1

    already_exposed = np.zeros(num_people, dtype=nb.boolean)  # global per-person flag to prevent double infection
    n_new_exposures = np.zeros((num_nodes, num_strains), dtype=np.int32)

    for node in nb.prange(num_nodes):
        # Calculate total exposures needed for this node across all strains
        total_exposures_needed = n_exposures_to_create_by_node_strain[node, :].sum()
        if total_exposures_needed <= 0:
            continue

        # Get and check count of susceptible agents in _this_ node
        sus_count = sus_by_node[node]
        if sus_count == 0:
            continue

        offset = offsets[node]  # offset of the first susceptible agent in this node
        sus_indices = sus_indices_storage[offset : offset + sus_count]  # indices of the susceptible agents in this node
        base_probs = sus_probs_storage[offset : offset + sus_count]  # base probabilities for susceptible agents in this node

        # Calculate total FOI for this node (sum across all strains)
        total_foi = prob_exp_by_node_strain[node, :].sum()
        if total_foi <= 0:
            continue

        # Scale base probabilities by total FOI for unique selection
        scaled_probs = base_probs * total_foi

        # Choose unique indices from susceptible population using weighted sampling
        n_uniq = 0  # How many unique indices have we selected so far
        p = scaled_probs.copy()  # working copy of probabilities
        size = min(total_exposures_needed, sus_count)  # can't expose more than available susceptibles

        selected_indices = np.empty(size, dtype=np.int32)  # store selected indices

        while n_uniq < size:
            # Weighted random sampling with unique selection
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
            if cdf[-1] <= 0:  # exit early if no susceptibles remaining
                break

            # Binary search for indices, modify x to be in [0..cdf[-1])
            indices = np.searchsorted(cdf, x * cdf[-1], side="right")
            indices = np.unique(indices)  # ensure unique indices only

            # Store the selected indices
            n_selected = min(indices.size, size - n_uniq)
            selected_indices[n_uniq : n_uniq + n_selected] = indices[:n_selected]
            n_uniq += n_selected

            if n_uniq < size:  # if we haven't selected enough unique indices, we need to retry
                p[indices] = 0.0  # set the probabilities for the selected indices to zero

                # Now allocate the selected individuals to strains based on relative FOI
        if n_uniq > 0:
            # Calculate strain probabilities for allocation
            strain_probs = prob_exp_by_node_strain[node, :] / total_foi

            # Allocate each selected individual to a strain
            for i in range(n_uniq):
                person_idx = sus_indices[selected_indices[i]]
                if not already_exposed[person_idx] and disease_state[person_idx] == 0:
                    # Randomly assign strain based on relative FOI
                    r = np.random.rand()
                    cumulative_prob = 0.0
                    assigned_strain = 0
                    for s in range(num_strains):
                        cumulative_prob += strain_probs[s]
                        if r < cumulative_prob:
                            assigned_strain = s
                            break

                    # Expose the individual
                    disease_state[person_idx] = 1
                    strain[person_idx] = assigned_strain
                    already_exposed[person_idx] = True
                    n_new_exposures[node, assigned_strain] += 1

    return n_new_exposures


class Transmission_ABM:
    def __init__(self, sim):
        self.sim = sim
        self.people = sim.people
        self.nodes = np.arange(len(sim.pars.init_pop))
        self.pars = sim.pars
        self.results = sim.results
        self.verbose = sim.pars["verbose"] if "verbose" in sim.pars else 1

        # Stash the R0 scaling factor
        self.r0_scalars = np.array(self.pars.r0_scalars)

        self._initialize_people_fields()
        self._initialize_common()

        return

    @classmethod
    def init_from_file(cls, sim):
        """Alternative constructor for loading from file without resetting people."""
        instance = cls.__new__(cls)
        instance.sim = sim
        instance.people = sim.people
        instance.nodes = np.arange(len(sim.pars.init_pop))
        instance.pars = sim.pars
        instance.results = sim.results
        instance.r0_scalars = instance.pars.r0_scalars
        instance.verbose = sim.pars["verbose"] if "verbose" in sim.pars else 1

        # This is our solution for getting daily_infectivity values aligned with pars.R0 when loading existing pop
        new_r0 = sim.pars.r0
        if new_r0 != sim.pars.old_r0:
            infectivity_scalar = new_r0 / sim.pars.old_r0
            sim.people.daily_infectivity *= infectivity_scalar  # seem fast enough

        instance._initialize_common()
        return instance

    def _initialize_people_fields(self):
        """Initialize individual-level transmission properties."""

        count = getattr(self.people, "true_capacity", self.people.capacity)

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
        )  # Individual daily infectivity (e.g., number of exposures generated per day in a fully susceptible population; mean = R0/dur_inf = 14/24)

        # Step 4: Transform normal variables into target distributions
        # Set individual heterogeneity properties
        populate_heterogeneous_values(0, count, self.people.acq_risk_multiplier, self.people.daily_infectivity, self.pars)
        # z = np.random.normal(size=(n, 2)) @ L.T

    def _initialize_common(self):
        """Initialize shared network and timers."""
        # Compute the infection migration network
        self.sim.results.add_vector_property("network", length=len(self.sim.nodes), dtype=np.float32)
        self.network = self.sim.results.network
        init_pops = self.sim.pars.init_pop
        # Get the distance matrix
        logger.info("This network calc is a little slow too...")
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
            epsilon = 1
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):  # Only compute upper triangle
                    d = distance(lats[i], lons[i], lats[j], lons[j])
                    if d == 0:
                        print(f"WARNING: Distance between nodes {i} and {j} is 0. Replacing with {epsilon}")
                        d = epsilon
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d  # Mirror to lower triangle
        # Setup the network
        logger.info("END of slow network calc.")
        if self.pars.migration_method.lower() == "gravity":
            k, a, b, c = (
                self.pars.gravity_k * 10 ** (self.pars.gravity_k_exponent),
                self.pars.gravity_a,
                self.pars.gravity_b,
                self.pars.gravity_c,
            )
            self.network = gravity(init_pops, dist_matrix, k, a, b, c)
            self.network /= np.power(init_pops.sum(), c)  # Normalize
        elif self.pars.migration_method.lower() == "radiation":
            k = 10**self.pars.radiation_k_log10  # Transform from log10 space
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

        self.sim.results.add_array_property(
            "new_exposed", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32
        )  # Includes all exposures (transmission + SIA)
        self.sim.results.add_array_property(
            "new_exposed_by_strain", shape=(self.sim.nt, len(self.nodes), len(self.pars.strain_ids)), dtype=np.int32
        )  # Includes all exposures (transmission + SIA)

        self.people.add_scalar_property("sus_indices", dtype=np.int32, default=0)
        self.people.add_scalar_property("sus_probs", dtype=np.float32, default=0.0)

        self.step_stats = TimingStats()

        return

    def step(self):
        # Manual debugging of transmission
        if self.verbose >= 3:
            logger.info(f"TIMESTEP: {self.sim.t}")

        with self.step_stats.start("Part 1"):
            # 1) Stash variables for later use
            strain = self.people.strain[: self.people.count]
            strain_r0_scalars = np.array(list(self.pars.strain_r0_scalars.values()))
            disease_state = self.people.disease_state[: self.people.count]
            node_ids = self.people.node_id[: self.people.count]
            infectivity = self.people.daily_infectivity[: self.people.count]
            risk = self.people.acq_risk_multiplier[: self.people.count]
            num_nodes = len(self.nodes)
            num_people = self.sim.people.count
            n_strains = len(self.pars.strain_ids)
            node_seeding_zero_inflation = self.sim.pars.node_seeding_zero_inflation
            node_seeding_dispersion = self.sim.pars.node_seeding_dispersion

            # Manual validation
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

        with self.step_stats.start("Part 2"):
            # 2) Compute force of infection, scale by seasonality and geographic scalars, and compute the number of new exposures
            beta_seasonality = lp.get_seasonality(self.sim)
            beta_by_node_strain, exposure_by_node, sus_by_node = tx_step_prep(
                num_nodes,
                num_people,
                n_strains,
                strain,
                strain_r0_scalars,
                disease_state,
                node_ids,
                infectivity,
                risk,
            )

        with self.step_stats.start("Part 2b"):
            # Step 2: Compute the force of infection for each node accounting for immigration and emigration.
            # network is a square matrix where network[i, j] is the migration fraction from node i to node j.
            # beta_by_node_strain is a vector where beta_by_node_strain[i] is the contagion/transmission rate for node i.
            # Save a copy before distributing infectivity to know which nodes have zero local infectivity.
            beta_by_node_strain_pre = beta_by_node_strain.copy()
            # This formulation, (beta * network.T).T, returns transfer so transfer[i, j] is the contagion transferred from node i to node j
            for s in range(n_strains):
                transfer = (beta_by_node_strain[:, s] * self.network.T).T  # beta_j * network_ij
                # sum(axis=0) sums each column, i.e., _incoming_ contagion to each node
                # sum(axis=1) sums each row, i.e., _outgoing_ contagion from each node
                beta_by_node_strain[:, s] += transfer.sum(axis=0) - transfer.sum(axis=1)  # Add incoming, subtract outgoing

            # Step 3: Scale by seasonality and R0 scalars
            beta_by_node_strain = beta_by_node_strain * beta_seasonality * self.r0_scalars[:, np.newaxis]

            # Step 4: Compute the exposure rate for each node
            alive_counts = self.results.pop[self.sim.t]
            per_agent_exp_rate = beta_by_node_strain / np.maximum(
                alive_counts[:, np.newaxis], 1
            )  # convert total FOI to per-agent exposure rate
            prob_exp_by_node_strain = 1 - np.exp(-per_agent_exp_rate)  # convert rate to probability of exposure

            # Ensure probabilities are non-negative (handle any remaining numerical issues)
            prob_exp_by_node_strain = np.maximum(prob_exp_by_node_strain, 0)

            # Debug: Check for negative values and log them
            negative_mask = prob_exp_by_node_strain < 0
            if np.any(negative_mask):
                logger.warning(f"Negative prob_exp_by_node_strain detected at timestep {self.sim.t}")
                logger.warning(f"Negative values: {prob_exp_by_node_strain[negative_mask]}")
                logger.warning(f"Corresponding per_agent_exp_rate: {per_agent_exp_rate[negative_mask]}")
                logger.warning(f"Corresponding beta_by_node_strain: {beta_by_node_strain[negative_mask]}")
                logger.warning(f"Corresponding alive_counts: {alive_counts[np.any(negative_mask, axis=1)]}")

            total_exposure_prob_per_node = prob_exp_by_node_strain.sum(axis=1)  # Total prob of exposure per node (sum across all strains)
            expected_exposures_per_node = exposure_by_node * total_exposure_prob_per_node  # Expected exposures per node

            # Step 5: Compute the number of new exposures per node, by strain
            n_exposures_to_create_by_node_strain = np.zeros_like(prob_exp_by_node_strain, dtype=np.int32)  # shape: (n_nodes, n_strains)

            for n in range(num_nodes):
                exposure = expected_exposures_per_node[n]  # scalar expected total exposure in node n
                if exposure < 0:
                    # If exposure is negative, set to 0 and print a warning
                    sc.printred(f"Warning: exposure_by_node[{n}] is negative: {exposure_by_node[n]}. Setting to 0.")
                    sc.printred(f"beta_by_node_strain_pre[{n}] is {beta_by_node_strain_pre[n]}")
                    sc.printred(f"alive_counts[{n}] is {alive_counts[n]}")
                    sc.printred(f"per_agent_exp_rate[{n}] is {per_agent_exp_rate[n]}")
                    exposure = 0

                if exposure == 0:
                    total_exposures_to_create = 0

                elif np.sum(beta_by_node_strain_pre[n]) == 0:
                    # If node has no local transmission, apply zero-inflated NB
                    if node_seeding_zero_inflation >= 1.0:
                        total_exposures_to_create = 0
                    else:
                        desired_mean = exposure / (1 - node_seeding_zero_inflation)
                        r = max(1, int(np.round(node_seeding_dispersion)))
                        p = r / (r + desired_mean)

                        if np.random.rand() < node_seeding_zero_inflation:
                            total_exposures_to_create = 0
                        else:
                            total_exposures_to_create = np.random.negative_binomial(r, p)

                else:
                    # If node has local transmission, sample from Poisson
                    total_exposures_to_create = np.random.poisson(exposure)

                # Strain allocation
                if total_exposures_to_create > 0:
                    total_prob = total_exposure_prob_per_node[n]
                    if total_prob > 0:
                        strain_probs = prob_exp_by_node_strain[n] / total_prob
                    else:
                        strain_probs = np.zeros(n_strains)

                    n_exposures_to_create_by_node_strain[n] = np.random.multinomial(total_exposures_to_create, strain_probs)

            # Manual validation
            if self.verbose >= 3:
                logger.info(f"beta_seasonality: {fmt(beta_seasonality, 2)}")
                logger.info(f"R0 scalars: {fmt(self.r0_scalars, 2)}")
                logger.info(f"beta: {fmt(beta_by_node_strain, 2)}")
                logger.info(f"Total beta: {fmt(beta_by_node_strain.sum(), 2)}")
                logger.info(f"Alive counts: {fmt(alive_counts, 2)}")
                logger.info(f"Base prob infection: {fmt(prob_exp_by_node_strain, 2)}")
                logger.info(f"Exp inf (sans acq risk): {fmt(num_susceptibles * prob_exp_by_node_strain, 2)}")
                disease_state_pre_infect = disease_state.copy()  # Copy before infection

        with self.step_stats.start("Part 3"):
            # 3) Distribute new exposures
            new_exposed_by_node_strain = tx_infect_nb(
                num_nodes,
                num_people,
                n_strains,
                sus_by_node,
                node_ids,
                strain,
                disease_state,
                self.people.sus_indices,
                self.people.sus_probs,
                risk,
                prob_exp_by_node_strain,
                n_exposures_to_create_by_node_strain,
            )
            self.sim.results.new_exposed[self.sim.t, :] += new_exposed_by_node_strain.sum(
                axis=1
            )  # Add here b/c we're also counting SIA exposures
            self.sim.results.new_exposed_by_strain[self.sim.t, :, :] += (
                new_exposed_by_node_strain  # Add here b/c we're also counting SIA exposures
            )

            # Manual validation
            if self.verbose >= 3:
                logger.info(f"exposure_by_node: {fmt(exposure_by_node, 2)}")
                logger.info(f"Expected new exposures: {n_exposures_to_create_by_node_strain}")
                logger.info(f"Observed new exposures: {new_exposed_by_node_strain}")
                total_expected = np.sum(expected_exposures_per_node)
                tot_poisson_draw = np.sum(n_exposures_to_create_by_node_strain)
                # Check the number of people that are newly exposed
                num_new_exposed = np.sum(disease_state == 1) - np.sum(disease_state_pre_infect == 1)
                logger.info(
                    f"Tot exp exposures: {total_expected:.2f}, Total pois draw: {tot_poisson_draw}, Tot realized exposures: {num_new_exposed}"
                )

        if self.sim.t == self.sim.nt - 1:
            self.step_stats.log(logger)

        return

    def log(self, t):
        # Get the counts for each node in one pass
        S_counts, E_counts, I_counts, R_counts, E_by_strain_counts, I_by_strain_counts, POTP_counts, P_counts = count_SEIRP(
            node_id=self.people.node_id,
            disease_state=self.people.disease_state,
            strain=self.people.strain,
            potentially_paralyzed=self.people.potentially_paralyzed,
            paralyzed=self.people.paralyzed,
            n_nodes=np.int32(len(self.nodes)),
            n_strains=np.int32(len(self.pars.strain_ids)),
            n_people=np.int32(self.people.count),
        )

        # Store them in results
        self.results.S[t, :] = S_counts
        self.results.E[t, :] = E_counts  # Already summed across strains
        self.results.I[t, :] = I_counts  # Already summed across strains
        self.results.E_by_strain[t, :, :] = E_by_strain_counts  # Store strain-specific counts
        self.results.I_by_strain[t, :, :] = I_by_strain_counts  # Store strain-specific counts
        # Note that we add to existing non-zero EULA values for R
        self.results.R[t, :] += R_counts
        self.results.potentially_paralyzed[t, :] = POTP_counts
        self.results.paralyzed[t, :] = P_counts

        if self.verbose >= 3:
            logger.info(f"Exposed logged at end of timestep: {self.results.E[t, :]}")
            logger.info("")

    def plot(self, save=False, results_path=""):
        plot_network(self.network, save=save, results_path=results_path)


@nb.njit(parallel=True, cache=True)
def sample_dobs(samples, bin_min_age_days, bin_max_age_days, dobs):
    for i in nb.prange(len(samples)):
        dobs[i] = -np.random.randint(bin_min_age_days[samples[i]], bin_max_age_days[samples[i]])

    return


def pbincounts(bins, num_nodes, weights):
    tl_weights = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.float32)
    tl_counts = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)
    nb_bincounts(bins, len(bins), weights, tl_counts, tl_weights)

    return tl_counts.sum(axis=0), tl_weights.sum(axis=0)


# Version of utils.bincount the does two bincounts at once
@nb.njit(parallel=True, cache=True)
def nb_bincounts(bins, num_indices, weights, tl_counts, tl_weights):
    for i in nb.prange(num_indices):
        bidx = bins[i]
        tidx = nb.get_thread_id()
        tl_counts[tidx, bidx] += 1
        tl_weights[tidx, bidx] += weights[i]

    return


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
        cumulative_deaths = lp.create_cumulative_deaths(np.sum(self.pars.init_pop), max_age_years=100)
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
        self.people.add_scalar_property("date_of_birth", dtype=np.int32, default=-1)

        # --- Initialize ages & births ---
        if pars.init_sus_by_age is not None:
            # If we're initializing only susceptibles, we need to sample from the init_sus_by_age table for each node
            for node in self.nodes:
                pyramid = pars.init_sus_by_age[pars.init_sus_by_age["node_id"] == node]
                pyramid = pyramid.reset_index(drop=True)
                sampler = AliasedDistribution(pyramid["n_susceptible"])
                samples = sampler.sample(pars.init_sus[node])
                bin_min_age_days = (pyramid["age_min_yr"] * 365).astype(int).to_numpy()
                bin_max_age_days = (pyramid["age_max_yr"] * 365).astype(int).to_numpy()

                # Initialize ages
                mask = np.zeros(len(samples), dtype=bool)
                ages = np.zeros(len(samples), dtype=np.int32)

                for i in range(len(pyramid)):
                    mask[:] = samples == i
                    count = mask.sum()
                    if count > 0:
                        ages[mask] = np.random.randint(bin_min_age_days[i], bin_max_age_days[i], size=count)
                ages[ages <= 0] = 1
                node_mask = self.people.node_id[: self.people.count] == node
                self.people.date_of_birth[np.where(node_mask)[0]] = -ages

        else:
            # If we're initializing the entire population, we need to sample ages from the age pyramid
            pyramid = load_pyramid_csv(pars.age_pyramid_path)
            MINCOL = 0
            MAXCOL = 1
            MCOL = 2
            FCOL = 3
            sampler = AliasedDistribution(pyramid[:, MCOL] + pyramid[:, FCOL])  # using the male population in this example
            samples = sampler.sample(self.people.count)
            bin_min_age_days = pyramid[:, MINCOL] * 365  # minimum age for bin, in days (include this value)
            bin_min_age_days = np.maximum(bin_min_age_days, 1)  # No one born on day 0
            bin_max_age_days = (pyramid[:, MAXCOL] + 1) * 365  # maximum age for bin, in days (exclude this value)
            mask = np.zeros(self.people.count, dtype=bool)
            ages = np.zeros(self.people.count, dtype=np.int32)

            for i in range(len(pyramid)):
                mask[:] = samples == i
                ages[mask] = np.random.randint(bin_min_age_days[i], bin_max_age_days[i], mask.sum())

            ages[ages == 0] = 1
            self.people.date_of_birth[: self.people.count] = -ages

    def _initialize_deaths(self):
        pars = self.pars
        if pars.cbr is not None:
            self.results.add_array_property("births", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
            self.results.add_array_property("deaths", shape=(self.sim.nt, len(self.nodes)), dtype=np.int32)
            self.people.add_scalar_property("date_of_death", dtype=np.int32, default=0)

            cumulative_deaths = lp.create_cumulative_deaths(np.sum(pars.init_pop), max_age_years=100)
            self.death_estimator = KaplanMeierEstimator(cumulative_deaths)

            # Only compute lifespans if date_of_birth was initialized
            if "date_of_birth" in self.people.__dict__:
                ages = -self.people.date_of_birth[: self.people.count]
                lifespans = self.death_estimator.predict_age_at_death(ages, max_year=100)
                dods = lifespans - ages
                self.people.date_of_death[: self.people.count] = dods

                # sim.death_estimator = KaplanMeierEstimator(cumulative_deaths)
                # lifespans = sim.death_estimator.predict_age_at_death(-dobs, max_year=100)

                # # Set pars.life_expectancies to mean lifespans by node.
                # # This is just to support placeholder mortality premodeling for EULAs.
                # # Would move this code block to EULA section but we've got lifespans here.

                # num_nodes = len(self.nodes)
                # node_ids = sim.people.node_id[: sim.people.count]
                # counts, weighted_sums = pbincounts(node_ids, num_nodes, lifespans)
                # weighted_sums /= 365  # Convert to years

                # # Map unique_nodes to their computed life expectancies (safely handle divide-by-zero)
                # life_expectancies = np.zeros_like(weighted_sums)
                # where = counts > 0
                # with np.errstate(divide="ignore", invalid="ignore"):
                #     np.divide(weighted_sums, counts, out=life_expectancies, where=where)
                # pars.life_expectancies = life_expectancies

                # dods = sim.people.date_of_death[: sim.people.count]
                # dods[:] = dobs
                # dods += lifespans

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

            if pars.init_sus_by_age is not None:
                # If we're only initializing susceptibles, we need to account for mortality in the immune/recovered population
                df = pars.init_sus_by_age
                # Step 1: Compute average age per bin (in years)
                df["avg_age_yr"] = (df["age_min_yr"] + df["age_max_yr"]) / 2  # Compute average age in years
                # Step 2: Use node-level life expectancy to compute expected remaining life
                df["life_expectancy_yr"] = df["node_id"].map(lambda nid: pars.life_expectancies[nid])
                df["remaining_life_yr"] = np.maximum(df["life_expectancy_yr"] - df["avg_age_yr"], 1.0)  # Prevent div by 0
                # Step 3: Estimate daily mortality rate per bin (1 / expected remaining lifespan in days)
                df["daily_mortality_rate"] = 1 / (df["remaining_life_yr"] * 365)
                # Step 4: Estimate deaths per timestep per bin
                df["expected_deaths_per_day"] = df["n_immune"] * df["daily_mortality_rate"]
                # Step 5: Compute deaths by node
                deaths_by_node = df.groupby("node_id")["expected_deaths_per_day"].sum().to_numpy()
                T = self.results.deaths.shape[0]
                expected_deaths = np.outer(np.ones(T), deaths_by_node)  # shape = [T, num_nodes]
                self.results.deaths += np.random.poisson(expected_deaths)
                # Set deaths on day 0 to 0 since we're not running vital dynamics on day 0, only recording the initial states
                self.results.deaths[0, :] = 0

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

        return

    def step(self):
        t = self.sim.t
        if t % self.step_size != 0:
            # Returning from VD step without doing anything except we need to store the new pop
            # no births or deaths this cycle.
            self.results.pop[t, :] = self.results.pop[t - 1, :]
            return

        # 1) Get vital statistics - alive and newly deceased
        num_nodes = len(self.nodes)
        tl_dying = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)
        deaths_count_by_node = np.zeros(num_nodes, dtype=np.int32)
        get_deaths(
            num_nodes,
            self.people.count,
            self.people.disease_state,
            self.people.node_id,
            self.people.date_of_death,
            t,
            tl_dying,
            deaths_count_by_node,
        )
        # 2) Compute births
        expected_births = self.step_size * self.birth_rate * self.results.pop[t - 1]
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

            """
            # This was really useful for troubleshooting newborns
            import pandas as pd
            df = pd.DataFrame({
                key: val[start:end]
                for key, val in self.people.__dict__.items()
                if isinstance(val, np.ndarray) and val.shape[0] >= end
            })
            df.to_csv(f"newborns_t{t}.csv", index=False)
            """

        # 3) Store the death counts
        # Actual "death" handled in get_vital_statistics() as we count newly deceased
        self.results.deaths[t] = deaths_count_by_node

        self.results.pop[t, :] = (
            self.results.pop[t - 1, :]
            + self.results.births[t, :]  # updated at beginning of current step in vital dynamics
            - self.results.deaths[t, :]  # updated at beginning of current step in vital dynamics
        )

        return

    def log(self, t):
        pass

    def plot(self, save=False, results_path=None):
        plot_age_pyramid(self.people, self.pars, self.sim, save=save, results_path=results_path)
        plot_vital_dynamics(self.results, save=save, results_path=results_path)


@nb.njit(
    (nb.int32, nb.int32, nb.int8[:], nb.int16[:], nb.int32[:], nb.int32, nb.int32[:, :], nb.int32[:]),
    parallel=True,
    cache=True,
)
def get_deaths(num_nodes, num_people, disease_state, node_id, date_of_death, t, tl_dying, num_dying):
    # Iterate in parallel over all people
    for i in nb.prange(num_people):
        if disease_state[i] >= 0 and date_of_death[i] <= t:  # If they're past their due date ...
            disease_state[i] = -1  # Mark them as deceased
            tl_dying[nb.get_thread_id(), node_id[i]] += 1  # Count 'em as deceased

    num_dying[:] = tl_dying.sum(axis=0)  # Merge per-thread results

    return


@nb.njit(
    (
        nb.int64,
        nb.int16[:],
        nb.int8[:],
        nb.int8[:],
        nb.int8[:],
        nb.int16[:],
        nb.int64,
        nb.float64[:],
        nb.float64[:],
        nb.int64,
        nb.int32[:, :],
        nb.int32[:, :],
        nb.int32[:, :],
        nb.uint8[:],
        nb.int8[:],
        nb.int8,
    ),
    parallel=True,
    cache=True,
)
def fast_ri(
    step_size,
    node_id,
    disease_state,
    strain,
    ipv_protected,
    ri_timer,
    sim_t,
    vx_prob_ri,
    vx_prob_ipv,
    num_people,
    local_ri_counts,
    local_ri_protected,
    local_ipv_counts,
    chronically_missed,
    potentially_paralyzed,
    ri_vaccine_strain,
):
    """
    Optimized vaccination step with thread-local storage and parallel execution.
    """
    for i in nb.prange(num_people):
        state = disease_state[i]
        if state < 0:  # skip dead or inactive agents
            continue
        if chronically_missed[i] == 1:  # skip chronically missed agents
            continue

        node = node_id[i]
        prob_ri = vx_prob_ri[node]
        prob_ipv = vx_prob_ipv[node]
        timer = ri_timer[i] - step_size
        ri_timer[i] = timer
        eligible = False
        # If first vx, account for the fact that no components are run on day 0
        if sim_t == step_size:
            eligible = timer <= 0 and timer >= -step_size
        elif sim_t > step_size:
            eligible = timer <= 0 and timer > -step_size

        if eligible:
            if np.random.rand() < prob_ri:
                local_ri_counts[nb.get_thread_id(), node] += 1
                if state == 0:
                    # Expose to vaccine strain instead of immediate recovery
                    disease_state[i] = 1  # Set to exposed
                    strain[i] = ri_vaccine_strain  # Set vaccine strain
                    local_ri_protected[nb.get_thread_id(), node] += 1  # Increment protected count
                    # TODO remove when we have strain tracking hooked up into paralysis
                    potentially_paralyzed[i] = 0  # Assume that vaccine strains don't cause paralysis
            if np.random.rand() < prob_ipv:
                local_ipv_counts[nb.get_thread_id(), node] += 1
                ipv_protected[i] = 1

    return


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
        self.people.add_scalar_property("ri_timer", dtype=np.int16, default=-1)
        dob = self.people.date_of_birth[: self.people.count]
        days_from_birth_to_ri = np.random.uniform(42, 98, self.people.count)
        self.people.ri_timer[: self.people.count] = (dob + days_from_birth_to_ri).astype(np.int32)

    def _initialize_common(self):
        """Initialize common result arrays."""
        self.sim.results.add_array_property(
            "ri_vaccinated",
            shape=(self.sim.nt, len(self.sim.nodes)),
            dtype=np.int32,
        )
        self.sim.results.add_array_property(
            "ri_protected",
            shape=(self.sim.nt, len(self.sim.nodes)),
            dtype=np.int32,
        )
        self.sim.results.add_array_property(
            "ri_new_exposed_by_strain",
            shape=(self.sim.nt, len(self.sim.nodes), len(self.pars.strain_ids)),
            dtype=np.int32,
        )
        self.sim.results.add_array_property(
            "ipv_vaccinated",
            shape=(self.sim.nt, len(self.sim.nodes)),
            dtype=np.int32,
        )
        self.results = self.sim.results

    def step(self):
        # Handle OPV RI. If vx_prob_ri is None, we don't run this step.
        if self.pars["vx_prob_ri"] is None:
            return
        vx_prob_ri = self.pars["vx_prob_ri"]  # Includes coverage & efficacy
        num_nodes = len(self.sim.nodes)

        # Handle IPV RI. If vx_prob_ipv is None, fill with zeros so that IPV is not impactful.
        if self.pars["vx_prob_ipv"] is None:
            vx_prob_ipv = np.zeros(len(self.sim.nodes), dtype=np.float64)
        else:
            vx_prob_ipv = self.pars["vx_prob_ipv"]

        # Determine RI vaccine strain based on vaccine type
        ri_vaccine_type = getattr(self.pars, "ri_vaccine_type", "tOPV")  # Default to tOPV if not specified
        if "nOPV" in ri_vaccine_type:
            ri_vaccine_strain = np.int8(2)  # nOPV strain
        elif any(vtype in ri_vaccine_type for vtype in ["mOPV2", "tOPV", "topv"]):
            ri_vaccine_strain = np.int8(1)  # Sabin strain
        else:
            ri_vaccine_strain = np.int8(1)  # Default to Sabin strain

        # Promote to 1D arrays if needed
        if np.isscalar(vx_prob_ri):
            vx_prob_ri = np.full(num_nodes, vx_prob_ri, dtype=np.float64)
        if np.isscalar(vx_prob_ipv):
            vx_prob_ipv = np.full(num_nodes, vx_prob_ipv, dtype=np.float64)

        if self.sim.t % self.step_size == 0:
            local_ri_counts = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)
            local_ri_protected = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)
            local_ipv_counts = np.zeros((nb.get_num_threads(), num_nodes), dtype=np.int32)
            fast_ri(
                step_size=np.int32(self.step_size),
                node_id=self.people.node_id,
                disease_state=self.people.disease_state,
                strain=self.people.strain,
                ipv_protected=self.people.ipv_protected,
                ri_timer=self.people.ri_timer,
                sim_t=np.int32(self.sim.t),
                vx_prob_ri=vx_prob_ri,
                vx_prob_ipv=vx_prob_ipv,
                num_people=np.int32(self.people.count),
                local_ri_counts=local_ri_counts,
                local_ri_protected=local_ri_protected,
                local_ipv_counts=local_ipv_counts,
                chronically_missed=self.people.chronically_missed,
                potentially_paralyzed=self.people.potentially_paralyzed,
                ri_vaccine_strain=ri_vaccine_strain,
            )
            # Sum up the counts from all threads
            self.results.ri_vaccinated[self.sim.t] = local_ri_counts.sum(
                axis=0
            )  # Count those who received the vaccine, but didn't necessarily get protected/exposed
            self.results.ri_protected[self.sim.t] = local_ri_protected.sum(axis=0)  # Count those who received the vaccine and got protected
            self.results.new_exposed[self.sim.t] += local_ri_protected.sum(
                axis=0
            )  # Count those who received the vaccine and got protected, including wild transmission
            self.results.new_exposed_by_strain[self.sim.t, :, ri_vaccine_strain] += local_ri_protected.sum(
                axis=0
            )  # Count those who received the vaccine and got protected, including wild transmission
            self.results.ri_new_exposed_by_strain[self.sim.t, :, ri_vaccine_strain] = local_ri_protected.sum(
                axis=0
            )  # Count those who received the vaccine and got protected, only including SIA exposures
            self.results.ipv_vaccinated[self.sim.t] = local_ipv_counts.sum(axis=0)

        return

    def log(self, t):
        pass

    def plot(self, save=False, results_path=None):
        plot_cum_ri_vx(self.results, save=save, results_path=results_path)


@nb.njit(parallel=True, cache=True)
def fast_sia(
    node_ids,
    disease_states,
    strain,
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
    chronically_missed,
    potentially_paralyzed,
    sia_vaccine_strain,
):
    """
    Numbified supplemental immunization activity (SIA) vaccination step.

    Parameters:
        node_ids: Array of node IDs for each agent.
        disease_states: Array of disease states for each agent.
        strain: Array of strain IDs for each agent.
        dobs: Array of date of birth for each agent.
        sim_t: Current simulation timestep.
        vx_prob: Array of vaccination probabilities by node.
        vx_eff: Vaccine efficacy for this vaccine type (scalar).
        count: Number of active agents.
        nodes_to_vaccinate: Array of nodes targeted by this campaign.
        min_age, max_age: Integers, age range eligibility in days.
        local_vaccinated: Output array for vaccinated counts (threads x nodes).
        local_protected: Output array for protected counts (threads x nodes).
        chronically_missed: Array indicating chronically missed individuals.
        potentially_paralyzed: Array for paralysis tracking.
        sia_vaccine_strain: Integer strain ID for this vaccine type.
    """
    num_people = count

    for i in nb.prange(num_people):
        # Skip if agent is not alive, not in targeted node, or not in age range
        if disease_states[i] < 0:
            continue

        if chronically_missed[i] == 1:
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
                    disease_states[i] = 1  # Move to Exposed state (vaccine infection)
                    strain[i] = sia_vaccine_strain  # Set vaccine strain
                    local_protected[thread_id, node] += 1  # Increment protected count
                    # TODO remove when we have strain tracking hooked up into paralysis
                    potentially_paralyzed[i] = 0  # Vaccine strains don't cause paralysis

    return


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
        self.results.add_array_property(
            "sia_new_exposed_by_strain",
            shape=(self.sim.nt, len(self.nodes), len(self.pars.strain_ids)),
            dtype=np.int32,
        )

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
                nodes_to_vaccinate = np.zeros(len(self.sim.nodes), np.uint8)
                nodes_to_vaccinate[event["nodes"]] = 1  # Mark nodes to vaccinate
                vx_prob_sia = np.array(self.pars["vx_prob_sia"], dtype=np.float32)  # Convert to NumPy array
                vaccinetype = event["vaccinetype"]
                vx_eff = self.pars["vx_efficacy"][vaccinetype]
                min_age, max_age = event["age_range"]

                # Determine SIA vaccine strain based on vaccine type
                if "nOPV" in vaccinetype:
                    sia_vaccine_strain = np.int8(2)  # nOPV strain
                elif any(vtype in vaccinetype for vtype in ["mOPV", "tOPV", "topv"]):
                    sia_vaccine_strain = np.int8(1)  # Sabin strain
                else:
                    sia_vaccine_strain = np.int8(1)  # Default to Sabin strain

                # Suppose we have num_people individuals
                local_vaccinated = np.zeros((nb.get_num_threads(), len(self.sim.nodes)), dtype=np.int32)
                local_protected = np.zeros((nb.get_num_threads(), len(self.sim.nodes)), dtype=np.int32)
                fast_sia(
                    self.people.node_id,
                    self.people.disease_state,
                    self.people.strain,
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
                    chronically_missed=self.people.chronically_missed,
                    potentially_paralyzed=self.people.potentially_paralyzed,
                    sia_vaccine_strain=sia_vaccine_strain,
                )
                self.results.sia_vaccinated[t] = local_vaccinated.sum(
                    axis=0
                )  # Count those who received the vaccine, but didn't necessarily get protected/exposed
                self.results.sia_protected[t] = local_protected.sum(axis=0)  # Count those who received the vaccine and got protected
                self.results.new_exposed[t] += local_protected.sum(
                    axis=0
                )  # Count those who received the vaccine and got protected, including wild transmission
                self.results.new_exposed_by_strain[t, :, sia_vaccine_strain] += local_protected.sum(
                    axis=0
                )  # Count those who received the vaccine and got protected, including wild transmission
                self.results.sia_new_exposed_by_strain[t, :, sia_vaccine_strain] += local_protected.sum(
                    axis=0
                )  # Count those who received the vaccine and got protected, only including SIA exposures

        return

    def log(self, t):
        pass

    def plot(self, save=False, results_path=None):
        plot_cum_vx_sia(self.results, save=save, results_path=results_path)
