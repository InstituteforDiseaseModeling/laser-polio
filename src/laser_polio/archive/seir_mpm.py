import matplotlib.pyplot as plt
import numpy as np
import sciris as sc  # for some datetime syntactic sugar
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet

from laser_polio.utils import get_seasonality


class CompartmentalSEIR:
    """
    A METAPOPULATION COMPARTMENTAL SEIR Model
    Each entry in the population is a node where counts of agents are tracked in the SEIR compartments.
    """

    def __init__(self, pars):
        self.pars = pars
        self.nodes = np.arange(len(pars.n_ppl))
        self.t = 0
        self.dates = sc.daterange(self.pars["start_date"], days=self.pars.dur)

        # Initialize node-level state variables
        self.results = LaserFrame(capacity=1)
        self.results.add_array_property("S", shape=(pars.dur, len(self.nodes)), dtype=np.float32)
        self.results.add_array_property("E", shape=(pars.dur, len(self.nodes)), dtype=np.float32)
        self.results.add_array_property("I", shape=(pars.dur, len(self.nodes)), dtype=np.float32)
        self.results.add_array_property("R", shape=(pars.dur, len(self.nodes)), dtype=np.float32)

        # Initialize populations
        self.results.S[0, :] = pars.n_ppl
        self.results.E[0, :] = 0
        self.results.I[0, :] = 0
        self.results.R[0, :] = 0

        # Seed initial infections
        for node, prev in enumerate(pars.init_prev):
            self.results.I[0, node] = pars.n_ppl[node] * prev
            self.results.S[0, node] -= self.results.I[0, node]

        # Components
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def run(self):
        for tick in range(1, self.pars.dur):
            for component in self.components:
                component.step(tick)
            self.t += 1

    def plot(self):
        self.plot_seir()
        self.plot_infected_by_node()

    def plot_seir(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.sum(self.results.S, axis=1), label="Susceptible (S)")
        plt.plot(np.sum(self.results.E, axis=1), label="Exposed (E)")
        plt.plot(np.sum(self.results.I, axis=1), label="Infected (I)")
        plt.plot(np.sum(self.results.R, axis=1), label="Recovered (R)")
        plt.title("SEIR Compartmental Dynamics")
        plt.xlabel("Time (Timesteps)")
        plt.ylabel("Population")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_infected_by_node(self):
        plt.figure(figsize=(10, 6))
        for node in self.nodes:
            plt.plot(self.results.I[:, node], label=f"Node {node}")
        plt.title("Infected Population by Node")
        plt.xlabel("Time (Timesteps)")
        plt.ylabel("Population")
        plt.legend()
        plt.grid()
        plt.show()


class SEIRTransitions:
    """
    Implements SEIR transition logic at the compartmental level.
    """

    def __init__(self, sim):
        self.sim = sim
        self.results = sim.results
        self.pars = sim.pars

    def step(self, t):
        """
        Update compartmental transitions for each node.
        """
        beta_global = self.pars.beta_global
        beta_dist = self.pars.beta_dist
        beta_seasonality = get_seasonality(self.sim)  # e.g., a single scalar
        sigma = self.pars.sigma  # Incubation rate (E -> I)
        gamma = self.pars.gamma  # Recovery rate (I -> R)

        S = self.results.S[t - 1, :]
        E = self.results.E[t - 1, :]
        I = self.results.I[t - 1, :]
        R = self.results.R[t - 1, :]
        N = S + E + I + R  # Total population per node

        # Compute the effective force of infection (FOI) using the contact matrix
        infectious_force = np.dot(beta_dist, I / np.maximum(N, 1))  # Weighted sum

        # Compute new transitions
        new_exposed = beta_global * beta_seasonality * S * infectious_force
        new_infected = sigma * E
        new_recovered = gamma * I

        # Update compartments
        self.results.S[t, :] = np.maximum(S - new_exposed, 0)
        self.results.E[t, :] = np.maximum(E + new_exposed - new_infected, 0)
        self.results.I[t, :] = np.maximum(I + new_infected - new_recovered, 0)
        self.results.R[t, :] = np.maximum(R + new_recovered, 0)


class SEIRVitalDynamics:
    """
    Births and deaths for compartmental SEIR.
    """

    def __init__(self, sim):
        self.sim = sim
        self.results = sim.results
        self.pars = sim.pars

    def step(self, t):
        """
        Handle births and deaths.
        """
        birth_rate = self.pars.cbr / (365 * 1000)  # Daily birth rate per person
        death_rate = self.pars.death_rate

        # Compute births
        births = birth_rate * (self.results.S[t - 1, :] + self.results.E[t - 1, :] + self.results.I[t - 1, :] + self.results.R[t - 1, :])

        # Compute deaths in each compartment
        deaths_S = death_rate * self.results.S[t - 1, :]
        deaths_E = death_rate * self.results.E[t - 1, :]
        deaths_I = death_rate * self.results.I[t - 1, :]
        deaths_R = death_rate * self.results.R[t - 1, :]

        # Apply births and deaths
        self.results.S[t, :] += births - deaths_S
        self.results.E[t, :] -= deaths_E
        self.results.I[t, :] -= deaths_I
        self.results.R[t, :] -= deaths_R


class RoutineVaccination:
    """
    Routine Immunization (RI) Component.
    Moves children from Susceptible (S) to Recovered (R).
    """

    def __init__(self, sim):
        self.sim = sim
        self.results = sim.results
        self.pars = sim.pars

    def step(self, t):
        """
        Vaccinate a fraction of newborns.
        """
        vx_prob = self.pars.vx_prob_ri  # Routine immunization probability
        births = self.pars.cbr / (365 * 1000) * np.sum(self.results.S[t - 1, :])  # Estimated newborns

        # Vaccinate newborns
        vaccinated = vx_prob * births
        self.results.S[t, :] -= vaccinated
        self.results.R[t, :] += vaccinated


# Example usage
if __name__ == "__main__":
    pars = PropertySet(
        {
            # Time
            "start_date": sc.date("2025-01-01"),  # Start date of the simulation
            "dur": 180,  # Number of timesteps
            # Population
            "n_ppl": np.array([30000, 10000, 15000, 20000, 25000]),
            # Disease
            "init_prev": np.array([0, 0.01, 0, 0, 0]),  # Initial prevalence per node (1% infected)
            "beta_global": 0.3,  # Global infection rate
            "seasonal_amplitude": 0.125,  # Seasonal variation in transmission
            "seasonal_peak_doy": 180,  # Phase of seasonal variation
            "sigma": 1 / 5,  # Incubation period ~5 days
            "gamma": 1 / 10,  # Recovery period ~10 days
            "p_paralysis": 1 / 20,  # Probability of paralysis
            # Migration
            "beta_dist": np.array(
                [
                    [0.8, 0.05, 0.05, 0.05, 0.05],
                    [0.05, 2.0, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.9, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 1.5, 0.05],
                    [0.05, 0.05, 0.05, 0.05, 0.5],
                ]
            ),
            # distances       = [[0, 20, 10, 30, 50],
            #                    []], # Distance in km
            # gravity_k       = 1,  # Gravity scaling constant
            # gravity_a       = 1,  # Origin population exponent
            # gravity_b       = 1,  # Destination population exponent
            # gravity_c       = 2.0,  # Distance exponent
            # migration_frac  = 0.01, # Fraction of population that migrates
            # Demographics & vital dynamics
            # age_pyramid_path= 'data/Nigeria_age_pyramid_2024.csv',  # From https://www.populationpyramid.net/nigeria/2024/
            "cbr": np.array([37, 41, 30, 25, 33]),  # Crude birth rate per 1000 per year
            "death_rate": 0.0001,  # Per capita daily death rate
            # Interventions
            "vx_prob_ri": np.array([0.1, 0.5, 0.01, 0, 0.2]),  # Probability of routine vaccination
        }
    )

    sim = CompartmentalSEIR(pars)
    sim.add_component(SEIRTransitions(sim))
    sim.add_component(SEIRVitalDynamics(sim))
    sim.add_component(RoutineVaccination(sim))

    sim.run()
    sim.plot()
