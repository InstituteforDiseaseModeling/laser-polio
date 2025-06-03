import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sc
import seaborn as sns
from laser_core.propertyset import PropertySet

import laser_polio as lp


def make_sim(n_ppl=100e3, n_nodes=1, dur=365):
    pop = n_ppl / n_nodes * np.ones(n_nodes)
    dist_matrix = np.ones((n_nodes, n_nodes))
    init_prev = np.zeros(n_nodes) + 0.01
    r0_scalars = np.ones(n_nodes)
    cbr = np.ones(n_nodes) * 30
    vx_prob_ri = np.ones(n_nodes) * 0.1

    # Set parameters
    pars = PropertySet(
        {
            # Time
            "start_date": sc.date("2025-01-01"),  # Start date of the simulation
            "dur": dur,  # Number of timesteps
            # Population
            "n_ppl": pop,  # np.array([30000, 10000, 15000, 20000, 25000]),
            "distances": dist_matrix,  # Distance in km
            # Disease
            "init_prev": init_prev,  # Initial prevalence per node (1% infected)
            "beta_global": 0.3,  # Global infection rate
            "r0_scalars": r0_scalars,  # Spatial transmission scalar (multiplied by global rate)
            "seasonal_amplitude": 0.125,  # Seasonal variation in transmission
            "seasonal_peak_doy": 180,  # Phase of seasonal variation
            "p_paralysis": 1 / 20,  # Probability of paralysis
            # Migration
            "gravity_k": 1,  # Gravity scaling constant
            "gravity_a": 1,  # Origin population exponent
            "gravity_b": 1,  # Destination population exponent
            "gravity_c": 2.0,  # Distance exponent
            "migration_frac": 0.01,  # Fraction of population that migrates
            # Demographics & vital dynamics
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "cbr": cbr,  # np.array([37, 41, 30, 25, 33]),  # Crude birth rate per 1000 per year
            # Interventions
            "vx_prob_ri": vx_prob_ri,  # Probability of routine vaccination
        }
    )

    # Initialize the sim
    sim = lp.SEIR_ABM(pars)
    sim.add_component(lp.DiseaseState_ABM(sim))
    sim.add_component(lp.Transmission_ABM(sim))
    sim.add_component(lp.Migration_ABM(sim))
    sim.add_component(lp.VitalDynamics_ABM(sim))
    sim.add_component(lp.RI_ABM(sim))

    # Run the simulation
    sim.run()


if __name__ == "__main__":
    # # Run profiler
    # T = sc.timer()
    # cpr = sc.cprofile()
    # cpr.start()
    # n_ppl = 1e3
    # n_nodes = 5
    # dur = 365
    # make_sim(n_ppl=n_ppl, n_nodes=n_nodes, dur=dur)
    # cpr.stop()
    # df=cpr.to_df()
    # filename = f'scripts/profiling/Profile for SEIR with pop{int(n_ppl/1000)}k {n_nodes} nodes and dur of {dur} with speedups.csv'
    # df.to_csv(filename, index=False)

    # Track sim run times
    n_ppl_values = [1e5, 1e6, 1e7]
    n_nodes_values = [1, 10, 100]
    results = []
    for n_ppl in n_ppl_values:
        for n_nodes in n_nodes_values:
            start_time = time.time()
            make_sim(n_ppl=n_ppl, n_nodes=n_nodes)
            end_time = time.time()
            elapsed_time = end_time - start_time
            results.append({"n_ppl": n_ppl, "n_nodes": n_nodes, "time": elapsed_time})
            print(f"n_ppl: {n_ppl}, n_nodes: {n_nodes}, time: {elapsed_time}")
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("data/simulation_times.csv", index=False)

    # # Version from 2025-02-04
    # results = dict(n_ppl=[1e5, 1e5, 1e5, 1e6, 1e6, 1e6, 1e7, 1e7,],
    #                n_nodes=[1, 10, 100, 1, 10, 100, 1, 10, ],
    #                time=[18.6, 142.3, 1105, 107, 1185, 13395, 928, 9916, ])
    # results['time'] = [t / 60 for t in results['time']]  # Convert time to minutes
    # results_df = pd.DataFrame(results)
    # print(results_df)
    # results_df.to_csv("data/simulation_times_20250204.csv", index=False)

    # # Version from 2025-02-06
    # results = dict(n_ppl=[1e5, 1e5, 1e5, 1e6, 1e6, 1e6, 1e7, 1e7, 1e7],
    #                n_nodes=[1, 10, 100, 1, 10, 100, 1, 10, 100],
    #                time=[2.4, 3.8, 12.3, 16.8, 37.9, 233.0, 142.7, 363.0, 2935.0])
    # results['time'] = [t / 60 for t in results['time']]  # Convert time to minutes
    # results_df = pd.DataFrame(results)
    # print(results_df)
    # results_df.to_csv("data/simulation_times_20250206.csv", index=False)

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="n_ppl", y="time", hue="n_nodes", marker="o", palette="bright")
    plt.xscale("log")
    plt.xlabel("Number of people (n_ppl)")
    plt.ylabel("Time (minutes)")
    plt.title("Model run times as of 2024-02-06")
    plt.legend(title="n_nodes")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("scripts/profiling/model_run_times_plot_with_speedups_20250206.png")
    plt.show()

    print("Done.")
