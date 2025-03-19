import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

laser_script = Path("laser.py").resolve(strict=True)


def process_data(filename):
    """
    Load simulation results from a CSV file and calculate interesting stats based on the I column.
    """
    df = pd.read_csv(filename)

    # Calculate statistics for the I column
    tot_infected = df["I"].sum()
    peak_infection_time = df.loc[df["I"].idxmax(), "Time"]

    data = {
        "total_infected": tot_infected,
        "peak_infection_time": peak_infection_time,
    }

    print(f"Total Infected: {tot_infected}")
    print(f"Peak Infection Time: {peak_infection_time}")

    return data


def compute_fit(actual, predicted, use_squared=False, normalize=False, weights=None):
    """
    Compute the fit between actual and predicted data.
    """

    fit = 0

    if weights is None:
        weights = {}

    for skey in actual:
        v1 = np.array(actual[skey], dtype=float)
        v2 = np.array(predicted[skey], dtype=float)

        gofs = abs(np.array(v1) - np.array(v2))
        if normalize:
            actual_max = abs(v1).max()
            if actual_max > 0:
                gofs /= actual_max
        if use_squared:
            gofs = gofs**2
        if skey not in weights:
            weights = {skey: 1}
        losses = gofs * weights[skey]
        mismatch = losses.sum()
        fit += mismatch

    return fit


# Paths to input/output files
PARAMS_FILE = "params.json"
RESULTS_FILE = "simulation_results.csv"
ACTUAL_DATA_FILE = "data/seir_counts_r0_200.csv"


def objective(trial):
    """Optuna objective function that runs laser.py with trial parameters and evaluates results."""

    Path(RESULTS_FILE).unlink(missing_ok=True)

    # Suggest values for calibration
    # migration_rate = trial.suggest_float("migration_rate", 0.0001, 0.01)
    r_nought = trial.suggest_float("r0", 4, 400)
    # migration_rate = trial.suggest_float("migration_rate", 0.004, 0.004)
    # transmission_rate = trial.suggest_float("transmission_rate", 0.145, 0.145)
    print(f"DEBUG: r_nought selected as {r_nought}")
    if r_nought < 4 or r_nought > 400:
        raise ValueError("optuna selected bogus value for r0!")

    # Set up parameters
    params = {"r0": r_nought}

    # Write parameters to JSON file
    with Path(PARAMS_FILE).open("w") as f:
        json.dump(params, f, indent=4)

    def get_docker_runstring():
        # cmd = f"docker run --rm -v .:/app/shared docker.io/jbloedow/my-laser-app:latest"
        cmd = "docker run --rm -v .:/app/shared my-laser-app:latest"
        return cmd.split()

    def get_native_runstring():
        return [sys.executable, str(laser_script)]

    NUM_REPLICATES_PER_TRIAL = 1
    print(f"Will be looking for {RESULTS_FILE}")
    # Run laser.py as a subprocess
    try:
        # Run the model 4 times and collect results
        scores = []
        for _ in range(NUM_REPLICATES_PER_TRIAL):
            # subprocess.run(get_docker_runstring(), check=True)
            # subprocess.run(["python3", "laser.py"], check=True)
            subprocess.run(get_native_runstring(), check=True)

            print(f"Waiting for output file {RESULTS_FILE}")
            # Wait until RESULTS_FILE is written
            while not Path(RESULTS_FILE).exists():
                print(os.listdir("."))
                time.sleep(0.1)

            print("Reference...")
            actual = process_data(ACTUAL_DATA_FILE)
            print("Simulation...")
            predicted = process_data(RESULTS_FILE)

            score = compute_fit(actual, predicted)  # Evaluate results
            scores.append(score)

        # Return the average score
        return np.mean(scores)

    except subprocess.CalledProcessError as e:
        print(f"Error running laser.py: {e}")
        return float("inf")  # Penalize failed runs
