import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import laser_polio as lp

# ------------------- USER PARAMETERS -------------------
model_script = Path(lp.root / "calib/demo_zamfara.py").resolve(strict=True)
PARAMS_FILE = "params.json"
RESULTS_FILE = lp.root / "calib/results/calib_demo_zamfara/simulation_results.csv"
ACTUAL_DATA_FILE = lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"
# -------------------------------------------------------


def process_data(filename):
    """Load CSV and compute total infections and peak infection time."""
    df = pd.read_csv(filename)
    return {
        "total_infected": df["I"].sum(),
        "peak_infection_time": df.loc[df["I"].idxmax(), "Time"],
    }


def compute_fit(actual, predicted, use_squared=False, normalize=False, weights=None):
    """Compute goodness-of-fit between actual and predicted data."""
    fit = 0
    weights = weights or {}

    for key in actual:
        v1 = np.array(actual[key], dtype=float)
        v2 = np.array(predicted[key], dtype=float)
        gofs = np.abs(v1 - v2)

        if normalize and v1.max() > 0:
            gofs /= v1.max()
        if use_squared:
            gofs **= 2

        loss_weight = weights.get(key, 1)
        fit += (gofs * loss_weight).sum()

    return fit


def get_native_runstring():
    return [sys.executable, str(model_script)]


def objective(trial):
    """Optuna objective: run model with trial parameters and score result."""
    Path(RESULTS_FILE).unlink(missing_ok=True)

    r_nought = trial.suggest_float("r0", 4, 400)
    if not (4 <= r_nought <= 400):
        raise ValueError("Invalid r0 value suggested!")

    params = {"r0": r_nought}
    Path(PARAMS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)

    scores = []
    for _ in range(1):  # adjust replicates if needed
        try:
            subprocess.run(get_native_runstring(), check=True)
            actual = process_data(ACTUAL_DATA_FILE)
            predicted = process_data(RESULTS_FILE)
            scores.append(compute_fit(actual, predicted))
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed: {e}")
            return float("inf")

    Path(RESULTS_FILE).unlink(missing_ok=True)
    return np.mean(scores)
