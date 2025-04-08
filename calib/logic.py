# calib/calib_logic.py

import numpy as np
import pandas as pd


def process_data(filename):
    """Load simulation results and extract features for comparison."""
    df = pd.read_csv(filename)
    return {
        "total_infected": df["I"].sum(),
        "peak_infection_time": df.loc[df["I"].idxmax(), "Time"],
    }


def compute_fit(actual, predicted, use_squared=False, normalize=False, weights=None):
    """Compute distance between actual and predicted summary metrics."""
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
