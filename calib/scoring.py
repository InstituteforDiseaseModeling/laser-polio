import numpy as np
import scipy.stats as sps
import sciris as sc
from scipy.stats import nbinom
from scipy.stats import poisson


def compute_fit(actual, predicted, use_squared=False, normalize=False, weights=None):
    """Compute distance between actual and predicted summary metrics."""
    fit = 0
    weights = weights or {}

    for key in actual:
        if key not in predicted:
            print(f"[WARN] Key missing in predicted: {key}")
            continue

        try:
            v1 = np.array(actual[key], dtype=float)
            v2 = np.array(predicted[key], dtype=float)

            if v1.shape != v2.shape:
                sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v1.shape} vs {v2.shape}")
                continue

            gofs = np.abs(v1 - v2)

            if normalize and v1.max() > 0:
                gofs = gofs / v1.max()
            if use_squared:
                gofs = gofs**2

            weight = weights.get(key, 1)
            fit += (gofs * weight).sum()

        except Exception as e:
            print(f"[ERROR] Skipping '{key}' due to: {e}")

    return fit


def compute_log_likelihood_fit(
    actual,
    predicted,
    method="poisson",
    dispersion=1.0,
    weights=None,
    norm_by_n=True,
):
    """
    Compute log-likelihood of actual data given predicted data, including nested dicts.

    Parameters:
        actual (dict): Observed summary statistics. Can include nested dicts.
        predicted (dict): Simulated summary statistics. Must mirror actual.
        method (str): Distribution to use.
        dispersion (float): Used for neg_binomial.
        weights (dict): Optional weights per target.
        norm_by_n (bool): Normalize by data length.

    Returns:
        dict: log-likelihoods per key + total.
    """
    log_likelihoods = {}
    weights = weights or {}

    for key in actual:
        try:
            a = actual[key]
            p = predicted[key]

            # Handle nested dictionaries
            if isinstance(a, dict) and isinstance(p, dict):
                if set(a.keys()) != set(p.keys()):
                    raise ValueError(f"Key mismatch in nested dict '{key}': {set(a.keys()) ^ set(p.keys())}")
                subkeys = sorted(a.keys())  # enforce consistent order
                v_obs = np.array([a[k] for k in subkeys], dtype=float)
                v_sim = np.array([p[k] for k in subkeys], dtype=float)

            # Handle flat arrays or lists
            else:
                v_obs = np.array(a, dtype=float)
                v_sim = np.array(p, dtype=float)

            # Ensure shape match
            if v_obs.shape != v_sim.shape:
                sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v_obs.shape} vs {v_sim.shape}")
                continue

            # Clip simulation values to avoid log(0)
            v_sim = np.clip(v_sim, 1e-6, None)

            if method == "poisson":
                logp = poisson.logpmf(v_obs, v_sim)
            elif method == "neg_binomial":
                r = dispersion
                p = r / (r + v_sim)
                logp = nbinom.logpmf(v_obs, r, p)
            else:
                raise ValueError(f"Unknown method '{method}'")

            weight = weights.get(key, 1.0)
            n = len(logp)
            normalizer = n if norm_by_n else 1
            neg_ll = -1.0 * weight * logp.sum() / normalizer
            log_likelihoods[key] = neg_ll

        except Exception as e:
            print(f"[ERROR] Skipping '{key}' due to: {e}")

    log_likelihoods["total_log_likelihood"] = sum(log_likelihoods.values())
    return log_likelihoods


def compute_nll_dirichlet(actual, predicted, weights=None):
    """
    Compute log-likelihood using mixed approaches:
    - Poisson for total_by_period (count data)
    - Dirichlet multinomial for monthly_timeseries and adm01_by_period (compositional data)

    Parameters:
        actual (dict): Observed summary statistics
        predicted (dict): Simulated summary statistics
        weights (dict): Optional weights per target
        norm_by_n (bool): Normalize by data length
        dirichlet_alpha (float): Concentration parameter for Dirichlet distribution

    Returns:
        dict: log-likelihoods per key + total
    """
    log_likelihoods = {}
    weights = weights or {}

    for key in actual:
        try:
            a = actual[key]
            p = predicted[key]

            # Handle nested dictionaries (for adm01_by_period)
            if isinstance(a, dict) and isinstance(p, dict):
                if set(a.keys()) != set(p.keys()):
                    raise ValueError(f"Key mismatch in nested dict '{key}': {set(a.keys()) ^ set(p.keys())}")
                subkeys = sorted(a.keys())  # enforce consistent order
                v_obs = np.array([a[k] for k in subkeys], dtype=float)
                v_sim = np.array([p[k] for k in subkeys], dtype=float)

            # Handle flat arrays or lists
            else:
                v_obs = np.array(a, dtype=float)
                v_sim = np.array(p, dtype=float)

            # Ensure shape match
            if v_obs.shape != v_sim.shape:
                sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v_obs.shape} vs {v_sim.shape}")
                continue

            # Clip simulation values to avoid log(0)
            v_sim = np.clip(v_sim, 1e-6, None)

            if key == "total_by_period":
                # Use Poisson likelihood for count data
                logp = poisson.logpmf(v_obs, v_sim)

            elif key in ["monthly_timeseries", "adm01_by_period"]:
                # Use Dirichlet multinomial likelihood for compositional data
                logp = sps.dirichlet_multinomial.logpmf(x=v_obs, n=v_obs.sum(), alpha=v_sim + 1)

            else:
                # Default to Poisson for other keys
                logp = poisson.logpmf(v_obs, v_sim)

            weight = weights.get(key, 1.0)
            neg_ll = -1.0 * weight * logp.sum()
            log_likelihoods[key] = neg_ll

        except Exception as e:
            print(f"[ERROR] Skipping '{key}' due to: {e}")

    log_likelihoods["total_log_likelihood"] = sum(log_likelihoods.values())
    return log_likelihoods
