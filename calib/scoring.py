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
    Negative log-likelihood across heterogeneous components:
      - 1-bin (scalar) targets -> Poisson
      - >=2 bins (vector/matrix/nested) -> Dirichlet-Multinomial on the flattened bins

    Parameters
    ----------
    actual, predicted : dict
        Component -> data. Values can be scalars, lists/arrays, dicts, or dict-of-dicts/lists.
    weights : dict | None
        Optional per-component weights (default 1.0).
    rho / tau / tau_per_bin :
        How to set DM concentration tau (fixed across models). Priority: tau > tau_per_bin > rho*n_obs.
    per_bin : bool
        If True, normalize each DM log-likelihood by the number of bins K.
    eps : float
        Small positive number to avoid log(0) / zero probabilities.

    Returns
    -------
    dict
        Per-component negative log-likelihoods and "total_neg_ll".
    """
    log_likelihoods: dict[str, float] = {}
    weights = weights or {}

    def _to_matrix(obj):
        """
        Convert obj into a 2D array and (row_labels, col_labels).
        - scalar                   -> (1,1), rows=[None], cols=[None]
        - list/1D array            -> (1,K), rows=[None], cols=[0..K-1]
        - dict (1-level)           -> (1,K), rows=[None], cols=sorted(keys)
        - dict of dict/list (2-lvl)-> (R,C), rows=sorted(outer), cols=sorted(union inner)
        """
        # scalar
        if not isinstance(obj, (dict, list, tuple, np.ndarray)):
            return np.array([[float(obj)]]), [None], [None]

        # list/array
        if not isinstance(obj, dict):
            arr = np.asarray(obj, dtype=float).reshape(1, -1)
            return arr, [None], list(range(arr.shape[1]))

        # empty dict -> empty row
        if len(obj) == 0:
            return np.zeros((1, 0), dtype=float), [None], []

        values = list(obj.values())

        # dict of dict/list -> matrix
        if isinstance(values[0], (dict, list, tuple, np.ndarray)):
            row_labels = sorted(obj.keys())
            # Collect the set of inner keys from each row of the observed data to determine the union of all possible column labels (i.e., the column structure)
            inner_sets_obs = []
            for r in row_labels:
                v = obj[r]
                if isinstance(v, dict):
                    inner_sets_obs.append(set(v.keys()))
                else:
                    inner_sets_obs.append(set(range(len(v))))
            col_labels = sorted(set().union(*inner_sets_obs))
            R, C = len(row_labels), len(col_labels)
            M = np.zeros((R, C), dtype=float)
            for i, r in enumerate(row_labels):
                v = obj[r]
                if isinstance(v, dict):
                    for j, c in enumerate(col_labels):
                        M[i, j] = float(v.get(c, 0.0))
                else:
                    v_list = list(v)
                    for j, c in enumerate(col_labels):
                        if isinstance(c, int) and 0 <= c < len(v_list):
                            M[i, j] = float(v_list[c])
            return M, row_labels, col_labels

        # dict (1-level) -> single row vector by sorted keys
        col_labels = sorted(obj.keys())
        row = [float(obj[k]) for k in col_labels]
        return np.asarray([row], dtype=float), [None], col_labels

    def _align_pred(pred, row_labels, col_labels):
        """
        Align prediction into the same (R,C) shape as observed.
        Missing entries are treated as 0.0 (simple, explicit).
        """
        # scalar
        if row_labels == [None] and col_labels == [None]:
            try:
                return np.array([[float(pred)]], dtype=float)
            except Exception:
                return np.array([[0.0]], dtype=float)

        # vector
        if row_labels == [None]:
            if isinstance(pred, dict):
                return np.array([[float(pred.get(c, 0.0)) for c in col_labels]], dtype=float)
            v = np.asarray(pred, dtype=float).reshape(1, -1)
            out = np.zeros((1, len(col_labels)), dtype=float)
            for j, c in enumerate(col_labels):
                if isinstance(c, int) and c < v.shape[1]:
                    out[0, j] = v[0, c]
            return out

        # matrix
        R, C = len(row_labels), len(col_labels)
        out = np.zeros((R, C), dtype=float)
        if isinstance(pred, dict):
            for i, r in enumerate(row_labels):
                sub = pred.get(r, {})
                if isinstance(sub, dict):
                    for j, c in enumerate(col_labels):
                        out[i, j] = float(sub.get(c, 0.0))
                else:
                    sub_list = sub
                    for j, c in enumerate(col_labels):
                        if isinstance(c, int) and 0 <= c < len(sub_list):
                            out[i, j] = float(sub_list[c])
        return out

    def dm_rowwise(v_obs, v_sim, rho=1.0, eps=1e-12, average=False):
        v_obs = np.asarray(v_obs, int)
        v_sim = np.clip(np.asarray(v_sim, float), eps, None)
        R = v_obs.shape[0]
        total = 0.0
        rows_used = 0
        for i in range(R):
            x = v_obs[i]
            n = int(x.sum())
            if n == 0:
                continue
            p = v_sim[i] / v_sim[i].sum()
            tau = rho * n
            alpha = tau * p
            total += sps.dirichlet_multinomial.logpmf(x=x, n=n, alpha=alpha)
            rows_used += 1
        if average and rows_used > 0:
            total /= rows_used  # optional: mean per row
        return float(total)

    for key in actual.keys():
        try:
            a = actual[key]
            p = predicted[key]

            # 1) Align to the same 2D shape
            v_obs, row_labels, col_labels = _to_matrix(a)
            v_sim = _align_pred(p, row_labels, col_labels)

            # Ensure shape match
            if v_obs.shape != v_sim.shape:
                try:
                    sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v_obs.shape} vs {v_sim.shape}")
                except Exception:
                    print(f"[WARN] Shape mismatch on '{key}': {v_obs.shape} vs {v_sim.shape}")
                continue

            # 2) Clip simulated values to avoid negatives or log(0)
            v_sim = np.clip(v_sim, 1e-12, None)

            # Decide likelihood: scalar -> Poisson, else -> Dirichlet-multinomial
            if v_obs.size == 1:
                # ---- Poisson for scalar target ----
                y = np.round(v_obs[0, 0]).astype(int)
                lam = max(float(v_sim[0, 0]), 1e-12)
                logp = poisson.logpmf(y, lam)
            else:
                # ---- Dirichlet-Multinomial on FLATTENED bins ----
                logp = dm_rowwise(v_obs, v_sim)

            weight = float(weights.get(key, 1.0))
            neg_ll = -1.0 * weight * logp
            log_likelihoods[key] = float(neg_ll)

        except Exception as e:
            print(f"[ERROR] Skipping '{key}' due to: {e}")

    log_likelihoods["total_log_likelihood"] = float(sum(log_likelihoods.values()))
    return log_likelihoods
