import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

# --- your data ---
actual = [0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 0]
preds = {
    "sametime_low": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    "sametime_high": [0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 0],
    "match": [0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 0],
    "difftime": [0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 0, 0],
    "longertime": [0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 0],
    "longertime_high": [0, 0, 0, 0, 0, 10, 10, 10, 100, 100, 100, 0],
}


# --- log-likelihood (Dirichlet-multinomial) ---
def dm_logp(x_counts, pred_counts, eps=1e-12):
    x = np.asarray(x_counts, dtype=int)
    n = int(x.sum())
    alpha = np.clip(np.asarray(pred_counts, dtype=float), eps, None) + 1.0
    return float(sps.dirichlet_multinomial.logpmf(x=x, n=n, alpha=alpha))


logps = {name: dm_logp(actual, yhat) for name, yhat in preds.items()}


# --- log-likelihood (Negative binomial) ---
# Fixed negative binomial function
def nb_logp(x_counts, pred_counts, dispersion=1.0, eps=1e-12):
    """
    Calculate log likelihood using negative binomial distribution.

    Parameters:
    -----------
    x_counts : array-like
        Actual observed counts
    pred_counts : array-like
        Predicted/expected counts
    dispersion : float
        Dispersion parameter (higher = more overdispersion)
        Can be tuned: smaller values (e.g., 0.1) make it more sensitive to differences
    eps : float
        Small value to avoid numerical issues
    """
    # Convert to numpy arrays first
    x = np.asarray(x_counts, dtype=float)
    pred = np.asarray(pred_counts, dtype=float)

    # Now clip to avoid zero values
    pred = np.clip(pred, eps, None)

    log_lik = 0
    for x_i, pred_i in zip(x, pred, strict=False):
        # Use predicted count as the mean of the negative binomial
        mean = pred_i

        # Convert mean and dispersion to n and p parameters
        # In scipy's parameterization: mean = n * (1-p) / p
        if mean > 0:
            p = dispersion / (dispersion + mean)
            n = dispersion
        else:
            # Handle zero mean case
            p = 1.0
            n = dispersion

        # Calculate log probability
        log_lik += sps.nbinom.logpmf(x_i, n, p)

    return log_lik


logps_nb = {name: nb_logp(actual, yhat) for name, yhat in preds.items()}

# --- plotting ---
names = list(preds.keys())
# Sort by logp (best at top of bar chart)
order = np.argsort([logps[n] for n in names])[::-1]
names_sorted = [names[i] for i in order]
logps_sorted = [logps[n] for n in names_sorted]

# Shared y-limit across all time-series panels
ymax = 1.05 * max(np.max(actual), max(np.max(preds[n]) for n in names))

rows = len(names)
fig = plt.figure(figsize=(12, 2.2 * rows + 2))
gs = GridSpec(rows, 2, width_ratios=[3.0, 1.4], hspace=0.5, wspace=0.3)

# Left column: one row per prediction
x = np.arange(len(actual))
for r, name in enumerate(names_sorted):
    ax = fig.add_subplot(gs[r, 0])
    ax.plot(x, actual, label="actual", color="black", lw=2)
    ax.plot(x, preds[name], label=name, lw=2, alpha=0.8)
    ax.set_ylim(0, ymax)
    if r == rows - 1:
        ax.set_xlabel("time (index)")
    ax.set_ylabel("cases")
    ax.grid(alpha=0.3)
    # annotate logp on the panel
    ax.text(
        0.98,
        0.95,
        f"logp = {logps[name]:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    if r == 0:
        ax.legend(loc="upper left", ncols=2, fontsize=9, frameon=True)

# Right column: bar chart of logps
axb = fig.add_subplot(gs[:, 1])
axb.barh(names_sorted, logps_sorted)
axb.invert_yaxis()  # best at top
axb.set_xlabel("Dirichlet-multinomial log-likelihood")
axb.set_title("Model fit (higher is better)")
# highlight best
best_val = logps_sorted[0]
axb.axvline(best_val, ls="--", lw=1)
for y, v in enumerate(logps_sorted):
    axb.text(v, y, f" {v:.2f}", va="center", ha="left", fontsize=9)

fig.suptitle("Actual vs Predicted (with log-likelihood)", y=0.995, fontsize=13)
fig.savefig("results/debug_dirichlet.png")
fig.show()


# --- safe Dirichlet-multinomial logpmf (includes multinomial coefficient) ---
def dirichlet_multinomial_logpmf(x, alpha):
    x = np.asarray(x, dtype=np.int64)
    alpha = np.asarray(alpha, dtype=np.float64)
    n = int(x.sum())
    A = float(alpha.sum())
    if np.any(alpha <= 0):
        return -np.inf
    # log [ n! / ∏ x_i! ] + log [ Γ(A) / Γ(A+n) ∏ Γ(alpha_i + x_i) / Γ(alpha_i) ]
    return gammaln(n + 1) - np.sum(gammaln(x + 1)) + gammaln(A) - gammaln(A + n) + np.sum(gammaln(alpha + x) - gammaln(alpha))


def normalize_probs(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, eps, None)
    v /= v.sum()
    return v


# --- Option A: fixed-tau comparison (simple and fair) ---
def dm_logp_fixed_tau(x_counts, pred_counts, tau=50.0, eps=1e-12):
    x = np.asarray(x_counts, dtype=np.int64)
    p = normalize_probs(pred_counts, eps=eps)
    alpha = p * float(tau)  # same strength across models
    return float(dirichlet_multinomial_logpmf(x, alpha))


# --- Option B: profile (maximize) tau per model (if you want to fit dispersion) ---
def dm_logp_mle_tau(x_counts, pred_counts, eps=1e-12):
    x = np.asarray(x_counts, dtype=np.int64)
    p = normalize_probs(pred_counts, eps=eps)

    # optimize over tau > 0 (work in log-space for stability)
    def obj(log_tau):
        tau = np.exp(log_tau)
        alpha = p * tau
        return -dirichlet_multinomial_logpmf(x, alpha)

    res = minimize_scalar(obj, bounds=(np.log(1e-3), np.log(1e6)), method="bounded")
    if not res.success:
        # fall back to a reasonable default
        return dm_logp_fixed_tau(x, pred_counts, tau=x.sum() or 1.0, eps=eps)
    tau_hat = float(np.exp(res.x))
    return float(dirichlet_multinomial_logpmf(x, p * tau_hat))


# Compare under a fixed tau (e.g., tau = sum(actual))
tau = sum(actual) or 50.0
logps_fixed = {name: dm_logp_fixed_tau(actual, yhat, tau=tau) for name, yhat in preds.items()}

# Or compare allowing each model to fit its own tau (profiled/MLE)
logps_prof = {name: dm_logp_mle_tau(actual, yhat) for name, yhat in preds.items()}

# PLOTTING


def plot_actual_vs_preds_with_logps(
    actual,
    preds: dict[str, list | np.ndarray],
    logps: dict[str, float],
    *,
    title: str = "Actual vs Predicted (with log-likelihood)",
    outfile: str | None = None,
    sort_by_logp: bool = True,
    common_ymax: float | None = None,
    ymax_pad_frac: float = 0.05,
    xlabel: str = "time (index)",
    ylabel: str = "cases",
):
    """
    Plot actual vs predicted time series (small multiples) + a bar chart of logps.

    Parameters
    ----------
    actual : array-like
        Observed counts/values over time.
    preds : dict[str, array-like]
        Mapping model name -> predicted time series (same length as `actual`).
    logps : dict[str, float]
        Mapping model name -> scalar log-likelihood (e.g., Dirichlet-multinomial).
    title : str
        Figure title.
    outfile : str | None
        If given, saves the figure to this path.
    sort_by_logp : bool
        If True, order models by logp descending (best at top/right).
    common_ymax : float | None
        If set, fixes all time-series panels to [0, common_ymax]. If None, compute from data.
    ymax_pad_frac : float
        Headroom fraction added to the computed ymax (ignored if common_ymax is given).
    xlabel, ylabel : str
        Axis labels for time-series panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    (axes_ts, ax_bar) : (list[Axes], Axes)
        Axes for time-series panels and the bar chart.
    """
    # Align names = intersection of preds & logps keys, preserving preds order then filtered
    names = [k for k in preds.keys() if k in logps]
    if not names:
        raise ValueError("No overlapping model names between `preds` and `logps`.")

    # Sort by logp (best first) if requested
    if sort_by_logp:
        names = sorted(names, key=lambda k: logps[k], reverse=True)

    # Validate lengths and compute common ymax if not provided
    actual = np.asarray(actual, dtype=float)
    x = np.arange(len(actual))

    for k in names:
        yk = np.asarray(preds[k], dtype=float)
        if yk.shape != actual.shape:
            raise ValueError(f"Prediction '{k}' has shape {yk.shape}, expected {actual.shape}.")

    if common_ymax is None:
        ymax_data = max(np.nanmax(actual), max(np.nanmax(np.asarray(preds[k], dtype=float)) for k in names))
        common_ymax = (1.0 + max(0.0, float(ymax_pad_frac))) * (0.0 if np.isnan(ymax_data) else float(ymax_data))

    rows = len(names)
    fig = plt.figure(figsize=(12, 2.2 * rows + 2))
    gs = GridSpec(rows, 2, width_ratios=[3.0, 1.4], hspace=0.5, wspace=0.3)

    axes_ts = []
    # Left column: one small-multiple per model
    for r, name in enumerate(names):
        ax = fig.add_subplot(gs[r, 0])
        ax.plot(x, actual, label="actual", color="black", lw=2)
        ax.plot(x, preds[name], label=name, lw=2, alpha=0.85)
        ax.set_ylim(0, common_ymax)
        if r == rows - 1:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

        # annotate logp on the panel
        ax.text(
            0.98,
            0.95,
            f"logp = {logps[name]:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        if r == 0:
            ax.legend(loc="upper left", ncols=2, fontsize=9, frameon=True)
        axes_ts.append(ax)

    # Right column: bar chart of logps
    axb = fig.add_subplot(gs[:, 1])
    vals = [logps[n] for n in names]
    axb.barh(names, vals)
    axb.invert_yaxis()  # best at top if sorted
    axb.set_xlabel("log-likelihood")
    axb.set_title("Model fit (higher is better)")
    if len(vals) > 0:
        best_val = max(vals)
        axb.axvline(best_val, ls="--", lw=1)
    for y, (_n, v) in enumerate(zip(names, vals, strict=False)):
        axb.text(v, y, f" {v:.2f}", va="center", ha="left", fontsize=9)

    fig.suptitle(title, y=0.995, fontsize=13)

    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    return fig, (axes_ts, axb)


fig, axes = plot_actual_vs_preds_with_logps(actual, preds, logps_fixed, title="Fixed tau", outfile="results/debug_dirichlet_fixed.png")
fig, axes = plot_actual_vs_preds_with_logps(actual, preds, logps_prof, title="Profiled tau", outfile="results/debug_dirichlet_prof.png")
fig, axes = plot_actual_vs_preds_with_logps(actual, preds, logps_nb, title="Negative binomial", outfile="results/debug_nb.png")

print("done")
