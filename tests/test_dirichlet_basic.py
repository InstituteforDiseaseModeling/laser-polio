import sys

import numpy as np

sys.path.append("calib")  # do not like

# If you’re using the debug version with logging:
from scoring import compute_nll_dirichlet as compute_nll_dirichlet


def triangle_counts(n=50, total=1000):
    # single-period triangle, scaled to integer-ish counts (round later if needed)
    up = np.linspace(0, 1, n // 2, endpoint=False)
    down = np.linspace(1, 0, n - len(up))
    tri = np.concatenate([up, down])
    tri = tri / tri.sum() * total
    return tri


def test_perfect_match_poisson():
    actual = {"scalar": 7.0}
    predicted_same = {"scalar": 7.0}
    predicted_pert = {"scalar": 8.5}

    nll_same = compute_nll_dirichlet(actual, predicted_same)["total_log_likelihood"]
    nll_pert = compute_nll_dirichlet(actual, predicted_pert)["total_log_likelihood"]

    print("[Poisson] same vs perturbed:", nll_same, nll_pert)
    assert nll_same <= nll_pert + 1e-9, "Exact Poisson match should not be worse than a perturbation."


def test_perfect_match_vector_dm():
    # Use integer counts to avoid rounding artifacts inside the scorer
    counts = np.array([3, 1, 0, 4, 2, 5, 7, 0, 6, 8], dtype=float)
    actual = {"vec": counts.tolist()}
    predicted_same = {"vec": counts.tolist()}
    predicted_pert = {"vec": (counts * 1.2).tolist()}  # scaled away from exact

    nll_same = compute_nll_dirichlet(actual, predicted_same)["total_log_likelihood"]
    nll_pert = compute_nll_dirichlet(actual, predicted_pert)["total_log_likelihood"]

    print("[Vector DM] same vs perturbed:", nll_same, nll_pert)
    assert nll_same <= nll_pert + 1e-9, "Exact DM match (vector) should not be worse than a perturbation."


def test_perfect_match_matrix_dm():
    # Two rows (outer keys), shared inner columns after union/sort
    actual = {
        "rowA": {"a": 5, "b": 0, "c": 2},
        "rowB": {"a": 1, "c": 7},  # missing 'b' -> will be treated as 0
    }
    predicted_same = {
        "rowA": {"a": 5, "b": 0, "c": 2},
        "rowB": {"a": 1, "c": 7},
    }
    # Slight perturbation: scale rowB and tweak rowA a bit
    predicted_pert = {
        "rowA": {"a": 6.0, "b": 0.0, "c": 1.0},
        "rowB": {"a": 1.2, "c": 6.8},
    }

    nll_same = compute_nll_dirichlet(actual, predicted_same)["total_log_likelihood"]
    nll_pert = compute_nll_dirichlet(actual, predicted_pert)["total_log_likelihood"]

    print("[Matrix DM] same vs perturbed:", nll_same, nll_pert)
    assert nll_same <= nll_pert + 1e-9, "Exact DM match (matrix) should not be worse than a perturbation."


def test_perfect_match_triangle_dm():
    # Triangle wave as counts (DM path). Keep totals decent for resolution.
    tri = triangle_counts(n=50, total=5000)  # total counts ~5k
    actual = {"wave": tri.tolist()}
    predicted_same = {"wave": tri.tolist()}
    predicted_scaled = {"wave": (tri * 1.25).tolist()}  # scale away from exact

    nll_same = compute_nll_dirichlet(actual, predicted_same)["total_log_likelihood"]
    nll_scaled = compute_nll_dirichlet(actual, predicted_scaled)["total_log_likelihood"]

    print("[Triangle DM] same vs scaled:", nll_same, nll_scaled)
    assert nll_same <= nll_scaled + 1e-9, "Exact DM match (triangle) should not be worse than scaled prediction."


def run_all():
    test_perfect_match_poisson()
    test_perfect_match_vector_dm()
    test_perfect_match_matrix_dm()
    test_perfect_match_triangle_dm()
    print("Perfect-match tests: OK")


if __name__ == "__main__":
    run_all()
