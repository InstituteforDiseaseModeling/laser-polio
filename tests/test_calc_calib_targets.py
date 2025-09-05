import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make "calib/" importable (adjust if your project layout differs)
sys.path.append("calib")

from targets import calc_calib_targets
from targets import calc_calib_targets_paralysis
from targets import calc_targets_regional
from targets import calc_targets_simplified_temporal
from targets import calc_targets_temporal_regional_nodes

import laser_polio as lp  # for monkeypatching find_latest_end_of_month when needed


# --------------------------
# Helpers
# --------------------------
def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_yaml(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)


def minimal_model_config(tmp_path: Path):
    """
    Returns a path to a minimal model_config YAML with region_groups:
      - REG_A -> nodes [0, 1]
      - REG_B -> nodes [2, 3]
    """
    cfg = {
        "summary_config": {
            "region_groups": {
                "REG_A": [0, 1],
                "REG_B": [2, 3],
            }
        }
    }
    cfg_path = tmp_path / "model_config.yaml"
    write_yaml(cfg, cfg_path)
    return cfg_path


def tiny_actual_df():
    """
    Create a small 'actual' CSV-like DataFrame with:
      - two regions (REG_A, REG_B)
      - four nodes (0..3) mapping to those regions via 'region' and 'dot_name'
      - dates across multiple months/years
      - P column as observed cases
    """
    data = {
        "date": [
            "2019-01-15",
            "2019-02-15",
            "2020-03-15",
            "2022-04-15",
            "2019-01-15",
            "2020-03-15",
            "2022-04-15",
            "2019-02-15",
        ],
        "node": [0, 0, 1, 1, 2, 2, 3, 3],
        "region": ["REG_A", "REG_A", "REG_A", "REG_A", "REG_B", "REG_B", "REG_B", "REG_B"],
        # dot_name format: AFRO:<adm0>:<adm1>:<whatever>
        "dot_name": [
            "AFRO:NIGERIA:STATE_A:D1",
            "AFRO:NIGERIA:STATE_A:D1",
            "AFRO:NIGERIA:STATE_A:D2",
            "AFRO:NIGERIA:STATE_A:D2",
            "AFRO:NIGERIA:STATE_B:D3",
            "AFRO:NIGERIA:STATE_B:D3",
            "AFRO:NIGERIA:STATE_B:D4",
            "AFRO:NIGERIA:STATE_B:D4",
        ],
        # Observed cases
        "P": [1, 2, 0, 3, 2, 0, 1, 4],
        # Optional: a ready-made "time_period" to use with calc_targets_regional
        "time_period": [
            "2018-2019",
            "2018-2019",
            "2020-2021",
            "2022-2023",
            "2018-2019",
            "2020-2021",
            "2022-2023",
            "2018-2019",
        ],
    }
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def tiny_sim_df():
    """Like tiny_actual_df but uses simulated outcome column new_potentially_paralyzed."""
    df = tiny_actual_df().drop(columns=["P"])
    # Make up some simulated case counts
    df["new_potentially_paralyzed"] = [10, 20, 0, 30, 20, 0, 10, 40]
    return df


# --------------------------
# Tests
# --------------------------


def test_calc_calib_targets(tmp_path):
    """
    calc_calib_targets(filename, model_config_path=None)
    - Uses column 'I'
    - Returns 'total_infected', 'monthly_cases', optionally 'regional_cases' via model_config.summary_config.region_groups
    """
    cfg_path = minimal_model_config(tmp_path)

    # Build a small df with 'I' column that sums predictably:
    df = tiny_actual_df().drop(columns=["P", "time_period"])
    # map P-like numbers into I to reuse structure (just make a simple pattern)
    df["I"] = [5, 0, 2, 3, 1, 0, 4, 0]  # total = 15
    csv_path = tmp_path / "actual_I.csv"
    write_csv(df, csv_path)

    out = calc_calib_targets(filename=csv_path, model_config_path=cfg_path)

    # total_infected
    assert out["total_infected"] == 15

    # monthly_cases: grouped by df['month']
    # months: Jan(1): nodes 0 & 2; Feb(2): 0 & 3; Mar(3): 1 & 2; Apr(4): 1 & 3
    # using I above -> compute expected per month
    df["month"] = df["date"].dt.month
    expected_months = df.groupby("month")["I"].sum().values
    assert np.allclose(out["monthly_cases"], expected_months)

    # regional_cases: REG_A = nodes [0,1], REG_B = nodes [2,3]
    reg_a_sum = df[df["node"].isin([0, 1])]["I"].sum()
    reg_b_sum = df[df["node"].isin([2, 3])]["I"].sum()
    assert np.allclose(out["regional_cases"], np.array([reg_a_sum, reg_b_sum]))


def test_calc_targets_temporal_regional_nodes_actual(tmp_path):
    """
    calc_targets_temporal_regional_nodes(filename, model_config_path=None, is_actual_data=True)
    Should return:
      - total_infected (array of 1)
      - yearly_cases
      - monthly_cases
      - monthly_timeseries
      - adm0_cases, adm01_cases (dicts) when multiple groups exist
      - nodes_with_cases_total / nodes_with_cases_timeseries
    """
    df = tiny_actual_df()
    csv_path = tmp_path / "actual.csv"
    write_csv(df, csv_path)

    out = calc_targets_temporal_regional_nodes(filename=csv_path, model_config_path=None, is_actual_data=True)

    # total infected: sum of P
    assert np.isclose(out["total_infected"][0], df["P"].sum())

    # yearly cases
    exp_yearly = df.groupby(df["date"].dt.year)["P"].sum().values
    assert np.allclose(out["yearly_cases"], exp_yearly)

    # monthly cases (by month number)
    exp_monthly = df.groupby(df["date"].dt.month)["P"].sum().values
    assert np.allclose(out["monthly_cases"], exp_monthly)

    # monthly_timeseries (period M, sorted)
    exp_series = df.groupby(df["date"].dt.to_period("M"))["P"].sum().sort_index().astype(float).values
    assert np.allclose(out["monthly_timeseries"], exp_series)

    # adm0 / adm01 dicts
    # From dot_name: adm0 in col 1, adm1 in col 2
    df2 = df.copy()
    parts = df2["dot_name"].str.split(":", expand=True)
    df2["adm0"] = parts[1]
    df2["adm1"] = parts[2]
    df2["adm01"] = df2["adm0"] + ":" + df2["adm1"]

    exp_adm0 = df2.groupby("adm0")["P"].sum().to_dict()
    exp_adm01 = df2.groupby("adm01")["P"].sum().to_dict()

    if len(exp_adm0) > 1:
        assert out["adm0_cases"] == exp_adm0
    else:
        assert "adm0_cases" not in out  # function only returns multi-region groups

    if len(exp_adm01) > 1:
        assert out["adm01_cases"] == exp_adm01
    else:
        assert "adm01_cases" not in out

    # nodes_with_cases_* (actual path)
    has_case = df["P"] > 0
    exp_nodes_total = df.loc[has_case, "node"].nunique()
    assert np.isclose(out["nodes_with_cases_total"][0], exp_nodes_total)

    df["month_period"] = df["date"].dt.to_period("M")
    months = df["month_period"].sort_values().unique()
    exp_nodes_by_month = df.loc[has_case].groupby("month_period")["node"].nunique().sort_index().reindex(months, fill_value=0).values
    assert np.allclose(out["nodes_with_cases_timeseries"], exp_nodes_by_month)


def test_calc_targets_regional_with_time_periods(tmp_path):
    """
    calc_targets_regional(filename, model_config_path=None, is_actual_data=True)
    Requires 'time_period' and 'region' columns in the CSV when is_actual_data=True.
    """
    df = tiny_actual_df()
    csv_path = tmp_path / "actual_with_period.csv"
    write_csv(df, csv_path)

    # Use defaults for case bins in function (no need for model_config)
    # minimal model config (can be empty or include summary_config; either is fine)
    cfg_path = tmp_path / "model_config.yaml"
    cfg_path.write_text("summary_config: {}\n")  # minimal, valid YAML

    out = calc_targets_regional(filename=csv_path, model_config_path=cfg_path, is_actual_data=True)

    # cases_total
    assert np.isclose(out["cases_total"], df["P"].sum())

    # cases_by_period
    exp_by_period = df.groupby("time_period", observed=True)["P"].sum().to_dict()
    assert out["cases_by_period"] == exp_by_period

    # cases_by_region
    exp_by_region = df.groupby("region")["P"].sum().to_dict()
    assert out["cases_by_region"] == exp_by_region

    # cases_by_region_period (nested)
    grp = df.groupby(["region", "time_period"], observed=True)["P"].sum()
    exp_nested = {}
    for (reg, per), val in grp.items():
        exp_nested.setdefault(reg, {})[str(per)] = val
    assert out["cases_by_region_period"] == exp_nested

    # cases_by_region_month: arrays per region (order: by time index sort)
    # Build expected arrays by region
    monthly = df.groupby(["region", df["date"].dt.to_period("M")])["P"].sum().sort_index().astype(float)
    exp_map = {}
    for reg in df["region"].unique():
        exp_map[reg] = (monthly.loc[reg] if reg in monthly.index.get_level_values(0) else pd.Series(dtype=float)).values
    assert {k: list(v) for k, v in out["cases_by_region_month"].items()} == {k: list(v) for k, v in exp_map.items()}

    # case_bins_by_region exists and sums match number of districts (unique dot_name) per region
    assert "case_bins_by_region" in out
    # verify each region's bins sum equals #dot_names in that region
    counts_per_region = df.groupby("region")["dot_name"].nunique().to_dict()
    for reg, bins in out["case_bins_by_region"].items():
        assert sum(bins) == counts_per_region[reg]


def test_calc_targets_simplified_temporal_actual(tmp_path):
    """
    calc_targets_simplified_temporal(filename, ..., is_actual_data=True)
    - Uses three fixed time bins
    - Returns 'total_by_period', 'monthly_timeseries', 'adm01_by_period'
    """
    df = tiny_actual_df()
    csv_path = tmp_path / "actual_simplified.csv"
    write_csv(df, csv_path)

    out = calc_targets_simplified_temporal(filename=csv_path, model_config_path=None, is_actual_data=True)

    # Recompute the same binning as function
    bins = [pd.Timestamp.min, pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01"), pd.Timestamp.max]
    labels = ["2018-2019", "2020-2021", "2022-2023"]
    temp = df.copy()
    temp["time_period"] = pd.cut(temp["date"], bins=bins, labels=labels, right=False)

    exp_total_by_period = temp.groupby("time_period", observed=True)["P"].sum().to_dict()
    assert out["total_by_period"] == exp_total_by_period

    # monthly_timeseries
    exp_series = temp.groupby(temp["date"].dt.to_period("M"))["P"].sum().sort_index().astype(float).values
    assert np.allclose(out["monthly_timeseries"], exp_series)

    # adm01_by_period
    parts = temp["dot_name"].str.split(":", expand=True)
    temp["adm0"] = parts[1]
    temp["adm1"] = parts[2]
    temp["adm01"] = temp["adm0"] + ":" + temp["adm1"]
    exp_adm01_by_period = temp.groupby(["adm01", "time_period"], observed=True)["P"].sum().to_dict()
    assert out["adm01_by_period"] == exp_adm01_by_period


def test_calc_calib_targets_paralysis_actual_and_sim(tmp_path, monkeypatch):
    """
    calc_calib_targets_paralysis(filename, model_config_path, is_actual_data)
    - Actual path uses 'P'
    - Sim path uses 'new_potentially_paralyzed' scaled by 1/2000 and trims dates to latest end-of-month
    - Also supports summary_config.region_groups
    """
    cfg_path = minimal_model_config(tmp_path)

    # --- Actual path ---
    df_actual = tiny_actual_df()
    csv_actual = tmp_path / "paralysis_actual.csv"
    write_csv(df_actual, csv_actual)

    out_a = calc_calib_targets_paralysis(csv_actual, model_config_path=cfg_path, is_actual_data=True)
    # total_infected equals sum(P)
    assert np.isclose(out_a["total_infected"][0], df_actual["P"].sum())

    # regional_cases (from region_groups nodes)
    # Build node->region group sums using df_actual node membership
    regA_nodes = [0, 1]
    regB_nodes = [2, 3]
    regA_total = df_actual[df_actual["node"].isin(regA_nodes)]["P"].sum()
    regB_total = df_actual[df_actual["node"].isin(regB_nodes)]["P"].sum()
    # calc_calib_targets_paralysis returns regional_cases only if summary_config in model_config
    assert "regional_cases" in out_a
    assert np.allclose(out_a["regional_cases"], np.array([regA_total, regB_total]))

    # --- Simulated path (scale 1/2000) ---
    df_sim = tiny_sim_df()
    # Force find_latest_end_of_month to return the max date (so no rows get dropped)
    monkeypatch.setattr(lp, "find_latest_end_of_month", lambda s: pd.to_datetime(df_sim["date"]).max())
    csv_sim = tmp_path / "paralysis_sim.csv"
    write_csv(df_sim, csv_sim)

    out_s = calc_calib_targets_paralysis(csv_sim, model_config_path=cfg_path, is_actual_data=False)
    # Scaling check on total_infected
    expected_scaled_total = df_sim["new_potentially_paralyzed"].sum() / 2000.0
    assert np.isclose(out_s["total_infected"][0], expected_scaled_total)

    # monthly_cases uses grouped sums (by month) scaled; verify length & non-neg
    assert len(out_s["monthly_cases"]) >= 1
    assert np.all(np.array(out_s["monthly_cases"]) >= 0.0)
