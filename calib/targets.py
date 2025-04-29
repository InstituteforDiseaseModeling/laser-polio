import numpy as np
import pandas as pd
import yaml

import laser_polio as lp


def calc_calib_targets_paralysis(filename, model_config_path=None, is_actual_data=True):
    """Load simulation results and extract features for comparison."""

    # Load the data & config
    df = pd.read_csv(filename)
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Choose the column to summarize
    if is_actual_data:
        case_col = "P"
        scale_factor = 1.0
    else:
        case_col = "new_exposed"
        scale_factor = 1 / 2000.0
        # The actual data is in months & the sim has a tendency to rap into the next year (e.g., 2020-01-01) so we need to exclude and dates beyond the last month of the actual data
        max_date = lp.find_latest_end_of_month(df["date"])
        df = df[df["date"] <= max_date]

    targets = {}

    # 1. Total infected (scaled if simulated)
    targets["total_infected"] = np.array([df[case_col].sum() * scale_factor])

    # 2. Yearly cases
    targets["yearly_cases"] = df.groupby("year")[case_col].sum().values * scale_factor

    # 3. Monthly cases
    targets["monthly_cases"] = df.groupby("month")[case_col].sum().values * scale_factor

    # 4. Regional group cases
    if model_config and "summary_config" in model_config:
        region_groups = model_config["summary_config"].get("region_groups", {})
        regional_cases = []
        for name in region_groups:
            node_list = region_groups[name]
            total = df[df["node"].isin(node_list)][case_col].sum() * scale_factor
            regional_cases.append(total)
        targets["regional_cases"] = np.array(regional_cases)

    print(f"{targets=}")
    return targets


def calc_calib_targets(filename, model_config_path=None):
    """Load simulation results and extract features for comparison."""

    # Load the data & config
    df = pd.read_csv(filename)
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    targets = {}

    # 1. Total infected
    targets["total_infected"] = df["I"].sum()

    # 2. Yearly cases

    # 3. Monthly cases
    targets["monthly_cases"] = df.groupby("month")["I"].sum().values

    # 4. Regional group cases as a single array
    if model_config and "summary_config" in model_config:
        region_groups = model_config["summary_config"].get("region_groups", {})
        regional_cases = []
        for name in region_groups:
            node_list = region_groups[name]
            total = df[df["node"].isin(node_list)]["I"].sum()
            regional_cases.append(total)
        targets["regional_cases"] = np.array(regional_cases)

    print(f"{targets=}")
    return targets


def process_data(filename):
    """Load simulation results and extract features for comparison."""
    df = pd.read_csv(filename)
    return {
        "total_infected": df["I"].sum(),
        "peak_infection_time": df.loc[df["I"].idxmax(), "Time"],
    }
