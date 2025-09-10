import calendar
import datetime
import datetime as dt
import json
import os
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from time import perf_counter_ns
from zoneinfo import ZoneInfo  # Python 3.9+

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import psutil
import sciris as sc
import yaml

import laser_polio as lp

__all__ = [
    "TimingStats",
    "add_regional_groupings",
    "add_temporal_groupings",
    "calc_r0_scalars_from_rand_eff",
    "calc_sia_prob_from_rand_eff",
    "clean_strings",
    "create_cumulative_deaths",
    "date",
    "daterange",
    "find_latest_end_of_month",
    "find_matching_dot_names",
    "get_distance_matrix",
    "get_doy",
    "get_epi_data",
    "get_node_lookup",
    "get_seasonality",
    "get_tot_pop_and_cbr",
    "inv_logit",
    "load_age_pyramid",
    "make_background_seeding_schedule",
    "pbincount",
    "print_memory",
    "process_sia_schedule_polio",
    "save_sim_results",
    "truncate_colormap",
]


def print_memory(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # In MB
    print(f"[{tag}] Memory usage: {mem:.2f} MB")


def calc_r0_scalars_from_rand_eff(rand_eff=None, R0=14, R_min=3.41, R_max=16.7, emod_scale=2.485, emod_center=-1.050):
    """
    Calculate R0 scalars from regression model random effects. We're going to scale and center this so it's similar
    to  what was done in EMOD. However the spatial pattern will likely be different.
    See workbook in scripts/sandbox/check_reff_random_effects.py
    """

    R_m = R_min / R0  # Min R0. Divide by R0 since we ultimately want scalars on R0 (e.g., r0_scalars)
    R_M = R_max / R0  # Max R0. Divide by R0 since we ultimately want scalars on R0 (e.g., r0_scalars)
    pim_scale = np.std(rand_eff)  # sd from polio immunity mapper (PIM)
    pim_center = np.median(rand_eff)  # median from PIM
    w = inv_logit(emod_scale * (rand_eff - pim_center) / pim_scale)  # Transform the random effects to a 0-1 scale
    R_c = np.exp(emod_center)  # Nigeria central R0 scalar
    reff_scalars = (
        R_c + (R_M - R_c) * np.maximum(w - 0.5, 0) * 2 + (R_c - R_m) * np.minimum(w - 0.5, 0) * 2
    )  # Rescale the random effects to the R0 bounds
    return reff_scalars


def calc_sia_prob_from_rand_eff(sia_re, center=0.5, scale=1.0):
    """Convert SIA random effects to probabilities."""
    vals_rescaled = scale * sia_re + np.log(center / (1 - center))  # Center & scale the random effects
    sia_probs = inv_logit(vals_rescaled)  # Convert to probabilities
    return sia_probs


def clean_strings(revval):
    """Clean up a string by removing diacritics, converting to upper case, and replacing spaces and other non-letter character with underscores."""

    # Upper case
    revval = revval.upper()

    # Diacritics
    revval = revval.replace("Â", "A")
    revval = revval.replace("Á", "A")
    revval = revval.replace("Ç", "C")
    revval = revval.replace("Ê", "E")
    revval = revval.replace("É", "E")
    revval = revval.replace("È", "E")
    revval = revval.replace("Ï", "I")
    revval = revval.replace("Ã¯", "I")
    revval = revval.replace("Í", "I")
    revval = revval.replace("Ñ", "NY")
    revval = revval.replace("Ô", "O")
    revval = revval.replace("Ó", "O")
    revval = revval.replace("Ü", "U")
    revval = revval.replace("Û", "U")
    revval = revval.replace("Ú", "U")

    # Alias characters to underscore
    revval = revval.replace(" ", "_")
    revval = revval.replace("-", "_")
    revval = revval.replace("/", "_")
    revval = revval.replace(",", "_")
    revval = revval.replace("\\", "_")

    # Remove ASCII characters
    revval = revval.replace("'", "")
    revval = revval.replace('"', "")
    revval = revval.replace("'", "")
    revval = revval.replace(".", "")
    revval = revval.replace("(", "")
    revval = revval.replace(")", "")
    revval = revval.replace("\x00", "")

    # Remove non-ASCII characters
    revval = revval.encode("ascii", "replace")
    revval = revval.decode()
    revval = revval.replace("?", "")

    # Condence and strip underscore characters
    while revval.count("__"):
        revval = revval.replace("__", "_")
    revval = revval.strip("_")

    return revval


def date(start_date_str):
    if isinstance(start_date_str, pd.Series):
        return start_date_str.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").replace(tzinfo=ZoneInfo("America/Los_Angeles")).date())
    elif isinstance(start_date_str, dt.date):
        return start_date_str
    else:
        return dt.datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=ZoneInfo("America/Los_Angeles")).date()


def daterange(start_date, days):
    """
    Generate an array of dates from the start date to the end date length in the future.

    Args:
        start_date (datetime.date or str): The start date.
        days (int): The number of days to generate.

    Returns:
        np.ndarray: An array of dates from the start date to the end date.
    """
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("America/Los_Angeles")).date()

    date_range = np.array([start_date + dt.timedelta(days=i) for i in range(days)])
    return date_range


def find_latest_end_of_month(dates=None):
    """
    Return the latest date that is the last day of its month.

    Parameters:
        dates (List[datetime.date]): A list of datetime.date objects.

    Returns:
        datetime.date or None: The most recent end-of-month date, or None if none found.
    """

    def is_end_of_month(d: datetime.date) -> bool:
        return d.day == calendar.monthrange(d.year, d.month)[1]

    end_of_month_dates = [d for d in dates if is_end_of_month(d)]

    return max(end_of_month_dates) if end_of_month_dates else None


def find_matching_dot_names(patterns, ref_file, verbose=2, admin_level=None):
    """
    Finds and returns dot_names from a CSV file that contain the input string patterns.
    For example, if the input string is 'ZAMFARA', the function will return all dot_names
    that contain 'ZAMFARA' in the 'dot_names' column of the CSV file (e.g., 'AFRO:NIGERIA:ZAMFARA:ANKA').

    Parameters:
    patterns (list of str): List of region names to pattern match in the specified column.
    ref_file (str): Path to the CSV file that contains a 'dot_name' column to serve as a reference of possible dot_name values.
    verbose (int): Verbosity level for output messages.
    admin_level (int, optional): Admin level to match against (0, 1, or 2).
                                If None (default), matches against 'dot_name' column.
                                If 0, matches against 'adm0_name' column.
                                If 1, matches against 'adm1_name' column.
                                If 2, matches against 'adm2_name' column.

    Returns:
    list of str: A list of matched region names.
    """

    # Load the CSV file
    df = pd.read_csv(ref_file)

    # Determine which column to match against
    if admin_level is None:
        match_column = "dot_name"
        if "dot_name" not in df.columns:
            raise ValueError("The CSV file must contain a 'dot_name' column")
    elif admin_level == 0:
        match_column = "adm0_name"
        if "adm0_name" not in df.columns:
            raise ValueError("The CSV file must contain an 'adm0_name' column")
    elif admin_level == 1:
        match_column = "adm1_name"
        if "adm1_name" not in df.columns:
            raise ValueError("The CSV file must contain an 'adm1_name' column")
    elif admin_level == 2:
        match_column = "adm2_name"
        if "adm2_name" not in df.columns:
            raise ValueError("The CSV file must contain an 'adm2_name' column")
    else:
        raise ValueError("admin_level must be None, 0, 1, or 2")

    # Convert input patterns to uppercase
    patterns = [pattern.upper() for pattern in patterns]

    if admin_level is None:
        # If no admin level is specified, find ANY matching patterns against dot_name using a regex (e.g., 'MALI' will match any dot_name containing 'MALI' including 'AFRO:SOMALIA:...')
        filtered_df = df[df[match_column].str.contains("|".join(patterns), case=False, na=False)]
    else:
        # If an admin level is specified, match against the specified column using an exact match
        filtered_df = df[df[match_column].isin(patterns)]

    # Filter rows where the specified column contains any of the patterns
    unique_matched_values = filtered_df[match_column].unique()
    matched_dot_names = np.unique(filtered_df["dot_name"].tolist())  # Find unique dot_names & sort

    # Check for unmatched patterns
    unmatched_patterns = []
    for pattern in patterns:
        if pattern not in unique_matched_values:
            unmatched_patterns.append(pattern)
    if unmatched_patterns and admin_level is not None:
        print(f"Warning: The following patterns did not match any {match_column} values: {unmatched_patterns}")
    # Check if the number of unique matched values differs from the number of patterns
    if len(unique_matched_values) != len(patterns) and admin_level is not None:
        print(f"Warning: Found {len(unique_matched_values)} unique {match_column} values but provided {len(patterns)} patterns")
        if verbose >= 2:
            print(f"Matched {match_column} values: {sorted(unique_matched_values)}")
    if verbose >= 2:
        print(f"Matched {match_column} values: {sorted(unique_matched_values)}")

    # Extract hierarchical levels
    regions = {name.split(":")[0] for name in matched_dot_names}
    adm0 = {":".join(name.split(":")[:2]) for name in matched_dot_names if len(name.split(":")) > 1}
    adm1 = {":".join(name.split(":")[:3]) for name in matched_dot_names if len(name.split(":")) > 2}
    adm2 = set(matched_dot_names)

    # Print summary
    if verbose >= 2:
        level_desc = f"admin level {admin_level}" if admin_level is not None else "dot_name"
        print(
            f"The input pattern(s) {patterns} matched against {level_desc} for {len(regions)} region(s), {len(adm0)} admin0, {len(adm1)} admin1, {len(adm2)} admin2 "
        )

    return matched_dot_names


def get_distance_matrix(distance_matrix_path, name_filter):
    """
    Get a subset of the distance matrix for the specified names.

    Args:
        distance_matrix_path (str): Path to the distance matrix CSV file.
        name_filter (list): List of names to match to the row and column names.

    Returns:
        np.ndarray: Subset of the distance matrix as an array.
    """

    # Check the file extension
    file_extension = os.path.splitext(distance_matrix_path)[1]

    # Load the distance matrix based on the file type
    if file_extension == ".csv":
        dist_df = pd.read_csv(distance_matrix_path, index_col=0)
    elif file_extension == ".h5":
        dist_df = pd.read_hdf(distance_matrix_path, key="dist_matrix")
    else:
        raise ValueError("Unsupported file type. Please provide a CSV or HDF5 file.")

    # Filter the DataFrame to include only the specified names
    dist_df_filtered = dist_df.loc[name_filter, name_filter]

    # Convert the filtered DataFrame to a NumPy array
    dist_matrix = dist_df_filtered.values

    return dist_matrix


def get_node_lookup(node_lookup_path, dot_names):
    """
    Load the node_lookup dictionary, filter by dot_names, and assign integer node_ids.

    :param node_lookup_path: Path to the JSON file containing the full node_lookup dictionary.
    :param dot_names: List of dot_names to filter the node_lookup dictionary.
    :return: A dictionary with integer node_ids as keys and filtered node_lookup values.
    """
    # Load the full node_lookup dictionary
    with open(node_lookup_path) as stream:
        full_node_lookup = json.load(stream)

    # Filter by dot_names
    node_lookup = {key: full_node_lookup[key] for key in dot_names}

    # Test ordering
    keys_in_order = list(node_lookup.keys())
    assert np.all(keys_in_order == dot_names), "Node lookup keys do not match or are not in the same order as dot_names."

    # Assign integer node_ids to the node_lookup dictionary
    node_lookup = dict(zip(range(len(dot_names)), node_lookup.values(), strict=False))

    return node_lookup


def get_tot_pop_and_cbr(file_path, regions=None, isos=None, year=None):
    """
    Load and filter the unwpp.csv file to return the tot_pop and cbr columns for specified regions and year.

    Args:
        file_path (str): Path to the unwpp.csv file with population (in thousands) and CBR columns.
        regions (list): List of regions to include.
        year (int): Year to filter the data.

    Returns:
        pd.DataFrame: DataFrame containing the tot_pop and cbr columns for the specified regions and year.
    """
    # Import the unwpp.csv
    df = pd.read_csv(file_path)

    # Filter the data by regions or isos
    if regions is not None:
        df_filtered = df[df["region"].isin(regions)]
        assert len(df_filtered["region"].unique()) == len(regions)
    elif isos is not None:
        df_filtered = df[df["iso"].isin(isos)]
        assert len(df_filtered["iso"].unique()) == len(isos)

    # Filter the data to the specified year
    df_filtered = df_filtered[df_filtered["year"] == year]

    # Return the tot_pop and cbr columns
    pop = (df_filtered["tot_pop"].to_numpy() * 1000).astype(int)  # Round to the nearest integer
    cbr = df_filtered["cbr"].to_numpy()
    return pop, cbr


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


def load_age_pyramid(pyramid_path=None, verbose=False) -> pd.DataFrame:
    """
    Load pyramid CSV and return a DataFrame suitable for merging with immunity and node pop data.

    Output Columns:
        - age_min (int): lower age bound in years
        - age_max (int): upper age bound in years
        - pop (int): total people in that age group (M + F)
    """
    if verbose:
        print(f"Reading population pyramid data from '{pyramid_path}' ...")
    with open(pyramid_path) as f:
        lines = [line.strip() for line in f.readlines()]

    # Validate header
    if lines[0] != "Age,M,F":
        raise ValueError("Expected header line 'Age,M,F'")

    # Parse lines
    text = lines[1:]  # Skip header
    if not all(re.match(r"\d+-\d+,\d+,\d+", line) for line in text[:-1]):
        raise ValueError("Non-final lines must be in 'low-high,m,f' format")
    if not re.match(r"\d+\+,\d+,\d+", text[-1]):
        raise ValueError("Final line must be in 'max+,m,f' format")

    # Tokenize and convert
    parsed = []
    for _i, line in enumerate(text):
        age_part, male, female = line.split(",")
        if "+" in age_part:
            age_min = int(age_part.replace("+", ""))
            age_max = age_min  # one-year bin
        else:
            age_min, age_max = map(int, age_part.split("-"))
        age_max += 1  # Make age_max exclusive (upper bound)
        pop = int(male) + int(female)
        parsed.append([age_min, age_max, pop])

    df = pd.DataFrame(parsed, columns=["age_min", "age_max", "pop"])
    df["pop_frac"] = df["pop"] / df["pop"].sum()
    return df


def make_background_seeding_schedule(
    node_lookup,
    start_date,
    sim_duration,
    prevalence,
    fraction_of_nodes=1.0,
    frequency=30,
    rng=None,
):
    """
    Generate a background seed_schedule using dates and dot_names (not timesteps or node_ids).

    Args:
        node_lookup (dict): Maps node_id → dict with 'dot_name' and other metadata.
        start_date (datetime.date): Simulation start date.
        sim_duration (int): Duration in days.
        prevalence (float): Seeding prevalence per event.
        fraction_of_nodes (float): Fraction of nodes to randomly select.
        frequency (int): Number of days between seedings.
        rng (np.random.Generator): Optional RNG for reproducibility.

    Returns:
        list[dict]: List of {'date', 'dot_name', 'prevalence'} entries.
    """
    if rng is None:
        rng = np.random.default_rng()

    node_ids = list(node_lookup.keys())
    dot_names = [node_lookup[nid]["dot_name"] for nid in node_ids]

    n_seed_nodes = int(np.ceil(len(dot_names) * fraction_of_nodes))
    selected_dot_names = rng.choice(dot_names, size=n_seed_nodes, replace=False)

    schedule = []
    for day in range(0, sim_duration, frequency):
        date = start_date + timedelta(days=day)
        for dot_name in selected_dot_names:
            schedule.append({"date": date, "dot_name": dot_name, "prevalence": prevalence})

    return schedule


def process_sia_schedule(csv_path, start_date):
    """
    Load an SIA schedule from a CSV file and convert real-world dates to simulation timesteps.

    Args:
        csv_path (str): Path to the CSV file containing the SIA schedule.
        start_date (sciris.Date): The start date of the simulation.

    Returns:
        list: A list of SIA events formatted for SIA_ABM.
    """
    # Load the SIA vaccination CSV
    df = pd.read_csv(csv_path)

    # Ensure column names are standardized
    required_columns = {"date", "nodes", "age_min", "age_max", "coverage"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV is missing required columns: {required_columns - set(df.columns)}")

    # Convert dates to timesteps
    df["sim_t_datedelta"] = df["date"].apply(lambda d: date(d) - start_date)
    df["sim_t"] = df["sim_t_datedelta"].dt.days

    # Convert nodes from string (e.g., "0,1,2") to list of integers
    df["nodes"] = df["nodes"].apply(lambda x: list(map(int, str(x).split(","))))

    # Format SIA schedule as a list of dictionaries
    sia_schedule = [
        {
            "date": row["sim_t"],  # Simulation timestep
            "nodes": row["nodes"],  # Targeted nodes
            "age_range": (row["age_min"], row["age_max"]),  # Age range in days
            "coverage": row["coverage"],  # Coverage rate (0-1)
        }
        for _, row in df.iterrows()
    ]

    return sia_schedule


def process_sia_schedule_polio(df, region_names, sim_start_date, n_days, filter_to_type2=True):
    """
    Processes an SIA schedule into a dictionary readable by the sim.
     The output file contains a list of the unique SIA dates and corresponding region_name indices included in that campaign.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the SIA schedule with columns:
    - region_names (list of str): List of full region names (e.g., 'AFRO:NIGERIA:ZAMFARA:ANKA').
    - sim_start_date (str): The beginning date in 'YYYY-MM-DD' format.
    - n_days (int): The number of days to include in the simulation.
    - filter_to_type2 (bool): If True, filter to only type 2 campaigns.

    Returns:
    - List of dictionaries in the format:
      [{'date': 'YYYY-MM-DD', 'nodes': [index1, index2, ...]}, ...]
    """

    # Filter dataset to include only matching adm2_name values
    df_filtered = df[df["dot_name"].isin(region_names)].copy()

    # Filter to only type 2 campaigns if specified
    if filter_to_type2:
        df_filtered = df_filtered[
            df_filtered["vaccinetype"].str.contains("OPV2", case=False, na=False)
            | df_filtered["vaccinetype"].str.contains("topv", case=False, na=False)
        ]

    # Process age data
    df_filtered["age_range"] = df_filtered.apply(lambda row: (row["age_min"], row["age_max"]), axis=1)  # Age range in days

    # Create a dictionary mapping dot_name to index in region_names
    dot_name_to_index = {name: idx for idx, name in enumerate(region_names)}

    # Map dot_name to index values
    df_filtered["node_index"] = df_filtered["dot_name"].apply(lambda x: dot_name_to_index.get(x, -1))

    # Summarize data by date, grouping node indices
    summary = df_filtered.groupby(["date", "age_range", "vaccinetype"])["node_index"].apply(list).reset_index()
    summary.rename(columns={"node_index": "nodes"}, inplace=True)

    # Filter for start dates on or after the simulation beginning date
    summary["date"] = date(summary["date"])
    summary = summary[summary["date"] >= sim_start_date]
    sim_end_date = sim_start_date + datetime.timedelta(days=n_days)
    summary = summary[summary["date"] <= sim_end_date]

    # Curate vaccine strain (the strain that will be transmitted in the sim)
    # Options are Sabin2 or nOPV2
    def assign_vaccine_strain(vaccinetype):
        """Assign vaccine strain based on vaccine type."""
        if "nOPV2" in vaccinetype:
            return "nOPV2"
        elif any(vtype in vaccinetype for vtype in ["mOPV2", "tOPV", "topv"]):
            return "Sabin2"
        else:
            return np.nan

    summary["vaccine_strain"] = summary["vaccinetype"].apply(assign_vaccine_strain)

    # Throw a warning if there are any NA values in the vaccine_strain column
    if summary["vaccine_strain"].isna().any():
        sc.printred(
            "WARNING from process_sia_schedule_polio(): There are NA values in the vaccine_strain column that will cause errors in the sim. Please check the SIA calendar."
        )

    # Convert to dictionary format
    result = summary.to_dict(orient="records")

    return result


def get_epi_data(filename, dot_names, node_lookup, start_year, n_days):
    """Curate the epi data for a specific set of dot names."""

    # Load data
    df = pd.read_hdf(filename, key="epi")

    # Filter by dot_names
    df = df[df["dot_name"].isin(dot_names)]

    # Filter by dates
    df["date"] = pd.to_datetime(df["month_start"])
    start_date = pd.to_datetime(f"{start_year}-01-01")
    end_date = pd.to_datetime(start_date + pd.DateOffset(days=n_days))
    df = df[(df["date"] >= start_date) & (df["date"] < end_date)]

    # Sort by date then node
    df = df.sort_values(by=["date", "dot_name"])
    df = df.reset_index(drop=True)

    # Assign node_ids based on the node_lookup dictionary
    dotname_to_nodeid = {v["dot_name"]: k for k, v in node_lookup.items()}
    df["node"] = df["dot_name"].map(dotname_to_nodeid)

    # Ensure that the nodes are in the same order
    assert np.all(df["dot_name"][0 : len(dot_names)].values == dot_names), "The nodes are not in the same order as the dot_names."

    return df


def get_doy(sim):
    """
    Get the day of year (1-365/366) for the current simulation time.

    Args:
        sim: Simulation object with datevec[t] containing the current date

    Returns:
        int: Day of year (1-based, January 1st is day 1)
    """
    time = sim.datevec[sim.t]
    year = time.year

    # Check if it's a leap year
    is_leap_year = calendar.isleap(year)

    if isinstance(time, dt.date):
        return time.timetuple().tm_yday
    else:
        # Handle floating point time
        days_in_year = 366 if is_leap_year else 365
        days = int((time - int(time)) * days_in_year)
        base_date = pd.to_datetime(f"{year}-01-01")
        datetime = base_date + pd.DateOffset(days=days)
        return datetime.timetuple().tm_yday


def get_seasonality(sim):
    doy = get_doy(sim)

    # Check if it's a leap year
    year = sim.datevec[sim.t].year
    is_leap_year = calendar.isleap(year)
    # Adjust period for leap year (366 days) or normal year (365 days)
    days_in_year = 366 if is_leap_year else 365

    return 1 + sim.pars["seasonal_amplitude"] * np.cos(2 * np.pi * (doy - sim.pars["seasonal_peak_doy"]) / days_in_year)


def _save_sim_results_old(data, filename="output/simulation_results_old.csv", summary_config=None):
    sim = data
    timesteps = sim.nt
    datevec = sim.datevec
    nodes = len(sim.nodes)
    results = sim.results
    node_lookup = sim.pars.node_lookup

    # Create DataFrame from simulation results
    rows = []
    for t in range(timesteps):
        for n in range(nodes):
            dot_name = node_lookup.get(n, {}).get("dot_name", "UNKNOWN")
            rows.append(
                {
                    "timestep": t,
                    "date": datevec[t],
                    "node": n,
                    "dot_name": dot_name,
                    "S": results.S[t, n],
                    "E": results.E[t, n],
                    "I": results.I[t, n],
                    "R": results.R[t, n],
                    "P": results.paralyzed[t, n],
                    "births": results.births[t, n],
                    "deaths": results.deaths[t, n],
                    "new_exposed": results.new_exposed[t, n],
                    "potentially_paralyzed": results.potentially_paralyzed[t, n],
                    "new_potentially_paralyzed": results.new_potentially_paralyzed[t, n],
                    "new_paralyzed": results.new_paralyzed[t, n],
                }
            )

    df = pd.DataFrame(rows)

    # Apply temporal and regional groupings if summary_config provided
    if summary_config is not None:
        # Ensure date column is datetime
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # Apply temporal groupings
        if "time_periods" in summary_config:
            df = add_temporal_groupings(df, summary_config["time_periods"])

        # Apply regional groupings
        if "region_groupings" in summary_config and "grouping_level" in summary_config:
            df = add_regional_groupings(df, summary_config["region_groupings"], grouping_level=summary_config["grouping_level"])
        elif "region_groupings" in summary_config:
            df = add_regional_groupings(df, summary_config["region_groupings"], grouping_level="adm0")
        elif "grouping_level" in summary_config:
            df = add_regional_groupings(df, grouping_level=summary_config["grouping_level"])
        else:
            df = add_regional_groupings(df, grouping_level="dot_name")  # Default to dot_name

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

    return df


def save_sim_results(sim, filename="simulation_results.h5", summary_config=None):
    """
    Save simulation results to a CSV file, optionally applying temporal and regional groupings.
    JUST LEAVING THIS HERE UNTIL RESEARCH IS 100% HAPPY NEW FUNCTION IS VALID :)

    Parameters:
    -----------
    data : sim object
        A sim object containing results arrays
    filename : str
        The name of the CSV file to save
    summary_config : dict, optional
        Configuration for temporal and regional groupings to apply to the data

    Returns:
    --------
    pd.DataFrame
        The processed DataFrame (useful for further analysis)

    Example:
    --------
    # For simulation results:
    save_sim_results(sim, "results.csv", summary_config=config["summary_config"])

    """
    # May want to run this as well as a test to comapre outputs. Then remove.
    # _save_sim_results_old( data=sim, summary_config=summary_config )
    timesteps = sim.nt
    datevec = sim.datevec
    nodes = len(sim.nodes)
    results = sim.results
    node_lookup = sim.pars.node_lookup

    # Precompute size
    total_rows = timesteps * nodes

    # Preallocate arrays
    data = {
        "timestep": np.repeat(np.arange(timesteps), nodes),
        "date": np.repeat(np.asarray(datevec), nodes),
        "node": np.tile(np.arange(nodes), timesteps),
        "dot_name": np.empty(
            total_rows, dtype=object
        ),  # Initialized as object for compatibility; converted to pandas 'category' type later to reduce memory usage
        "S": results.S.flatten(),
        "E": results.E.flatten(),
        "I": results.I.flatten(),
        "R": results.R.flatten(),
        "P": results.paralyzed.flatten(),
        "births": results.births.flatten(),
        "deaths": results.deaths.flatten(),
        "new_exposed": results.new_exposed.flatten(),
        "potentially_paralyzed": results.potentially_paralyzed.flatten(),
        "new_potentially_paralyzed": results.new_potentially_paralyzed.flatten(),
        "new_paralyzed": results.new_paralyzed.flatten(),
    }

    # Vectorized dot_name fill
    dot_name_lookup = np.array([node_lookup.get(n, {}).get("dot_name", "UNKNOWN") for n in range(nodes)], dtype=object)
    data["dot_name"] = np.tile(dot_name_lookup, timesteps)

    df = pd.DataFrame(data)

    # Optional: Convert dot_name to categorical to reduce memory
    df["dot_name"] = df["dot_name"].astype("category")

    # Apply grouping logic if any
    if summary_config is not None:
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        if "time_periods" in summary_config:
            df = add_temporal_groupings(df, summary_config["time_periods"])

        region_groupings = summary_config.get("region_groupings", None)
        grouping_level = summary_config.get("grouping_level", None)

        if region_groupings is not None and grouping_level is not None:
            df = add_regional_groupings(df, region_groupings, grouping_level)
        elif region_groupings is not None:
            df = add_regional_groupings(df, region_groupings, grouping_level="adm0")
        elif grouping_level is not None:
            df = add_regional_groupings(df, grouping_level=grouping_level)
        else:
            # fallback: always apply grouping on dot_name
            df = add_regional_groupings(df, grouping_level="dot_name")
    # Save to HDF5 or CSV
    if Path(filename).suffix == ".h5":
        if "date" in df.columns and df["date"].dtype == "object":
            # Convert to datetime64[ns] if it's a list of date objects
            df["date"] = pd.to_datetime(df["date"])
        df.to_hdf(filename, key="results", mode="w", format="table", complevel=5)
    else:  # just going to assume it's CSV
        df.to_csv(filename, index=False)

    print(f"Results saved to {filename}")
    return df


def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=256):
    base_cmap = plt.get_cmap(cmap_name)
    new_colors = base_cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_trunc_{minval}_{maxval}", new_colors)


def create_cumulative_deaths(total_population, max_age_years):
    """
    Generate a cumulative deaths array with back-loaded mortality.

    Parameters
    ----------
    total_population : int
        Total population size.
    max_age_years : int
        Maximum age in years for the cumulative deaths array.

    Returns
    -------
    cumulative_deaths : np.ndarray
        Cumulative deaths array.
    """
    ages_years = np.arange(max_age_years + 1)
    base_mortality_rate = 0.0001
    growth_factor = 2
    mortality_rates = base_mortality_rate * (growth_factor ** (ages_years / 10))
    cumulative_deaths = np.cumsum(mortality_rates * total_population).astype(int)
    return cumulative_deaths


class TimingStats:
    """
    This is a lightweight hierarchical timing tool that can be used to time code sections (even nested blocks) and log the time spent.
    - .start("Some step") creates a Stopwatch object.
        - The Stopwatch records start_time on entering a `with` (context) block.
        - On exiting the `with` (context) block, it records end_time, computes elapsed time, and calls stats._stop(...) to accumulate stats.
    - .log(logger) formats and prints out timing data nicely.
    """

    def __init__(self):
        """
        Initializes an instance of the class.
        Attributes:
            stats (defaultdict): A dictionary with default integer values to track statistics.
            depth (int): An integer representing the current "indentation" depth, initialized to 0.
        """

        self.stats = defaultdict(int)
        self.depth = 0

        return

    class Stopwatch:
        def __init__(self, key: str, stats):
            """
            Initialize an instance with a key and associated statistics.
            Args:
                key (str): A unique identifier for the instance.
                stats: The TimingStats object accumulating statistics.
            """

            self.key = key
            self.stats = stats

            return

        def __enter__(self):
            """
            Enter the runtime context for the object and start a timer.
            This method is called when the runtime context is entered using the
            'with' statement. It initializes and records the start time in
            nanoseconds for measuring the duration of the context.
            Returns:
                self: The instance of the class, allowing access to its attributes
                and methods within the context.
            """

            self.start_time = perf_counter_ns()

            return self

        def __exit__(self, exc_type, exc_value, traceback):
            """
            Exit the runtime context and perform cleanup actions.
            This method is called when the context manager is exited. It records
            the end time, calculates the elapsed time, and updates the statistics
            with the elapsed duration for the given key.
            Args:
                exc_type (type): The exception type, if an exception was raised.
                exc_value (Exception): The exception instance, if an exception was raised.
                traceback (TracebackType): The traceback object, if an exception was raised.
            Returns:
                None
            """

            self.end_time = perf_counter_ns()
            elapsed = self.end_time - self.start_time
            self.stats._stop(self.key, elapsed)

            return

    def start(self, label):
        """
        Starts a new timing session with the given label.
        This method creates a new key based on the current depth and the provided label,
        increments the depth, and initializes a new Stopwatch instance associated with
        the key. The key is also added to the stats dictionary with an initial value of 0.
        Args:
            label (str): A descriptive label for the timing session.
        Returns:
            TimingStats.Stopwatch: An instance of the Stopwatch class associated with the
            given label and timing session.
        """

        key = (" " * (4 * self.depth)) + label
        self.depth += 1
        sw = TimingStats.Stopwatch(key, self)
        self.stats[key] += 0

        return sw

    def _stop(self, key, elapsed):
        """
        Stops the timer for a given key, updates the elapsed time in the stats,
        and decreases the depth counter.
        Args:
            key (str): The identifier for the timer being stopped.
            elapsed (float): The elapsed time to add to the stats for the given key.
        Returns:
            None
        """

        self.stats[key] += elapsed
        self.depth -= 1

        return

    def log(self, logger):
        """
        Logs the elapsed time statistics stored in the `self.stats` dictionary.
        Each entry in `self.stats` is formatted and logged using the provided logger.
        The elapsed time is converted from nanoseconds to microseconds and rounded
        before being logged.
        Args:
            logger (logging.Logger): The logger instance used to log the formatted
                elapsed time statistics.
        Returns:
            None
        """

        width = max(map(len, self.stats.keys()))
        fmt = f"{{label:<{width}}} : {{value:11,}} µsecs"

        for label, elapsed in self.stats.items():
            logger.info(fmt.format(label=label, value=round(elapsed / 1000)))

        return


def pbincount(bins, num_bins, weights=None, dtype=None):
    """
    Compute the histogram of a set of data in parallel, similar to `numpy.bincount`.

    This function is a parallelized version of `numpy.bincount`, which counts the
    occurrences of integers in an array. It supports optional weighting of the counts
    and allows specifying the data type of the output.

    Parameters:
    ----------
    bins : array-like
        An array of non-negative integers to be counted. Each value in `bins` represents
        an index in the histogram.
    num_bins : int
        The number of bins (size of the histogram). This determines the length of the
        output array.
    weights : array-like, optional
        An array of weights, of the same shape as `bins`. Each value in `weights` is
        added to the corresponding bin instead of incrementing by 1. If `None`, each
        bin is incremented by 1 for each occurrence in `bins`. Default is `None`.
    dtype : data-type, optional
        The desired data type for the output array. If not specified, the data type is
        inferred from `bins` if `weights` is `None`, or from `weights` otherwise.

    Returns:
    -------
    numpy.ndarray
        A 1D array of length `num_bins` containing the counts or weighted sums for
        each bin. The counts are computed in parallel for improved performance.

    Notes:
    -----
    - This function uses thread-local storage to compute the counts in parallel, and
      then aggregates the results across threads.
    - The input `bins` must contain non-negative integers, as they are used as indices
      in the histogram.
    - If `weights` is provided, it must have the same shape as `bins`.

    Example:
    -------
    >>> import numpy as np
    >>> bins = np.array([0, 1, 1, 2, 2, 2])
    >>> num_bins = 4
    >>> pbincount(bins, num_bins)
    array([1, 2, 3, 0])

    >>> weights = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    >>> pbincount(bins, num_bins, weights=weights)
    array([0.5, 2.5, 7.5, 0.0])
    """

    num_indices = len(bins)
    return_type = dtype or (bins.dtype if weights is None else weights.dtype)
    tls = np.zeros((nb.get_num_threads(), num_bins), dtype=return_type)

    if weights is None:
        nb_bincount(bins, num_indices, tls)
    else:
        nb_bincount_weighted(bins, num_indices, weights, tls)

    return tls.sum(axis=0)


# Perform a parallel bincount operation using Numba.
# Args:
#     bins (np.ndarray): Array of bin indices.
#     num_indices (int): Number of elements in the bins array.
#     tls (np.ndarray): Thread-local storage array for counting.
# Returns:
#     None: The results are stored in the tls array.
@nb.njit(parallel=True, cache=True)
def nb_bincount(bins, num_indices, tls):
    for i in nb.prange(num_indices):
        tls[nb.get_thread_id(), bins[i]] += 1
    return


# Perform a parallel, weighted bincount operation using Numba.
# Args:
#     bins (np.ndarray): Array of bin indices.
#     num_indices (int): Number of elements in the bins array.
#     weights (np.ndarray): Array of weights corresponding to the bins.
#     tls (np.ndarray): Thread-local storage array for counting.
# Returns:
#     None: The results are stored in the tls array.
@nb.njit(parallel=True, cache=True)
def nb_bincount_weighted(bins, num_indices, weights, tls):
    for i in nb.prange(num_indices):
        tls[nb.get_thread_id(), bins[i]] += weights[i]
    return


def add_temporal_groupings(df, time_config):
    """
    Add time period columns to a DataFrame based on summary_config time configuration.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing simulation results with a 'date' column
    time_config : dict
        Configuration dictionary with 'bins' and 'labels' keys

    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'time_period' column

    Example:
    --------
    time_config = {
        "bins": ["2020-07-01", "2022-07-01"],
        "labels": ["2018-2020.5", "2020.5-2022.5", "2022.5-2024"]
    }
    df = add_temporal_groupings(df, time_config)
    """
    df = df.copy()

    if "bins" in time_config and "labels" in time_config:
        bin_dates = [pd.Timestamp(d) for d in time_config["bins"]]
        # Add min/max bounds
        bins = [pd.Timestamp.min, *bin_dates, pd.Timestamp.max]
        labels = time_config["labels"]
        df["time_period"] = pd.cut(df["date"], bins=bins, labels=labels, right=False)

    return df


def add_regional_groupings(df, region_groupings=None, grouping_level="adm0", regions_yaml_path=None):
    """
    Add regional groupings based on region_groupings list.
    Countries in the list use regions.yaml, all others use adm0.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing simulation results with 'dot_name' column
    region_groupings : list
        List of countries that need custom regions from regions.yaml
    regions_yaml_path : str, optional
        Path to regions.yaml file. If None, uses default path at data/regions.yaml

    Returns:
    --------
    pd.DataFrame
        DataFrame with added administrative level columns (adm0, adm1, adm01) and 'region' column

    Example:
    --------
    # In YAML config:
    # region_groupings: ["NIGERIA", "BENIN"]

    df = add_regional_groupings(df, ["NIGERIA", "BENIN"])
    """

    df = df.copy()

    # Split dot_name into administrative level columns
    dot_parts = df["dot_name"].str.split(":", expand=True)
    df["adm0"] = dot_parts[1]
    df["adm1"] = dot_parts[2]
    df["adm01"] = df["adm0"] + ":" + df["adm1"]

    # Set default grouping based on specified admin level
    if grouping_level not in {"adm0", "adm01", "dot_name"}:
        raise ValueError(f"Invalid grouping_level: {grouping_level}. Must be one of 'adm0', 'adm01', or 'dot_name'.")
    df["region"] = df[grouping_level]

    # Apply custom regions for specified countries if provided
    if region_groupings:
        # Load regions.yaml for custom regions
        regions_yaml_path = regions_yaml_path or lp.root / "data/regions.yaml"
        try:
            with open(regions_yaml_path) as f:
                regions_data = yaml.safe_load(f)

            # Apply custom regions for specified countries
            for country in region_groupings:
                country_upper = country.upper()

                if country_upper in regions_data:
                    # Get mask for this country's data
                    country_mask = df["adm0"] == country_upper
                    region_groups = regions_data[country_upper]

                    # Apply custom region groups for this country
                    for group_name, patterns in region_groups.items():
                        pattern_mask = pd.Series(False, index=df.index)

                        # for pattern in patterns:
                        #     if ":" in pattern and not pattern.startswith("AFRO:"):
                        #         # Pattern like "NIGERIA:JIGAWA" - match against adm01
                        #         pattern_mask |= df["adm01"] == pattern
                        #     else:
                        #         # Pattern for dot_name matching
                        #         pattern_mask |= df["dot_name"].str.contains(pattern, case=False, na=False)
                        for pattern in patterns:
                            if grouping_level == "adm01" and ":" in pattern:
                                pattern_mask |= df["adm01"] == pattern
                            elif grouping_level == "adm1":
                                pattern_mask |= df["adm1"] == pattern
                            else:
                                pattern_mask |= df["dot_name"].str.contains(pattern, case=False, na=False)

                        # Apply to this country only
                        final_mask = country_mask & pattern_mask
                        df.loc[final_mask, "region"] = group_name
                else:
                    print(f"Warning: Custom regions not found for {country} in regions.yaml, using adm0")

        except (FileNotFoundError, ImportError):
            print(f"Warning: Could not load {regions_yaml_path}, using adm0 for all countries")

    return df


# warm up Numba
def __warmup_numba():
    """
    Warm up the Numba JIT compiler by executing a series of operations
    involving the `pbincount` function with various input and output data types.
    This function performs the following steps:
    1. Generates random integer bins and random floating-point weights.
    2. Calls `pbincount` with different combinations of integer input and output types.
    3. Calls `pbincount` with different combinations of floating-point input and output types,
       including weighted bin counting.
    The purpose of this function is to ensure that the Numba JIT compiler has precompiled
    the necessary code paths for the `pbincount` function, reducing runtime overhead
    during subsequent calls.
    Note:
        This function is intended for internal use only and does not return any value.
    """

    _bins = np.random.randint(0, 16, size=1_000)
    for in_type, out_type in [(np.int32, np.int32), (np.int32, np.int64), (np.int64, np.int32), (np.int64, np.int64)]:
        _ = pbincount(_bins.astype(in_type), num_bins=16, dtype=out_type)
    _wf64 = np.random.rand(1_000)
    for in_type, out_type in [(np.float32, np.float32), (np.float32, np.float64), (np.float64, np.float32), (np.float64, np.float64)]:
        _ = pbincount(_bins, num_bins=16, weights=_wf64.astype(in_type), dtype=out_type)

    return


__warmup_numba()
