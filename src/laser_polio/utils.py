import csv
import datetime as dt
import json
import os
from zoneinfo import ZoneInfo  # Python 3.9+

import numpy as np
import pandas as pd

__all__ = [
    "calc_r0_scalars_from_rand_eff",
    "calc_sia_prob_from_rand_eff",
    "clean_strings",
    "create_cumulative_deaths",
    "date",
    "daterange",
    "find_matching_dot_names",
    "get_distance_matrix",
    "get_epi_data",
    "get_node_lookup",
    "get_seasonality",
    "get_tot_pop_and_cbr",
    "get_woy",
    "inv_logit",
    "process_sia_schedule_polio",
    "save_results_to_csv",
]


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


def calc_sia_prob_from_rand_eff(sia_re, center=0.5, scale=0.8):
    """Convert SIA random effects to probabilities."""
    vals_rescaled = scale * sia_re + np.log(center / (1 - center))  # Center & scale the random effects (source = Hil???)
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
    revval = revval.replace("’", "")
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


def find_matching_dot_names(patterns, ref_file, verbose=2):
    """
    Finds and returns dot_names from a CSV file that contain the input string patterns.
    For example, if the input string is 'ZAMFARA', the function will return all dot_names
    that contain 'ZAMFARA' in the 'dot_names' column of the CSV file (e.g., 'AFRO:NIGERIA:ZAMFARA:ANKA').

    Parameters:
    patterns (list of str): List of region names to patern match in the 'dot_names' column.
    ref_file (str): Path to the CSV file that contains a 'dot_names' column to serve as a reference of possible dot_name values.

    Returns:
    list of str: A list of matched region names.
    """

    # Load the CSV file
    df = pd.read_csv(ref_file)

    # Ensure the 'dot_names' column exists
    if "dot_name" not in df.columns:
        raise ValueError("The CSV file must contain a 'dot_name' column")

    # Convert input patterns to uppercase
    patterns = [pattern.upper() for pattern in patterns]

    # Filter rows where 'dot_names' contain any of the country names
    matched_regions = df[df["dot_name"].str.contains("|".join(patterns), case=False, na=False)]
    matched_dot_names = np.unique(matched_regions["dot_name"].tolist())  # Find unique dot_names & sort

    # Extract hierarchical levels
    regions = {name.split(":")[0] for name in matched_dot_names}
    adm0 = {":".join(name.split(":")[:2]) for name in matched_dot_names if len(name.split(":")) > 1}
    adm1 = {":".join(name.split(":")[:3]) for name in matched_dot_names if len(name.split(":")) > 2}
    adm2 = set(matched_dot_names)

    # Print summary
    if verbose >= 2:
        print(
            f"The input pattern(s) {patterns} matched dot_names for {len(regions)} region(s), {len(adm0)} admin0, {len(adm1)} admin1, {len(adm2)} admin2 "
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


def process_sia_schedule_polio(df, region_names, sim_start_date):
    """
    Processes an SIA schedule into a dictionary readable by the sim.
     The output file contains a list of the unique SIA dates and corresponding region_name indices included in that campaign.

    Parameters:
    - region_names (list of str): List of full region names (e.g., 'AFRO:NIGERIA:ZAMFARA:ANKA').
    - sim_start_date (str): The beginning date in 'YYYY-MM-DD' format.

    Returns:
    - List of dictionaries in the format:
      [{'date': 'YYYY-MM-DD', 'nodes': [index1, index2, ...]}, ...]
    """

    # Filter dataset to include only matching adm2_name values
    df_filtered = df[df["dot_name"].isin(region_names)].copy()

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


def get_woy(sim):
    time = sim.datevec[sim.t]

    if isinstance(time, dt.date):
        date = time
    else:
        days = int((time - int(time)) * 365.25)
        base_date = pd.to_datetime(f"{int(time)}-01-01")
        datetime = base_date + pd.DateOffset(days=days)
        date = date(datetime)

    woy = date.isocalendar()[1]
    return woy


def get_seasonality(sim):
    woy = get_woy(sim)
    return 1 + sim.pars["seasonal_factor"] * np.cos((2 * np.pi * woy / 52) + sim.pars["seasonal_phase"])


def save_results_to_csv(sim, filename="simulation_results.csv"):
    """
    Save simulation results (S, E, I, R) to a CSV file with columns: Time, Node, S, E, I, R.

    :param sim: The sim object containing a results object with numpy arrays for S, E, I, and R.
    :param filename: The name of the CSV file to save.
    """

    timesteps = sim.t
    datevec = sim.datevec
    nodes = len(sim.nodes)
    results = sim.results

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["timestep", "date", "node", "S", "E", "I", "R", "P"])

        # Write data
        for t in range(timesteps):
            for n in range(nodes):
                writer.writerow(
                    [t, datevec[t], n, results.S[t, n], results.E[t, n], results.I[t, n], results.R[t, n], results.paralyzed[t, n]]
                )

    print(f"Results saved to {filename}")


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
