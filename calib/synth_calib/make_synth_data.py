import numpy as np
import pandas as pd
import sciris as sc
import yaml

import laser_polio as lp

###################################
######### USER PARAMETERS #########

seed = 187
stop_if_no_cases = False
results_path = "calib/synth_calib/results"


######### END OF USER PARS ########
###################################

# Load the synthetic model config. We'll use the same file during calibration to ensure we have the same config.
with open("calib/model_configs/synthetic_model_config.yaml") as f:
    config = yaml.safe_load(f)

# Run the simulation
sim = lp.run_sim(
    seed=seed,
    stop_if_no_cases=stop_if_no_cases,
    results_path=results_path,
    config=config,
)


def make_synth_df_from_results(sim):
    """
    Build a DataFrame from simulation results (S, E, I, R, P, new_exposed).

    :param sim: The sim object containing a results object with numpy arrays for S, E, I, R, etc.
    :return: pandas.DataFrame with columns: timestep, date, node, S, E, I, R, P, new_exposed
    """
    timesteps = sim.nt
    datevec = sim.datevec
    nodes = len(sim.nodes)
    results = sim.results
    node_lookup = sim.pars.node_lookup

    # Prepare list of records
    records = []
    for t in range(timesteps):
        for n in range(nodes):
            dot_name = node_lookup[n]["dot_name"] if n in node_lookup else "Unknown"
            records.append(
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
                    "new_exposed": results.new_exposed[t, n],
                }
            )

    # Create DataFrame
    df = pd.DataFrame.from_records(records)

    # Ensure date column is datetime (if not already)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Create a month_start column (1st of the month)
    df["month_start"] = df["date"].values.astype("datetime64[M]")  # fast way

    # Group by dot_name and month_start, then sum the P column
    grouped = df.groupby(["dot_name", "month_start"])["new_exposed"].sum().reset_index()

    # Divide by 2000 & convert to integers
    grouped["new_exposed"] /= 2000

    cases = []
    for i in range(len(grouped["new_exposed"])):
        expected = grouped["new_exposed"][i]
        paralytic_cases = np.random.poisson(expected)
        cases.append(paralytic_cases)
    grouped["cases"] = cases

    return grouped


# Extract & summarize the results
df = make_synth_df_from_results(sim)
df["month_start"] = pd.to_datetime(df["month_start"]).astype("datetime64[ns]")
print(df.dtypes)
print(df.head())

# Save as h5
synth_filename = f"{results_path}/synth_data.h5"
df.to_hdf(synth_filename, key="epi", mode="w", format="table")

# Load the saved data to verify
loaded_df = pd.read_hdf(synth_filename, key="epi")
print(loaded_df.dtypes)
print(loaded_df.head())

sc.printcyan("Done.")
