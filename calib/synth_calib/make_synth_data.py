import matplotlib.pyplot as plt
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
                    "new_exposed": results.new_exposed[t, n],
                    "new_potentially_paralyzed": results.new_potentially_paralyzed[t, n],
                    "new_paralyzed": results.new_paralyzed[t, n],
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
    grouped = df.groupby(["dot_name", "month_start"])[["new_potentially_paralyzed", "new_paralyzed"]].sum().reset_index()
    grouped["cases"] = grouped["new_paralyzed"]

    return grouped


# Extract & summarize the results
df = make_synth_df_from_results(sim)
df["month_start"] = pd.to_datetime(df["month_start"]).astype("datetime64[ns]")
print(df.dtypes)
print(df.head())

# Plot the case data over time
plt.figure()
cases_by_month = df.groupby("month_start")["new_paralyzed"].sum()
plt.plot(cases_by_month, label="New Paralyzed")
plt.legend()
plt.title("New Paralyzed Cases by Month")
plt.xlabel("Month")
plt.ylabel("Cases")
plt.grid()
plt.savefig(f"{results_path}/new_paralyzed_cases_by_month.png")
plt.show()

# Save as h5
synth_filename = f"{results_path}/synth_data.h5"
df.to_hdf(synth_filename, key="epi", mode="w", format="table")

# Load the saved data to verify
loaded_df = pd.read_hdf(synth_filename, key="epi")
print(loaded_df.dtypes)
print(loaded_df.head())

sc.printcyan("Done.")
