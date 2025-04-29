import pandas as pd
import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2018
n_days = 365
pop_scale = 1 / 1
init_region = "ANKA"
init_prev = 0.01
r0 = 14
migration_method = "radiation"
radiation_k = 0.5
max_migr_frac = 1.0
results_path = "results/synth_data_zamfara"

######### END OF USER PARS ########
###################################


sim = lp.run_sim(
    regions=regions,
    start_year=start_year,
    n_days=n_days,
    pop_scale=pop_scale,
    init_region=init_region,
    init_prev=init_prev,
    r0=r0,
    migration_method=migration_method,
    radiation_k=radiation_k,
    max_migr_frac=max_migr_frac,
    results_path=results_path,
    save_plots=True,
    save_data=True,
    verbose=1,
    seed=117,
    stop_if_no_cases=True,
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
    grouped = df.groupby(["dot_name", "month_start"])["P"].sum().reset_index()

    # Rename for clarity
    grouped = grouped.rename(columns={"P": "cases"})

    return grouped


# Extract & summarize the results
df = make_synth_df_from_results(sim)
df["month_start"] = pd.to_datetime(df["month_start"]).astype("datetime64[ns]")
print(df.dtypes)
print(df.head())

# Save as h5
synth_filename = f"{results_path}/synth_data_nigeria_r0{r0}_k{radiation_k}.h5"
df.to_hdf(synth_filename, key="epi", mode="w", format="table")

# Load the saved data to verify
loaded_df = pd.read_hdf(synth_filename, key="epi")
print(loaded_df.dtypes)
print(loaded_df.head())

sc.printcyan("Done.")
