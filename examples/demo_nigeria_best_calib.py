import matplotlib.pyplot as plt
import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["NIGERIA"]
start_year = 2018
n_days = 365  # 6 years
pop_scale = 1 / 1
init_region = "ANKA"
migration_method = "radiation"
max_migr_frac = 1.0
results_path = "results/demo_nigeria_best_calib"
vx_prob_ri = None

# From best calib
init_prev = 0
seed_schedule = [
    {"date": "2018-02-06", "dot_name": "AFRO:NIGERIA:JIGAWA:BIRINIWA", "prevalence": 200},
    {"date": "2020-11-24", "dot_name": "AFRO:NIGERIA:ZAMFARA:SHINKAFI", "prevalence": 200},
]
r0 = 18.42868241650029
radiation_k = 0.0956175096168661
seasonal_amplitude = 0.19102804223372347
seasonal_peak_doy = 193

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
    vx_prob_ri=vx_prob_ri,
    migration_method=migration_method,
    radiation_k=radiation_k,
    max_migr_frac=max_migr_frac,
    results_path=results_path,
    save_plots=True,
    save_data=True,
    verbose=1,
    seed=1,
    save_pop=False,
    seed_schedule=seed_schedule,
    seasonal_amplitude=seasonal_amplitude,
    seasonal_peak_doy=seasonal_peak_doy,
)


# --- Setup ---
node_lookup = sim.pars.node_lookup
S = sim.results.S
I = sim.results.I
sia_schedule = sim.pars.sia_schedule
start_date = sim.pars.start_date
datevec = sim.datevec
max_sim_date = max(datevec)

# --- Choose regions to plot ---
regions_to_plot = ["ZAMFARA", "JIGAWA"]


# --- Generate plots ---
for region in regions_to_plot:
    region = region.upper()

    # 1. Get nodes for this region
    region_nodes = [node_id for node_id, info in node_lookup.items() if region in info["dot_name"].upper()]
    if not region_nodes:
        print(f"⚠️ No nodes found for region: {region}")
        continue

    # 2. Get SIA dates targeting this region
    region_sia_dates = [
        campaign["date"]
        for campaign in sia_schedule
        if campaign["date"] <= max_sim_date and any(n in region_nodes for n in campaign["nodes"])
    ]

    # 3. Plot
    plt.figure(figsize=(12, 6))

    # Plot susceptible traces
    for idx in region_nodes:
        label = node_lookup[idx]["dot_name"].split(":")[-1]
        plt.plot(datevec, S[:, idx], label=label, linestyle="-", alpha=0.7)

    # Short vertical ticks for SIA dates
    ymin = 0
    ymax = plt.ylim()[1]
    tick_height = (ymax - ymin) / 20
    for dt in region_sia_dates:
        plt.plot([dt, dt], [ymin, ymin + tick_height], color="red", lw=1.5)

    # Format
    plt.xlabel("Date")
    plt.ylabel("Number of Susceptibles")
    plt.title(f"Susceptible Traces in {region}")
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(bottom=ymin, top=ymax)
    # plt.legend(fontsize='small', ncol=2, loc='upper right')
    # Move legend outside the plot
    plt.legend(fontsize="small", ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Nodes")

    # Adjust layout to make space for legend
    plt.subplots_adjust(right=0.85)  # Shrink plot area

    # 4. Save
    filename = f"{results_path}/susceptibles_{region}.png"
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved plot for {region} to {filename}")

    # --- INFECTED PLOT ---

    plt.figure(figsize=(12, 6))

    # Plot infected traces
    for idx in region_nodes:
        label = node_lookup[idx]["dot_name"].split(":")[-1]
        plt.plot(datevec, I[:, idx], label=label, linestyle="--", alpha=0.7)

    # Short vertical ticks for SIA dates
    ymin = 0
    ymax = plt.ylim()[1]
    tick_height = (ymax - ymin) / 20
    for dt in region_sia_dates:
        plt.plot([dt, dt], [ymin, ymin + tick_height], color="red", lw=1.5)

    # Axis and legend
    plt.xlabel("Date")
    plt.ylabel("Number of Infected")
    plt.title(f"Infected Traces in {region}")
    plt.grid(True)
    plt.ylim(bottom=0, top=ymax)

    # Legend outside
    plt.legend(fontsize="small", ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Nodes")
    plt.subplots_adjust(right=0.78)
    plt.tight_layout()

    # Save plot
    inf_filename = f"{results_path}/infected_{region}.png"
    plt.savefig(inf_filename)
    plt.close()
    print(f"📈 Saved infected plot for {region} to {inf_filename}")


sc.printcyan("Done.")
