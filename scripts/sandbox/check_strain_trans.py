from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

# Common simulation parameters
regions = ["SOKOTO"]
start_year = 2020
n_days = 365 * 2  # 2 years
pop_scale = 1  # Small scale for faster testing
init_region = "AFRO:NIGERIA:SOKOTO:BINJI"
init_prev = 200  # Seed VDPV2 infections in BINJI
r0 = 14
verbose = 1
save_plots = False
save_data = False
plot_pars = False
seed = 42
results_path = Path("results") / "check_strain_trans"
results_path.mkdir(parents=True, exist_ok=True)

######### END OF USER PARS ########
###################################


def run_scenario(scenario_name, vx_prob_sia=None, strain_r0_scalars=None):
    """Run a single simulation scenario."""
    print(f"\nRunning {scenario_name}...")

    # Base configuration
    config = {
        "regions": regions,
        "start_year": start_year,
        "n_days": n_days,
        "pop_scale": pop_scale,
        "init_region": init_region,
        "init_prev": init_prev,
        "results_path": None,
        "save_plots": save_plots,
        "save_data": save_data,
        "plot_pars": plot_pars,
        "verbose": verbose,
        "seed": seed,
        "r0": r0,
        "vx_prob_ri": 0.0,  # No routine immunization
        "stop_if_no_cases": False,
    }

    # Add SIA parameters if provided
    if vx_prob_sia is not None:
        config["vx_prob_sia"] = vx_prob_sia

    # Add custom strain R0 scalars if provided
    if strain_r0_scalars is not None:
        config["strain_r0_scalars"] = strain_r0_scalars

    sim = lp.run_sim(**config)
    return sim


# Run three scenarios
print("=" * 60)
print("SOKOTO Strain Transmission Comparison")
print("=" * 60)

# Scenario 1: No SIAs (baseline)
vx_prob_zero = np.full(23, 0.0)
sim1 = run_scenario("Scenario 1: No SIAs", vx_prob_sia=vx_prob_zero)

# Scenario 2: With SIAs but no vaccine strain transmission (strain_r0_scalars = 0)
strain_r0_scalars_zero = {"VDPV2": 1.0, "Sabin2": 0.0, "nOPV2": 0.0}
sim2 = run_scenario(
    "Scenario 2: SIAs with strain_r0_scalars = 0",
    strain_r0_scalars=strain_r0_scalars_zero,
)

# Scenario 3: With SIAs and default strain transmission
sim3 = run_scenario("Scenario 3: SIAs with default strain_r0_scalars")

# --- PLOTTING ---
# Create 3x3 plot grid
fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)

# Strain names and simulation data
strain_names = ["VDPV2", "Sabin2", "nOPV2"]
scenario_names = ["No SIAs", "SIAs (OPV R0 scalars = 0)", "SIAs (default OPV R0 scalars)"]
sims = [sim1, sim2, sim3]

# Plot data for each strain and scenario
for strain_idx, strain_name in enumerate(strain_names):
    for sim_idx, (sim, scenario_name) in enumerate(zip(sims, scenario_names, strict=False)):
        ax = axes[strain_idx, sim_idx]

        # Get strain index for this simulation
        if strain_name in sim.pars.strain_ids:
            strain_id = sim.pars.strain_ids[strain_name]

            # Sum exposures by strain across all nodes
            total_exposures = np.sum(sim.results.new_exposed_by_strain[:, :, strain_id], axis=1)
            sia_exposures = np.sum(sim.results.sia_new_exposed_by_strain[:, :, strain_id], axis=1)
            trans_exposures = total_exposures - sia_exposures

            # Plot transmission exposures on primary axis
            ax.plot(trans_exposures, label="Transmission Exposures", linewidth=2, color="green")

            # Create secondary axis for total exposures
            ax2 = ax.twinx()
            ax2.plot(total_exposures, label="Total Exposures", linewidth=2, color="blue", linestyle="--")

            # Add legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            # Calculate total exposures for title
            total_sum_exposures = np.sum(total_exposures)
            ax.set_title(f"{strain_name}\n{scenario_name}\nTotal: {total_sum_exposures:,}", fontsize=10, fontweight="bold")
        else:
            # If strain doesn't exist in this sim, show empty plot
            ax.set_title(f"{strain_name}\n{scenario_name}\nTotal: 0", fontsize=10, fontweight="bold")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=12, alpha=0.5)

        # Formatting
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="y", labelcolor="green")
        ax2.tick_params(axis="y", labelcolor="blue")

        # Only add axis labels on edges
        if strain_idx == 2:  # Bottom row
            ax.set_xlabel("Time (days)")
        if sim_idx == 0:  # Left column
            ax.set_ylabel("Transmission exposures", color="green")
        if sim_idx == 2:  # Right column
            ax2.set_ylabel("Total exposures", color="blue")


# Overall title and layout
fig.suptitle("New Exposures by Strain and Scenario\nSOKOTO 2020-2022", fontsize=16, fontweight="bold")
plt.tight_layout()

# Add text box with scenario descriptions
textstr = """
Scenario 1: VDPV2 seeding only, no SIA campaigns
Scenario 2: VDPV2 seeding + SIA campaigns, but vaccine strains cannot transmit
Scenario 3: VDPV2 seeding + SIA campaigns with normal vaccine strain transmission
"""
fig.text(0.02, 0.02, textstr, fontsize=9, verticalalignment="bottom", bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8})

plt.savefig(Path(results_path) / "plot_strain_trans.png")
plt.show()

# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

for sim_idx, (sim, scenario_name) in enumerate(zip(sims, scenario_names, strict=False)):  # noqa: B007
    print(f"\n{scenario_name}:")
    for strain_name in strain_names:
        if strain_name in sim.pars.strain_ids:
            strain_id = sim.pars.strain_ids[strain_name]
            total_exposures = np.sum(sim.results.new_exposed_by_strain[:, :, strain_id])
            print(f"  {strain_name}: {total_exposures:,} exposures")
        else:
            print(f"  {strain_name}: 0 exposures, 0 infections")

sc.printcyan("Done.")
