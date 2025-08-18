from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["NIGERIA"]
start_year = 2018
n_days = 365
pop_scale = 1 / 1
init_region = "BIRINIWA"
init_prev = 200
r0 = 14
migration_method = "radiation"
radiation_k_log10 = -0.3
max_migr_frac = 1.0
results_path = "results/demo_nigeria"
save_plots = True
save_data = True
save_init_pop = False
run = False

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
    radiation_k_log10=radiation_k_log10,
    max_migr_frac=max_migr_frac,
    results_path=results_path,
    save_plots=save_plots,
    save_data=save_data,
    verbose=1,
    seed=1,
    save_init_pop=save_init_pop,
    run=run,
)

# Check the SIA schedule
sia_schedule = sim.pars["sia_schedule"]
pop = sim.pars["init_pop"]
for _i, instance in enumerate(sia_schedule):
    date = instance["date"]
    vx = instance["vaccinetype"]
    nodes = instance["nodes"]
    pop_in_nodes = pop[nodes].sum() / 1e6  # Convert to millions
    print(f"{date}; vx: {vx}; n_nodes: {len(nodes)}; pop (millions): {pop_in_nodes}")

# Plot SIA coverage
vx_sia_prob = sim.pars["vx_prob_sia"]
avg_sia_prob = np.mean(vx_sia_prob)
print(f"Average SIA coverage: {avg_sia_prob:.2f}")
plt.figure(figsize=(8, 5))
plt.hist(vx_sia_prob, bins=20, edgecolor="black", alpha=0.75)
# Add vertical dashed mean line
plt.axvline(avg_sia_prob, color="red", linestyle="--", linewidth=2, label=f"Mean = {avg_sia_prob:.2f}")
plt.title("Histogram of SIA Coverage for Nigeria")
plt.xlabel("SIA Coverage")
plt.ylabel("Count")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(Path(results_path) / "plot_vx_prob_sia.png")
plt.show()

# Run sim
sim.run()

print("Done.")
