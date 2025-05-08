import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["NIGERIA"]
start_year = 2018
n_days = 2190  # 6 years
pop_scale = 1 / 1
init_region = "BIRINIWA"
init_prev = 200
r0 = 14
migration_method = "radiation"
radiation_k = 0.5
max_migr_frac = 1.0
results_path = "results/demo_nigeria"
save_plots = True
save_data = True
save_pop = False
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
    radiation_k=radiation_k,
    max_migr_frac=max_migr_frac,
    results_path=results_path,
    save_plots=save_plots,
    save_data=save_data,
    verbose=1,
    seed=1,
    save_pop=save_pop,
    run=run,
)


sia_schedule = sim.pars["sia_schedule"]
pop = sim.pars["n_ppl"]

for _i, instance in enumerate(sia_schedule):
    date = instance["date"]
    vx = instance["vaccinetype"]
    nodes = instance["nodes"]
    pop_in_nodes = pop[nodes].sum() / 1e6  # Convert to millions
    print(f"{date}; vx: {vx}; n_nodes: {len(nodes)}; pop (millions): {pop_in_nodes}")

sim.run()

print("Done.")
