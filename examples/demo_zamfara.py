import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2018
n_days = 365
pop_scale = 1 / 1
init_region = "ANKA"
init_prev = 200
r0 = 14
migration_method = "radiation"
radiation_k = 0.5
max_migr_frac = 1.0
verbose = 1
vx_prob_ri = None
missed_frac = 0.1
seed_schedule = [
    {"date": "2018-01-02", "dot_name": "AFRO:NIGERIA:ZAMFARA:BAKURA", "prevalence": 200},  # day 1
    {"date": "2018-11-07", "dot_name": "AFRO:NIGERIA:ZAMFARA:GUMMI", "prevalence": 200},  # day 2
]
save_plots = True
save_data = False
plot_pars = True
seed = 1
# Diffs from demo_zamfara_load_init_pop.py
results_path = "results/demo_zamfara"
save_pop = True
init_pop_file = None


######### END OF USER PARS ########
###################################


sim = lp.run_sim(
    regions=regions,
    start_year=start_year,
    n_days=n_days,
    pop_scale=pop_scale,
    init_region=init_region,
    init_prev=init_prev,
    results_path=results_path,
    save_plots=save_plots,
    save_data=save_data,
    plot_pars=plot_pars,
    verbose=verbose,
    seed=seed,
    r0=r0,
    migration_method=migration_method,
    radiation_k=radiation_k,
    max_migr_frac=max_migr_frac,
    save_pop=save_pop,
    vx_prob_ri=vx_prob_ri,
    init_pop_file=init_pop_file,
    seed_schedule=seed_schedule,
    missed_frac=missed_frac,
)

sc.printcyan("Done.")
