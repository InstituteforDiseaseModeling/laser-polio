import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["NIGERIA"]
start_year = 2018
n_days = 365 * 3
pop_scale = 1 / 1
init_region = "BIRINIWA"
init_prev = 200
r0 = 14
migration_method = "radiation"
radiation_k = 0.5
max_migr_frac = 1.0
results_path = "results/perf_config"
seed_schedule = [
    {"date": "2018-02-06", "dot_name": "AFRO:NIGERIA:JIGAWA:BIRINIWA", "prevalence": 200},  # day 1
    {"date": "2020-11-24", "dot_name": "AFRO:NIGERIA:ZAMFARA:SHINKAFI", "prevalence": 200},  # day 2
]

best_params = {"r0": 10.645693173058095, "radiation_k": 0.3414061580723608, "seasonal_factor": 0.04574156616477009, "seasonal_phase": 239}

r0 = best_params["r0"]
radiation_k = best_params["radiation_k"]
seasonal_factor = best_params["seasonal_factor"]
seasonal_phase = best_params["seasonal_phase"]

######### END OF USER PARS ########
###################################


sim = lp.run_sim(
    regions=regions,
    start_year=start_year,
    n_days=n_days,
    pop_scale=pop_scale,
    init_region=init_region,
    init_prev=init_prev,
    seed_schedule=seed_schedule,
    r0=r0,
    migration_method=migration_method,
    radiation_k=radiation_k,
    max_migr_frac=max_migr_frac,
    results_path=results_path,
    save_plots=True,
    save_data=True,
    verbose=1,
    seed=1,
    save_pop=True,
    plot_pars=True,
    seasonal_factor=seasonal_factor,
    seasonal_phase=seasonal_phase,
)

sc.printcyan("Done.")
