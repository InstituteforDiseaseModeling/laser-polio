import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2018
n_days = 365
pop_scale = 1 / 1
init_region = "ANKA"
init_prev = 500
r0 = 14
migration_method = "radiation"
radiation_k = 0.5
max_migr_frac = 1.0
results_path = "results/demo_zamfara"
verbose = 1

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
    save_plots=True,
    save_data=False,
    verbose=verbose,
    seed=1,
    r0=r0,
    migration_method=migration_method,
    radiation_k=radiation_k,
    max_migr_frac=max_migr_frac,
    save_pop=True,
    vx_prob_ri=None,
)

sc.printcyan("Done.")
