import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["NIGERIA"]
start_year = 2018
n_days = 365
pop_scale = 1 / 1
init_region = "BIRINIWA"
init_prev = 0.01
r0 = 60
gravity_k = 0.9
max_migr_frac = 0.9
results_path = "results/synth_data_nigeria"

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
    gravity_k=gravity_k,
    max_migr_frac=max_migr_frac,
    results_path=results_path,
    save_plots=True,
    save_data=True,
    verbose=1,
    seed=117,
    stop_if_no_cases=True,
)

sc.printcyan("Done.")
