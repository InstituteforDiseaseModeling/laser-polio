import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["NIGERIA"]
start_year = 2019
n_days = 365
pop_scale = 1 / 100
init_region = "PLATEAU"
init_prev = 0.01
r0 = 14
results_path = "results/demo_nigeria"

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
    results_path=results_path,
    save_plots=True,
    save_data=False,
    verbose=1,
    seed=1,
)

sc.printcyan("Done.")
