import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2019
n_days = 365
pop_scale = 1 / 1
init_region = "ANKA"
init_prev = 0.01
r0 = 14 * 3
gravity_k = 1000.0
results_path = "results/demo_zamfara"

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
    verbose=3,
    seed=1,
    r0=r0,
    gravity_k=gravity_k,
)

sc.printcyan("Done.")
