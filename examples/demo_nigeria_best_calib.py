import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["NIGERIA"]
start_year = 2018
n_days = 2190
pop_scale = 1 / 1
init_region = "BIRINIWA"
init_prev = 0
migration_method = "gravity"
max_migr_frac = 1.0
vx_prob_ri = 0.0
missed_frac = 0.0
node_seeding_dispersion = 1.0
use_pim_scalars = False
results_path = "results/demo_nigeria_best_calib"
seed_schedule = [
    {"date": "2018-01-02", "dot_name": "AFRO:NIGERIA:JIGAWA:HADEJIA", "prevalence": 100},  # day 1
    {"date": "2018-01-02", "dot_name": "AFRO:NIGERIA:JIGAWA:GARKI", "prevalence": 100},  # day 1
    {"date": "2020-07-01", "dot_name": "AFRO:NIGERIA:ZAMFARA:TALATA_MAFARA", "prevalence": 100},  # day 2
    {"date": "2020-10-01", "dot_name": "AFRO:NIGERIA:NIGER:SULEJA", "prevalence": 100},  # day 2
]

# Calib pars
seed = 1750269819
r0 = 15.506866736240125
seasonal_amplitude = 0.176162343111089
seasonal_peak_doy = 178
gravity_k_exponent = -12.043699682205505
gravity_a = 0.8490349873341755
gravity_b = 1.7323377362260461
gravity_c = 0.19707497280084302
node_seeding_zero_inflation = 0.8565107799601033
r0_scalar_wt_slope = 63.92688046855615
r0_scalar_wt_intercept = 0.011494954640707458
r0_scalar_wt_center = 0.43106321507924644
sia_re_center = 0.3297917372188261
sia_re_scale = 0.4866729181159698
init_immun_scalar = 1.7383406836075246


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
    migration_method=migration_method,
    max_migr_frac=max_migr_frac,
    vx_prob_ri=vx_prob_ri,
    missed_frac=missed_frac,
    use_pim_scalars=use_pim_scalars,
    r0=r0,
    seasonal_amplitude=seasonal_amplitude,
    seasonal_peak_doy=seasonal_peak_doy,
    gravity_k_exponent=gravity_k_exponent,
    gravity_a=gravity_a,
    gravity_b=gravity_b,
    gravity_c=gravity_c,
    node_seeding_zero_inflation=node_seeding_zero_inflation,
    r0_scalar_wt_slope=r0_scalar_wt_slope,
    r0_scalar_wt_intercept=r0_scalar_wt_intercept,
    r0_scalar_wt_center=r0_scalar_wt_center,
    sia_re_center=sia_re_center,
    sia_re_scale=sia_re_scale,
    init_immun_scalar=init_immun_scalar,
    node_seeding_dispersion=node_seeding_dispersion,
    results_path=results_path,
    save_plots=True,
    save_data=True,
    verbose=1,
    seed=seed,
    save_init_pop=False,
    plot_pars=True,
)

sc.printcyan("Done.")
