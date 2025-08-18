import sciris as sc

import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = [
    "BENIN",
    "BURKINA_FASO",
    "COTE_DIVOIRE",
    "GAMBIA",
    "GHANA",
    "GUINEA",
    "GUINEA_BISSAU",
    "LIBERIA",
    "MALI",
    "MAURITANIA",
    "NIGER",
    "NIGERIA",
    "SENEGAL",
    "SIERRA_LEONE",
    "TOGO",
]
admin_level = 0
start_year = 2017
n_days = 2655
pop_scale = 1 / 1
init_region = "BIRINIWA"
init_prev = 200
r0 = 10
migration_method = "radiation"
radiation_k_log10 = -0.3
max_migr_frac = 1.0
vx_prob_ri = 0.0
missed_frac = 0.1
use_pim_scalars = False
results_path = "results/demo_west_africa"
seed_schedule = [
    {"date": "2017-10-01", "dot_name": "AFRO:NIGERIA:JIGAWA:HADEJIA", "prevalence": 100},
    {"date": "2017-10-01", "dot_name": "AFRO:NIGERIA:JIGAWA:GARKI", "prevalence": 100},
    {"date": "2020-07-01", "dot_name": "AFRO:NIGERIA:ZAMFARA:TALATA_MAFARA", "prevalence": 100},
    {"date": "2020-10-01", "dot_name": "AFRO:NIGERIA:NIGER:SULEJA", "prevalence": 100},
]


######### END OF USER PARS ########
###################################

lp.print_memory("Before run_sim")
sim = lp.run_sim(
    regions=regions,
    admin_level=admin_level,
    start_year=start_year,
    n_days=n_days,
    pop_scale=pop_scale,
    init_region=init_region,
    init_prev=init_prev,
    seed_schedule=seed_schedule,
    r0=r0,
    migration_method=migration_method,
    radiation_k_log10=radiation_k_log10,
    max_migr_frac=max_migr_frac,
    vx_prob_ri=vx_prob_ri,
    missed_frac=missed_frac,
    use_pim_scalars=use_pim_scalars,
    results_path=results_path,
    save_plots=True,
    save_data=True,
    verbose=1,
    seed=1,
    save_init_pop=False,
    plot_pars=True,
)
lp.print_memory("After run_sim")

sc.printcyan("Done.")
