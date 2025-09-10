import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import laser_polio as lp

"""
Test that r0 generates the expected number of infections WITH heterogeneity.
E.g., R0 = 14 should generate ~14 infections.

Assumes:
- No immunity
- WITH heterogeneity
- R0 spatial scalars are all 1.0
- No seasonality
- No births or deaths
- No routine immunization
- No SIAs
"""

# Key assumptions
init_immun_scalar = 0.0
individual_heterogeneity = True
r0_scalar_wt_slope = 0.0  # ensures that r0_scalars = 1.0
r0_scalar_wt_intercept = 0.5  # ensures that r0_scalars = 1.0
seasonal_amplitude = 0.0  # no seasonality
cbr = np.array([0])  # no births or deaths
vx_prob_ri = None  # no routine immunization
vx_prob_sia = None  # no SIA
ipv_vx = False
n_days = 30
dur_inf = lp.constant(value=25)  # Single infection will expire before end of sim
dur_exp = lp.constant(value=60)  # Long exposures ensure that exposed individuals will be in the E state at the end of the simulation
n_reps = 3

# Setting pars
r0 = 14
regions = ["ZAMFARA"]
start_year = 2018
pop_scale = 1 / 1
init_region = "ANKA"
init_prev = 1
migration_method = "radiation"
radiation_k_log10 = -0.3
max_migr_frac = 0.1
verbose = 0
save_plots = False
save_data = False
plot_pars = False
results_path = "results/test_r0_sans_heterogeneity"
init_pop_file = None
use_pim_scalars = True

Es = []
Is = []
risks = []
infectivities = []
for _rep in range(n_reps):
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
        r0=r0,
        migration_method=migration_method,
        radiation_k_log10=radiation_k_log10,
        max_migr_frac=max_migr_frac,
        init_pop_file=init_pop_file,
        use_pim_scalars=use_pim_scalars,
        individual_heterogeneity=individual_heterogeneity,
        init_immun_scalar=init_immun_scalar,
        r0_scalar_wt_slope=r0_scalar_wt_slope,
        r0_scalar_wt_intercept=r0_scalar_wt_intercept,
        seasonal_amplitude=seasonal_amplitude,
        cbr=cbr,
        vx_prob_ri=vx_prob_ri,
        vx_prob_sia=vx_prob_sia,
        ipv_vx=ipv_vx,
        dur_exp=dur_exp,
        dur_inf=dur_inf,
    )

    E = np.sum(sim.results.E_by_strain[:, :, 0], axis=1)  # Filter to VDPV2 strain & sum over nodes
    Es.append(E)
    I = np.sum(sim.results.I_by_strain[:, :, 0], axis=1)  # Filter to VDPV2 strain & sum over nodes
    Is.append(I)

    # Check individual heterogeneity
    risk = sim.people.acq_risk_multiplier
    risks.append(risk)
    infectivity = sim.people.daily_infectivity
    infectivities.append(infectivity)

I_init = np.array([x[0] for x in Is])
I_final = np.array([x[-1] for x in Is])
E_final = np.array([x[-1] for x in Es])
assert np.all(I_init == 1), f"There should be one infection at the start of the simulation, but got {I_init}."
assert np.all(I_final == 0), f"There should be no infections at the end of the simulation, but got {I_final}."
assert np.isclose(np.mean(E_final), 14, atol=7), (
    f"There should be approximately 14 exposures at the end of the simulation, but got {np.mean(E_final)}."
)

# Pool all risk values across reps for more statistical power
pooled_risks = np.concatenate(risks)
pooled_infectivities = np.concatenate(infectivities)

# Plot risk and infectivity
plt.hist(pooled_risks, bins=100, alpha=0.5, label="Risk")
plt.legend()
plt.show()

plt.hist(pooled_infectivities, bins=100, alpha=0.5, label="Infectivity")
plt.show()

# Plot correlation between risk and infectivity
pooled_corr = spearmanr(pooled_risks, pooled_infectivities).correlation
print(f"Correlation between risk and infectivity: {pooled_corr:.3f}")
plt.scatter(pooled_risks, pooled_infectivities, alpha=0.1, s=5)
plt.xlabel("Risk")
plt.ylabel("Infectivity")
plt.title("Correlation between risk and infectivity")
plt.show()
