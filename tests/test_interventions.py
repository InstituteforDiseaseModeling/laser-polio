import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp


# Fixture to set up the simulation environment
def setup_sim(dur=30, n_ppl=None, vx_prob_ri=0.5, cbr=None, r0=14, new_pars=None):
    if n_ppl is None:
        n_ppl = [50000, 50000]
    if cbr is None:
        cbr = np.array([30, 25])
    pars = PropertySet(
        {
            "dur": dur,
            "n_ppl": n_ppl,
            "cbr": cbr,  # Birth rate per 1000/year
            "init_immun": 0.0,  # initially immune
            "init_prev": 0.0,  # initially infected from any age
            "r0": r0,  # Basic reproduction number
            "dur_exp": lp.constant(value=2),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "vx_prob_ri": vx_prob_ri,  # Routine immunization probability
        }
    )
    pars += new_pars if new_pars is not None else {}
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.RI_ABM, lp.SIA_ABM, lp.Transmission_ABM]
    return sim


# --- RI_ABM Tests ---


def test_ri_initialization():
    """Ensure that RI_ABM initializes correctly."""
    sim = setup_sim()
    sim.run()
    assert hasattr(sim.people, "ri_timer")
    assert hasattr(sim.results, "ri_vaccinated")


def test_ri_manually_seeded():
    """Ensure that routine immunization occurs when manually seeded."""
    n_vx = 1000
    dur = 28
    sim = setup_sim(dur=dur, vx_prob_ri=1.0)
    sim.people.ri_timer[:n_vx] = np.random.randint(0, dur, n_vx)  # Set timers to trigger vaccination
    sim.run()
    assert sim.results.ri_vaccinated.sum() >= n_vx, "The number of vaccinations was lower than the number manually seeded."


def test_ri_on_births():
    dur = 365
    cbr = np.array([300, 250])
    sim = setup_sim(dur=dur, cbr=cbr, vx_prob_ri=1.0)
    sim.run()
    assert np.sum(sim.results.ri_vaccinated) > 0, "No routine immunizations occurred on births."


def test_ri_zero():
    dur = 365

    # Test RI when there are no births (there can still be some RI in existing population)
    cbr = np.array([0, 0])
    vx_prob_ri = 1.0
    sim_no_births = setup_sim(dur=dur, cbr=cbr, vx_prob_ri=vx_prob_ri)
    sim_no_births.run()
    assert np.sum(sim_no_births.results.ri_vaccinated[(98 + 14) :]) == 0, (
        "No RI vaccinations should've occurred after initial cohort aged out of RI (oldest 98 days + time_step)."
    )

    # Zero routine immunization probability
    cbr = np.array([300, 250])
    vx_prob_ri = 0.0
    sim_zero_ri_prob = setup_sim(dur=dur, cbr=cbr, vx_prob_ri=vx_prob_ri)
    sim_zero_ri_prob.run()
    assert np.sum(sim_zero_ri_prob.results.ri_vaccinated) == 0, "RI vaccinations occurred, but there should've been zero."


def test_ri_vx_prob():
    """Ensure that the vaccination probability is respected when no births are scheduled."""
    n_ppl = np.array([50, 50])
    n_vx = np.sum(n_ppl)
    dur = 28
    vx_prob_ri = 0.65
    sim = setup_sim(n_ppl=n_ppl, dur=dur, vx_prob_ri=vx_prob_ri, cbr=np.array([0, 0]))
    sim.people.ri_timer[:n_vx] = np.random.randint(0, dur, n_vx)  # Set timers to trigger vaccination
    sim.run()

    n_exp = n_vx * vx_prob_ri
    n_vx = np.sum(sim.results.ri_vaccinated)
    n_r = np.sum(sim.results.R[-1])

    assert np.isclose(n_exp, n_vx, atol=10), "Vaccination rate does not match probability."
    assert n_vx == n_r, "Vaccinated and Recovered counts should be equal if vx efficacy is 100%"


def test_ri_no_effect_on_non_susceptibles():
    """Ensure RI does not affect infected or recovered individuals."""
    n_ppl = np.array([10, 10])
    r0 = 0
    vx_prob_ri = 1.0
    sim = setup_sim(n_ppl=n_ppl, r0=r0, vx_prob_ri=vx_prob_ri)
    sim.people.ri_timer[:20] = 0
    sim.people.disease_state[:5] = 1  # Exposed
    sim.people.disease_state[5:10] = 2  # Infected
    sim.people.disease_state[10:15] = 3  # Recovered
    sim.run()
    assert np.sum(sim.results.ri_vaccinated) == np.sum(sim.results.R[-1]) == 20, "All individuals should've been vaccinated."


# --- SIA_ABM Tests ---


def test_sia_schedule():
    """Ensure that SIA occurs on the correct date."""
    sia_pars = {
        "sia_schedule": [{"date": "2019-01-10", "nodes": [0], "age_range": (0, 5 * 365), "vaccinetype": "nOPV2"}],
        "vx_prob_sia": [0.6, 0.8],  # SIA effectiveness per node
    }
    sim = setup_sim(vx_prob_ri=0, new_pars=sia_pars)
    sim.run()

    assert hasattr(sim.results, "sia_vaccinated"), "SIA component is missing results array"
    assert hasattr(sim.results, "sia_protected"), "SIA component is missing results array"

    n_vx_day10 = np.sum(sim.results.sia_vaccinated[9, :])
    n_vx_rest = np.sum(sim.results.sia_vaccinated) - n_vx_day10

    assert np.any(n_vx_day10 > 0), "SIA did not execute on the scheduled date."
    assert np.all(n_vx_rest == 0), "SIA should not run on any other dates."
    assert sim.results.sia_vaccinated[9, 1] == 0, "SIA should not have affected individuals in node 1."

    exp_protected = n_vx_day10 * 0.56  # 56% of vaccinated individuals should be protected since using nOPV2
    assert np.isclose(np.sum(sim.results.sia_protected), exp_protected, atol=100), (
        "SIA should protect 56% of vaccinated individuals since using nOPV2."
    )

    # Check if the number vaccinated is close to the expected value
    alive_in_node = (sim.people.node_id == 0) & (sim.people.disease_state >= 0)
    age = sim.t - sim.people.date_of_birth[alive_in_node]  # Age of individuals
    age_eligible = np.sum(age < 5 * 365)  # Filter to <5 years old
    exp_vx = age_eligible * sim.pars.vx_prob_sia[0]  # Expected number of vaccinated individuals
    assert np.isclose(n_vx_day10, exp_vx, atol=100), "Number of vaccinated individuals does not match expected value."

    # Check recovered count
    n_recovered = np.sum(sim.results.R[-1])
    assert n_recovered > 0, "No individuals recovered after SIA."
    assert n_recovered == np.sum(sim.results.sia_protected), "Number of recovered individuals does not match expected value."

    # Check ages of recovereds
    recovereds = sim.people.disease_state == 3
    ages = sim.t - sim.people.date_of_birth[recovereds]
    assert np.all(ages <= (5 * 365 + 22)), "Recovered individuals should be <5 years old."


if __name__ == "__main__":
    test_ri_initialization()
    test_ri_manually_seeded()
    test_ri_on_births()
    test_ri_zero()
    test_ri_vx_prob()
    test_ri_no_effect_on_non_susceptibles()
    test_sia_schedule()

    print("All initialization tests passed.")
