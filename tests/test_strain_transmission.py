from pathlib import Path
from unittest.mock import patch

import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp


def setup_strain_sim(dur=30, n_ppl=None, init_prev=None, vx_prob_sia=None, sia_schedule=None, r0=14, seed=123):
    """
    Setup simulation for strain transmission testing.

    Args:
        dur: Duration in days
        n_ppl: Population per node
        init_prev: Initial prevalence per node (if None, defaults to zero)
        vx_prob_sia: SIA vaccination probability per node (if None, no SIA)
        sia_schedule: SIA schedule (if None, no SIA)
        r0: Basic reproduction number
        seed: Random seed
    """
    if n_ppl is None:
        n_ppl = np.array([10000, 10000])
    if init_prev is None:
        init_prev = np.array([0.0, 0.0])

    pars = PropertySet(
        {
            "seed": seed,
            "start_date": lp.date("2019-01-01"),
            "dur": dur,
            "n_ppl": n_ppl,
            "cbr": np.array([30, 25]),  # Birth rate per 1000/year
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",
            "init_immun": np.array([0.0, 0.0]),  # No initial immunity to see transmission clearly
            "init_prev": init_prev,  # Initial prevalence (VDPV2 strain by default)
            "r0": r0,
            "r0_scalars": np.array([1.0, 1.0]),  # No spatial scaling
            "seasonal_amplitude": 0.0,  # No seasonality
            "seasonal_peak_doy": 180,
            "distances": np.array([[0, 50], [50, 0]]),  # 50km between nodes
            "gravity_k": 0.5,
            "max_migr_frac": 0.01,
            "sia_schedule": sia_schedule,
            "vx_prob_sia": vx_prob_sia,
            "vx_prob_ri": None,  # No routine immunization
            "stop_if_no_cases": False,
        }
    )

    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM, lp.VitalDynamics_ABM, lp.SIA_ABM]
    return sim


# Test (a): E_by_strain & I_by_strain exist in simulation results
def test_strain_arrays_exist():
    """Test that E_by_strain and I_by_strain arrays exist in simulation results."""
    sim = setup_strain_sim(dur=10, init_prev=np.array([0.001, 0.0]))  # Small outbreak
    sim.run()

    # Check that strain-specific arrays exist
    assert hasattr(sim.results, "E_by_strain"), "E_by_strain array should exist in simulation results"
    assert hasattr(sim.results, "I_by_strain"), "I_by_strain array should exist in simulation results"

    # Check array shapes are correct
    n_time, n_nodes, n_strains = sim.results.I_by_strain.shape
    expected_strains = len(sim.pars.strain_ids)  # Should be 3: VDPV2, Sabin2, nOPV2

    assert n_time == sim.pars.dur + 1, f"Time dimension should be {sim.pars.dur + 1}, got {n_time}"
    assert n_nodes == len(sim.pars.n_ppl), f"Node dimension should be {len(sim.pars.n_ppl)}, got {n_nodes}"
    assert n_strains == expected_strains, f"Strain dimension should be {expected_strains}, got {n_strains}"

    # Check that E_by_strain and I_by_strain have same shape
    assert sim.results.E_by_strain.shape == sim.results.I_by_strain.shape, "E_by_strain and I_by_strain should have the same shape"

    # Check strain indices are consistent with strain_ids
    strain_ids = sim.pars.strain_ids  # {"VDPV2": 0, "Sabin2": 1, "nOPV2": 2}
    assert "VDPV2" in strain_ids and strain_ids["VDPV2"] == 0, "VDPV2 should map to strain index 0"  # noqa: PT018
    assert "Sabin2" in strain_ids and strain_ids["Sabin2"] == 1, "Sabin2 should map to strain index 1"  # noqa: PT018
    assert "nOPV2" in strain_ids and strain_ids["nOPV2"] == 2, "nOPV2 should map to strain index 2"  # noqa: PT018


# Test (b): VDPV infections with no SIA should only produce VDPV2 strains
def test_vdpv_only_transmission():
    """Test that with VDPV2 seeding and no SIA, only VDPV2 strain transmits."""
    sim = setup_strain_sim(
        dur=60,  # Longer to see transmission
        init_prev=np.array([0.01, 0.0]),  # VDPV2 outbreak in node 0
        vx_prob_sia=None,  # No SIA
        sia_schedule=None,
    )
    sim.run()

    # Get strain indices
    vdpv2_idx = sim.pars.strain_ids["VDPV2"]  # Should be 0
    sabin2_idx = sim.pars.strain_ids["Sabin2"]  # Should be 1
    nopv2_idx = sim.pars.strain_ids["nOPV2"]  # Should be 2

    # Check that only VDPV2 strain has exposures and infections
    total_E_vdpv2 = sim.results.E_by_strain[:, :, vdpv2_idx].sum()
    total_I_vdpv2 = sim.results.I_by_strain[:, :, vdpv2_idx].sum()
    total_E_sabin2 = sim.results.E_by_strain[:, :, sabin2_idx].sum()
    total_I_sabin2 = sim.results.I_by_strain[:, :, sabin2_idx].sum()
    total_E_nopv2 = sim.results.E_by_strain[:, :, nopv2_idx].sum()
    total_I_nopv2 = sim.results.I_by_strain[:, :, nopv2_idx].sum()

    assert total_E_vdpv2 > 0, "VDPV2 strain should have exposures with initial VDPV2 seeding"
    assert total_I_vdpv2 > 0, "VDPV2 strain should have infections with initial VDPV2 seeding"
    assert total_E_sabin2 == 0, "Sabin2 strain should have NO exposures without SIA"
    assert total_I_sabin2 == 0, "Sabin2 strain should have NO infections without SIA"
    assert total_E_nopv2 == 0, "nOPV2 strain should have NO exposures without SIA"
    assert total_I_nopv2 == 0, "nOPV2 strain should have NO infections without SIA"

    # Verify that total E and I match VDPV2 only
    total_E_all = sim.results.E.sum()
    total_I_all = sim.results.I.sum()
    assert total_E_all == total_E_vdpv2, "Total exposures should equal VDPV2 exposures only"
    assert total_I_all == total_I_vdpv2, "Total infections should equal VDPV2 infections only"


# Test (c): No VDPV seeding and no SIA should result in no transmission
def test_no_transmission_without_seeding():
    """Test that without initial infections and no SIA, there is no transmission."""
    sim = setup_strain_sim(
        dur=60,
        init_prev=np.array([0.0, 0.0]),  # No initial infections
        vx_prob_sia=None,  # No SIA
        sia_schedule=None,
    )
    sim.run()

    # Check that there are no exposures or infections for any strain
    total_E_all_strains = sim.results.E_by_strain.sum()
    total_I_all_strains = sim.results.I_by_strain.sum()
    total_E = sim.results.E.sum()
    total_I = sim.results.I.sum()

    assert total_E_all_strains == 0, "No exposures should occur without initial seeding or SIA"
    assert total_I_all_strains == 0, "No infections should occur without initial seeding or SIA"
    assert total_E == 0, "Total exposures should be zero without transmission"
    assert total_I == 0, "Total infections should be zero without transmission"

    # Verify strain-specific counts are all zero
    for strain_name, strain_idx in sim.pars.strain_ids.items():
        strain_E = sim.results.E_by_strain[:, :, strain_idx].sum()
        strain_I = sim.results.I_by_strain[:, :, strain_idx].sum()
        assert strain_E == 0, f"{strain_name} strain should have no exposures"
        assert strain_I == 0, f"{strain_name} strain should have no infections"


# Test (d): SIA without VDPV seeding should produce vaccine strain transmission
def test_sia_induced_transmission():
    """Test that SIA without VDPV seeding produces Sabin2 or nOPV2 transmission."""
    # Create SIA schedule for early in simulation
    sia_schedule = [
        {
            "date": "2019-01-05",  # Day 5 of simulation
            "nodes": [0, 1],  # Both nodes
            "age_range": (0, 5 * 365),  # 0-5 years
            "vaccinetype": "nOPV2",  # This should create nOPV2 strain
        }
    ]

    sim = setup_strain_sim(
        dur=60,
        init_prev=np.array([0.0, 0.0]),  # No initial VDPV infections
        vx_prob_sia=np.array([0.8, 0.8]),  # High SIA coverage
        sia_schedule=sia_schedule,
    )
    sim.run()

    # Get strain indices
    vdpv2_idx = sim.pars.strain_ids["VDPV2"]
    sabin2_idx = sim.pars.strain_ids["Sabin2"]
    nopv2_idx = sim.pars.strain_ids["nOPV2"]

    # Check strain-specific counts
    total_E_vdpv2 = sim.results.E_by_strain[:, :, vdpv2_idx].sum()
    total_I_vdpv2 = sim.results.I_by_strain[:, :, vdpv2_idx].sum()
    total_E_sabin2 = sim.results.E_by_strain[:, :, sabin2_idx].sum()
    total_I_sabin2 = sim.results.I_by_strain[:, :, sabin2_idx].sum()
    total_E_nopv2 = sim.results.E_by_strain[:, :, nopv2_idx].sum()
    total_I_nopv2 = sim.results.I_by_strain[:, :, nopv2_idx].sum()

    # Should have NO VDPV2 transmission
    assert total_E_vdpv2 == 0, "VDPV2 strain should have NO exposures without VDPV2 seeding"
    assert total_I_vdpv2 == 0, "VDPV2 strain should have NO infections without VDPV2 seeding"

    # Should have vaccine strain transmission (nOPV2 in this case)
    vaccine_strain_transmission = (total_E_sabin2 + total_E_nopv2 > 0) or (total_I_sabin2 + total_I_nopv2 > 0)
    assert vaccine_strain_transmission, "Should have vaccine strain (Sabin2 or nOPV2) transmission from SIA"

    # For nOPV2 vaccine, should specifically see nOPV2 strain
    # (Note: exact behavior may depend on vaccine strain assignment in the model)
    total_vaccine_E = total_E_sabin2 + total_E_nopv2
    total_vaccine_I = total_I_sabin2 + total_I_nopv2
    assert total_vaccine_E > 0, "Should have vaccine strain exposures from SIA"
    assert total_vaccine_I > 0, "Should have vaccine strain infections from SIA"


# Test SIA with different vaccine types
def test_sia_different_vaccine_types():
    """Test that different SIA vaccine types produce appropriate strains."""
    # Test nOPV2 vaccine
    sia_schedule_nopv2 = [{"date": "2019-01-05", "nodes": [0], "age_range": (0, 5 * 365), "vaccinetype": "nOPV2"}]

    sim_nopv2 = setup_strain_sim(
        dur=30,
        init_prev=np.array([0.0, 0.0]),
        vx_prob_sia=np.array([1.0, 0.0]),  # Full coverage in node 0 only
        sia_schedule=sia_schedule_nopv2,
    )
    sim_nopv2.run()

    # Test mOPV2 vaccine (should produce Sabin2)
    sia_schedule_mopv2 = [{"date": "2019-01-05", "nodes": [0], "age_range": (0, 5 * 365), "vaccinetype": "mOPV2"}]

    sim_mopv2 = setup_strain_sim(dur=30, init_prev=np.array([0.0, 0.0]), vx_prob_sia=np.array([1.0, 0.0]), sia_schedule=sia_schedule_mopv2)
    sim_mopv2.run()

    # Both should produce some vaccine strain transmission
    vdpv2_idx = sim_nopv2.pars.strain_ids["VDPV2"]

    # nOPV2 simulation
    nopv2_has_vaccine_transmission = (
        sim_nopv2.results.E_by_strain[:, :, 1:].sum() > 0  # Sabin2 or nOPV2
        or sim_nopv2.results.I_by_strain[:, :, 1:].sum() > 0
    )
    nopv2_has_vdpv2 = sim_nopv2.results.E_by_strain[:, :, vdpv2_idx].sum() > 0 or sim_nopv2.results.I_by_strain[:, :, vdpv2_idx].sum() > 0

    # mOPV2 simulation
    mopv2_has_vaccine_transmission = (
        sim_mopv2.results.E_by_strain[:, :, 1:].sum() > 0  # Sabin2 or nOPV2
        or sim_mopv2.results.I_by_strain[:, :, 1:].sum() > 0
    )
    mopv2_has_vdpv2 = sim_mopv2.results.E_by_strain[:, :, vdpv2_idx].sum() > 0 or sim_mopv2.results.I_by_strain[:, :, vdpv2_idx].sum() > 0

    assert nopv2_has_vaccine_transmission, "nOPV2 SIA should produce vaccine strain transmission"
    assert not nopv2_has_vdpv2, "nOPV2 SIA should NOT produce VDPV2 transmission"
    assert mopv2_has_vaccine_transmission, "mOPV2 SIA should produce vaccine strain transmission"
    assert not mopv2_has_vdpv2, "mOPV2 SIA should NOT produce VDPV2 transmission"


# Integration test combining multiple scenarios
def test_mixed_strain_transmission():
    """Test scenario with both VDPV seeding AND SIA to verify strain separation."""
    sia_schedule = [
        {
            "date": "2019-01-10",  # Day 10
            "nodes": [1],  # Only node 1 gets SIA
            "age_range": (0, 5 * 365),
            "vaccinetype": "nOPV2",
        }
    ]

    sim = setup_strain_sim(
        dur=60,
        init_prev=np.array([0.01, 0.0]),  # VDPV2 in node 0 only
        vx_prob_sia=np.array([0.0, 0.8]),  # SIA only in node 1
        sia_schedule=sia_schedule,
    )
    sim.run()

    vdpv2_idx = sim.pars.strain_ids["VDPV2"]
    sabin2_idx = sim.pars.strain_ids["Sabin2"]
    nopv2_idx = sim.pars.strain_ids["nOPV2"]

    # Should have VDPV2 transmission (from initial seeding)
    total_E_vdpv2 = sim.results.E_by_strain[:, :, vdpv2_idx].sum()
    total_I_vdpv2 = sim.results.I_by_strain[:, :, vdpv2_idx].sum()

    # Should have vaccine strain transmission (from SIA)
    total_E_vaccine = sim.results.E_by_strain[:, :, sabin2_idx].sum() + sim.results.E_by_strain[:, :, nopv2_idx].sum()  # Sabin2 + nOPV2
    total_I_vaccine = sim.results.I_by_strain[:, :, sabin2_idx].sum() + sim.results.I_by_strain[:, :, nopv2_idx].sum()  # Sabin2 + nOPV2

    assert total_E_vdpv2 > 0, "Should have VDPV2 exposures from initial seeding"
    assert total_I_vdpv2 > 0, "Should have VDPV2 infections from initial seeding"
    assert total_E_vaccine > 0, "Should have vaccine strain exposures from SIA"
    assert total_I_vaccine > 0, "Should have vaccine strain infections from SIA"

    # Total should be sum of all strains
    total_E_all_strains = sim.results.E_by_strain.sum()
    total_I_all_strains = sim.results.I_by_strain.sum()
    total_E_summed = total_E_vdpv2 + total_E_vaccine
    total_I_summed = total_I_vdpv2 + total_I_vaccine

    assert total_E_all_strains == total_E_summed, "Total exposures should equal sum of strain-specific exposures"
    assert total_I_all_strains == total_I_summed, "Total infections should equal sum of strain-specific infections"


@patch("laser_polio.root", Path("tests/"))
def test_realistic_strain_transmission_sokoto():
    """Test realistic strain transmission in SOKOTO with VDPV seeding and SIA campaigns.

    This test simulates a 2-year period in SOKOTO starting in 2020, seeds VDPV2 infections
    in BINJI, and includes SIA campaigns to ensure all 3 strains transmit.
    """
    # Seed schedule to introduce VDPV2 in BINJI early in simulation
    seed_schedule = [
        {
            "date": "2020-01-15",  # Day 15 of simulation
            "dot_name": "AFRO:NIGERIA:SOKOTO:BINJI",
            "prevalence": 200,
        }
    ]

    sim = lp.run_sim(
        regions=["SOKOTO"],
        start_year=2020,
        n_days=365 * 2,  # 2 years as requested
        pop_scale=0.1,  # Small scale for faster testing
        init_region="AFRO:NIGERIA:SOKOTO:BINJI",
        init_prev=0.0,  # No initial prevalence (will use seed_schedule instead)
        results_path=None,
        save_plots=False,
        save_data=False,
        verbose=0,
        r0=12,  # Moderate R0 for transmission
        seed_schedule=seed_schedule,
        vx_prob_ri=0.0,  # No routine immunization to focus on strain dynamics
        stop_if_no_cases=False,  # Don't stop early
        seed=42,  # Fixed seed for reproducibility
    )

    # Verify simulation setup
    assert sim.results.E.shape[0] == 365 * 2 + 1, f"Duration should be {365 * 2 + 1} days, got {sim.pars.dur}"
    assert len(sim.pars.n_ppl) > 20, f"SOKOTO should have many nodes, got {len(sim.pars.n_ppl)}"

    # Get strain indices
    vdpv2_idx = sim.pars.strain_ids["VDPV2"]  # Should be 0
    sabin2_idx = sim.pars.strain_ids["Sabin2"]  # Should be 1
    nopv2_idx = sim.pars.strain_ids["nOPV2"]  # Should be 2

    # Calculate total strain-specific infections over entire simulation
    total_E_vdpv2 = sim.results.E_by_strain[:, :, vdpv2_idx].sum()
    total_I_vdpv2 = sim.results.I_by_strain[:, :, vdpv2_idx].sum()
    total_E_sabin2 = sim.results.E_by_strain[:, :, sabin2_idx].sum()
    total_I_sabin2 = sim.results.I_by_strain[:, :, sabin2_idx].sum()
    total_E_nopv2 = sim.results.E_by_strain[:, :, nopv2_idx].sum()
    total_I_nopv2 = sim.results.I_by_strain[:, :, nopv2_idx].sum()

    # Test that all 3 strains have transmission
    assert total_E_vdpv2 > 0, f"VDPV2 strain should have exposures from seeding. Got {total_E_vdpv2}"
    assert total_I_vdpv2 > 0, f"VDPV2 strain should have infections from seeding. Got {total_I_vdpv2}"
    assert total_E_sabin2 > 0, f"Sabin2 strain should have exposures from mOPV2 SIA. Got {total_E_sabin2}"
    assert total_I_sabin2 > 0, f"Sabin2 strain should have infections from mOPV2 SIA. Got {total_I_sabin2}"
    assert total_E_nopv2 > 0, f"nOPV2 strain should have exposures from nOPV2 SIA. Got {total_E_nopv2}"
    assert total_I_nopv2 > 0, f"nOPV2 strain should have infections from nOPV2 SIA. Got {total_I_nopv2}"

    # Verify consistency between strain-specific and total counts
    total_E_all_strains = total_E_vdpv2 + total_E_sabin2 + total_E_nopv2
    total_I_all_strains = total_I_vdpv2 + total_I_sabin2 + total_I_nopv2
    total_E_sim = sim.results.E.sum()
    total_I_sim = sim.results.I.sum()

    assert total_E_all_strains == total_E_sim, (
        f"Sum of strain-specific exposures ({total_E_all_strains}) should equal total exposures ({total_E_sim})"
    )
    assert total_I_all_strains == total_I_sim, (
        f"Sum of strain-specific infections ({total_I_all_strains}) should equal total infections ({total_I_sim})"
    )

    # Test that VDPV2 appears early (from seeding)
    early_period = slice(0, 30)  # First 30 days
    early_vdpv2_E = sim.results.E_by_strain[early_period, :, vdpv2_idx].sum()
    early_vdpv2_I = sim.results.I_by_strain[early_period, :, vdpv2_idx].sum()

    assert early_vdpv2_E > 0, "VDPV2 exposures should appear early from seeding"
    assert early_vdpv2_I > 0, "VDPV2 infections should appear early from seeding"

    # Test that vaccine strains appear after SIA campaigns
    late_period = slice(180, 731)  # nOPV2 use mostly in second year
    early_sabin_E = sim.results.E_by_strain[early_period, :, sabin2_idx].sum()
    early_nopv2_E = sim.results.E_by_strain[early_period, :, nopv2_idx].sum()
    late_sabin_E = sim.results.E_by_strain[late_period, :, sabin2_idx].sum()
    late_nopv2_E = sim.results.E_by_strain[late_period, :, nopv2_idx].sum()

    # At least one of the periods should have vaccine strain transmission
    assert early_sabin_E == 0, "Sabin2 should not appear prior to SIAs"
    assert early_nopv2_E == 0, "nOPV2 should not appear prior to SIAs"
    assert late_sabin_E > 0, "Sabin2 strain transmission should be detected after SIA campaigns"
    assert late_nopv2_E > 0, "nOPV2 strain transmission should be detected after SIA campaigns"
    assert late_nopv2_E > late_sabin_E, "nOPV2 should have more transmissions than Sabin2 b/c there were more nOPV2 SIAs"


def test_strain_r0_scalars_nopv2_vs_sabin2():
    """Test that nOPV2 has lower transmission than Sabin2 due to strain_r0_scalars.

    This test compares transmission between Sabin2 and nOPV2 strains under identical
    SIA conditions to verify that nOPV2's lower strain_r0_scalars result in fewer
    exposures and infections.
    """
    # Test Sabin2 transmission with mOPV2 SIA
    sia_schedule_sabin = [
        {
            "date": "2019-01-10",  # Day 10 of simulation
            "nodes": [0, 1],  # Both nodes
            "age_range": (0, 5 * 365),  # 0-5 years
            "vaccinetype": "mOPV2",  # This should create Sabin2 strain
        }
    ]

    sim_sabin = setup_strain_sim(
        dur=90,  # 3 months to see transmission
        init_prev=np.array([0.0, 0.0]),  # No initial VDPV infections
        vx_prob_sia=np.array([0.2, 0.2]),  # High SIA coverage for strong signal
        sia_schedule=sia_schedule_sabin,
        r0=5,  # Higher R0 to ensure transmission
        seed=123,
    )
    sim_sabin.run()

    # Test nOPV2 transmission with nOPV2 SIA
    sia_schedule_nopv = [
        {
            "date": "2019-01-10",  # Same timing as Sabin2 test
            "nodes": [0, 1],  # Both nodes
            "age_range": (0, 5 * 365),  # 0-5 years
            "vaccinetype": "nOPV2",  # This should create nOPV2 strain
        }
    ]

    sim_nopv = setup_strain_sim(
        dur=90,  # Same duration
        init_prev=np.array([0.0, 0.0]),  # No initial VDPV infections
        vx_prob_sia=np.array([0.2, 0.2]),  # Same SIA coverage
        sia_schedule=sia_schedule_nopv,
        r0=5,  # Same R0
        seed=123,  # Same random seed for fair comparison
    )
    sim_nopv.run()

    # # Plot the number of new exposures by strain for each sim
    # import matplotlib.pyplot as plt

    # plt.plot(np.sum(sim_sabin.results.new_exposed_by_strain[:, :, sim_sabin.pars.strain_ids["Sabin2"]], axis=1), label="Sabin2")
    # plt.plot(np.sum(sim_nopv.results.new_exposed_by_strain[:, :, sim_nopv.pars.strain_ids["nOPV2"]], axis=1), label="nOPV2")
    # plt.xlabel("Day")
    # plt.ylabel("New Exposures")
    # plt.title("New Exposures by Strain")
    # plt.legend()
    # plt.show()

    # Get strain indices
    sabin2_idx = sim_sabin.pars.strain_ids["Sabin2"]
    nopv2_idx = sim_nopv.pars.strain_ids["nOPV2"]

    # Calculate total exposures and infections for each strain
    total_E_sabin2 = sim_sabin.results.E_by_strain[:, :, sabin2_idx].sum()
    total_I_sabin2 = sim_sabin.results.I_by_strain[:, :, sabin2_idx].sum()
    total_E_nopv2 = sim_nopv.results.E_by_strain[:, :, nopv2_idx].sum()
    total_I_nopv2 = sim_nopv.results.I_by_strain[:, :, nopv2_idx].sum()

    # Verify both strains have some Es and Is (sanity check)
    assert total_E_sabin2 > 0, f"Sabin2 should have exposures from SIA. Got {total_E_sabin2}"
    assert total_I_sabin2 > 0, f"Sabin2 should have infections from SIA. Got {total_I_sabin2}"
    assert total_E_nopv2 > 0, f"nOPV2 should have exposures from SIA. Got {total_E_nopv2}"
    assert total_I_nopv2 > 0, f"nOPV2 should have infections from SIA. Got {total_I_nopv2}"

    # Verify that both strains have Es on the date of the SIA
    sia_day = 9
    sabin_E_sia_day = sim_sabin.results.E_by_strain[sia_day, :, sabin2_idx].sum()
    nopv_E_sia_day = sim_nopv.results.E_by_strain[sia_day, :, nopv2_idx].sum()
    assert sabin_E_sia_day > 0, f"Sabin2 should have exposures on SIA day. Got {sabin_E_sia_day}"
    assert nopv_E_sia_day > 0, f"nOPV2 should have exposures on SIA day. Got {nopv_E_sia_day}"

    # Verify that Sabin2 should have more Es than nOPV2 on the SIA day b/c it has higher vx_efficacy
    assert sabin_E_sia_day > nopv_E_sia_day, (
        f"Sabin2 should have more exposures than nOPV2 on SIA day. Sabin2: {sabin_E_sia_day}, nOPV2: {nopv_E_sia_day}"
    )

    # Test that nOPV2 has lower transmission than Sabin2 due to strain_r0_scalars
    post_sia = slice(16, 90)  # start 5 days after the SIA when primary recipients should mostly have transitioned to the infectious state
    new_sabin_E_post_sia = sim_sabin.results.new_exposed_by_strain[post_sia, :, sabin2_idx].sum()
    new_nopv_E_post_sia = sim_nopv.results.new_exposed_by_strain[post_sia, :, nopv2_idx].sum()
    assert new_nopv_E_post_sia < new_sabin_E_post_sia, (
        f"nOPV2 should have fewer new exposures than Sabin2 after SIA. nOPV2: {new_nopv_E_post_sia}, Sabin2: {new_sabin_E_post_sia}"
    )

    # Verify other strains don't interfere
    vdpv2_idx = sim_sabin.pars.strain_ids["VDPV2"]
    assert sim_sabin.results.E_by_strain[:, :, vdpv2_idx].sum() == 0, "No VDPV2 transmission in Sabin2 test"
    assert sim_sabin.results.E_by_strain[:, :, nopv2_idx].sum() == 0, "No nOPV2 transmission in Sabin2 test"
    assert sim_nopv.results.E_by_strain[:, :, vdpv2_idx].sum() == 0, "No VDPV2 transmission in nOPV2 test"
    assert sim_nopv.results.E_by_strain[:, :, sabin2_idx].sum() == 0, "No Sabin2 transmission in nOPV2 test"


@patch("laser_polio.root", Path("tests/"))
def test_new_exposed_by_strain_results():
    """Test that the new_exposed_by_strain results are correct.

    This test checks that the new_exposed, new_exposed_by_strain, and sia_new_exposed_by_strain results are correct.
    new_exposed_by_strain should contain all new exposures, including those from SIA.
    sia_new_exposed_by_strain should contain only new exposures from SIA.
    new_exposed should contain all new exposures, including those from SIA.
    """

    sim = lp.run_sim(
        regions=["SOKOTO"],
        start_year=2020,
        n_days=365 * 2,  # 2 years as requested
        pop_scale=0.1,  # Small scale for faster testing
        init_region="AFRO:NIGERIA:SOKOTO:BINJI",
        init_prev=200,
        results_path=None,
        save_plots=False,
        save_data=False,
        verbose=0,
        r0=12,  # Moderate R0 for transmission
        vx_prob_ri=0.0,  # No routine immunization to focus on strain dynamics
        stop_if_no_cases=False,  # Don't stop early
        seed=42,  # Fixed seed for reproducibility
    )

    # Get the new_exposed results
    new_exposed = sim.results.new_exposed
    new_exposed_by_strain = sim.results.new_exposed_by_strain
    sia_new_exposed_by_strain = sim.results.sia_new_exposed_by_strain

    # # Plot the number of new exposures by strain for each sim
    # import matplotlib.pyplot as plt

    # plt.plot(np.sum(new_exposed, axis=1), label="Total new exposures")
    # plt.plot(np.sum(new_exposed_by_strain, axis=1).sum(axis=1), label="New exposures by strain", linestyle="--")
    # plt.plot(np.sum(sia_new_exposed_by_strain, axis=1).sum(axis=1), label="SIA new exposures by strain", linestyle="--")
    # plt.xlabel("Day")
    # plt.ylabel("New Exposures")
    # plt.legend()
    # plt.show()

    # Check that new_exposed is the sum of new_exposed_by_strain
    assert np.all(new_exposed == new_exposed_by_strain.sum(axis=2)), "new_exposed should be the sum of new_exposed_by_strain"
    assert np.all(new_exposed >= sia_new_exposed_by_strain.sum(axis=2)), (
        "new_exposed should include transmission and SIA exposures, and therefore be less than or equal to sia_new_exposed_by_strain"
    )


if __name__ == "__main__":
    test_strain_arrays_exist()
    test_vdpv_only_transmission()
    test_no_transmission_without_seeding()
    test_sia_induced_transmission()
    test_sia_different_vaccine_types()
    test_mixed_strain_transmission()
    test_realistic_strain_transmission_sokoto()
    test_strain_r0_scalars_nopv2_vs_sabin2()
    test_new_exposed_by_strain_results()
    print("\n🎉 All strain transmission tests passed!")
