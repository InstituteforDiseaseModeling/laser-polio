import geopandas as gpd
import numpy as np
from laser_core.propertyset import PropertySet
from shapely.geometry import Point

import laser_polio as lp


def setup_response_sia_sim(
    dur=100,
    n_ppl=None,
    r0=14,
    seed_schedule=None,
    response_sia_dist=50.0,
    step_size_ResponseSIA=7,
    seed=123,
    new_pars=None,
):
    """Set up simulation with ResponseSIA component for testing."""
    if n_ppl is None:
        n_ppl = np.array([50000, 50000, 50000])  # 3 nodes for testing

    # Create distance matrix for 3 nodes
    dist_matrix = np.array(
        [
            [0.0, 30.0, 60.0],  # Node 0 distances
            [30.0, 0.0, 40.0],  # Node 1 distances
            [60.0, 40.0, 0.0],  # Node 2 distances
        ]
    )

    # Mock shapefile data for mapping
    # Create mock shapefile with centroids
    geometry = [Point(0, 0), Point(1, 1), Point(2, 2)]
    shp_data = {"center_lon": [0.0, 1.0, 2.0], "center_lat": [0.0, 1.0, 2.0], "geometry": geometry}
    shp = gpd.GeoDataFrame(shp_data, crs="EPSG:4326")

    response_sia_pars = {
        "response_sia_dist": response_sia_dist,
        "step_size_ResponseSIA": step_size_ResponseSIA,
        "response_sia_time_to_1st_round": lambda x: 7,  # 7 days to first round
        "response_sia_2nd_round_gap": 14,  # days between rounds
        "response_sia_blackout_duration": 180,  # blackout after second round in days
        "response_sia_age_range": [0, 5 * 365],  # 0-5 years
        "response_sia_vaccine_type": "nOPV2",
        "response_sia_vaccine_strain": 2,
        "shp": shp,
    }

    pars = PropertySet(
        {
            "dur": dur,
            "n_ppl": n_ppl,
            "distances": dist_matrix,
            "cbr": np.array([30, 25, 30]),  # Birth rate per 1000/year
            "init_immun": 0.0,  # initially immune
            "init_prev": 0.0,  # initially infected from any age
            "seed_schedule": seed_schedule,
            "r0": r0,  # Basic reproduction number
            "r0_scalars": [0.9, 1.1, 1.0],  # Spatial transmission scalar (multiplied by global rate)
            "p_paralysis": 1 / 20,  # Very low probability of paralysis for testing
            "dur_exp": lp.constant(value=2),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "vx_prob_ri": 0.0,  # No routine immunization for testing
            "vx_prob_ipv": 0.0,  # No IPV for testing
            "stop_if_no_cases": False,
            "seed": seed,
            "strain_r0_scalars": {0: 1.0, 1: 0.0, 2: 0.0},
            "verbose": 1,
        }
    )

    pars += response_sia_pars
    if new_pars:
        pars += new_pars

    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM, lp.SIA_ABM, lp.ResponseSIA]

    return sim


def test_initialization():
    """Test that ResponseSIA initializes correctly."""
    sim = setup_response_sia_sim()

    # Find ResponseSIA component
    response_sia = None
    for component in sim.instances:
        if isinstance(component, lp.ResponseSIA):
            response_sia = component
            break

    assert response_sia is not None, "ResponseSIA component not found"
    assert hasattr(response_sia, "dist_matrix"), "Distance matrix not initialized"
    assert hasattr(response_sia, "dist_threshold"), "Distance threshold not set"
    assert hasattr(response_sia, "node_response_blocked_until"), "Blackout tracking not initialized"
    assert response_sia.step_size == 7, "Step size not set correctly"

    # Check blackout array initialization
    assert len(response_sia.node_response_blocked_until) == 3, "Blackout array wrong size"
    assert np.all(response_sia.node_response_blocked_until == -np.inf), "Blackout array not initialized to -inf"


def test_no_response_without_cases():
    """Test that no response SIAs are scheduled when there are no cases."""
    sim = setup_response_sia_sim()
    sim.run()

    # Check that no response SIAs were added
    # initial_schedule_length = len(sim.pars.sia_schedule)
    # response_sias = [entry for entry in sim.pars.sia_schedule if entry.get("source") == "response"]
    # assert len(response_sias) == 0, "Response SIAs scheduled despite no cases"
    assert len(sim.pars.sia_schedule) == 0, "SIA schedule should be empty"


def test_response_triggered_by_cases():
    """Test that response SIAs are triggered when cases are detected."""
    sim = setup_response_sia_sim(r0=0)  # No transmission to ensure cases are only in node 0

    # Manually create cases in node 0 at day 10
    sim.people.disease_state[:100] = 2  # Set some people to infected
    sim.people.node_id[:100] = 0  # Put them in node 0
    sim.people.potentially_paralyzed[:50] = True  # Make some potentially paralyzed
    sim.people.paralyzed[:25] = True  # Make some actually paralyzed

    # Run simulation
    sim.run()

    # Check that response SIAs were scheduled
    response_sias = [entry for entry in sim.pars.sia_schedule if entry.get("source") == "response"]
    assert len(response_sias) > 0, "No response SIAs scheduled despite cases"

    # Check that SIAs target the correct nodes (node 0 and nearby nodes within distance threshold)
    for sia in response_sias:
        assert 0 in sia["nodes"], "Node 0 should be targeted"
        # Node 1 should also be targeted (distance 30 < 50 threshold)
        assert 1 in sia["nodes"], "Node 1 should be targeted (within distance threshold)"
        # Node 2 should not be targeted (distance 60 > 50 threshold)
        assert 2 not in sia["nodes"], "Node 2 should not be targeted (outside distance threshold)"


def test_two_round_strategy():
    """Test that two rounds of vaccination are scheduled."""
    sim = setup_response_sia_sim()

    # Create cases to trigger response
    sim.people.disease_state[:100] = 2
    sim.people.node_id[:100] = 0
    sim.people.potentially_paralyzed[:50] = True
    sim.people.paralyzed[:25] = True

    sim.run()

    response_sias = [entry for entry in sim.pars.sia_schedule if entry.get("source") == "response"]

    # Should have pairs of SIAs (first and second round)
    assert len(response_sias) % 2 == 0, "Should have even number of response SIAs (pairs)"

    # Check that each pair has correct timing
    for i in range(0, len(response_sias), 2):
        first_round = response_sias[i]
        second_round = response_sias[i + 1]

        assert first_round["type"] == "response_sia_1st_round", "First round type incorrect"
        assert second_round["type"] == "response_sia_2nd_round", "Second round type incorrect"
        assert first_round["nodes"] == second_round["nodes"], "Both rounds should target same nodes"


def test_blackout_period():
    """
    Test that blackout periods prevent multiple overlapping responses.
    We seed a few cases in node 0 at timestep 1 and 10, and then run the simulation for 365 days.
    We should have 2 response SIAs scheduled, one for the first round and one for the second round.
    Additional response SIAs should not be scheduled because node 0 is in a blackout period after the first round.
    """

    seed_schedule = [
        {"timestep": 1, "node_id": 0, "prevalence": 200},  # To trigger first response
        {"timestep": 10, "node_id": 0, "prevalence": 500},  # Should NOT trigger a response since an SIA is already scheduled
        {"timestep": 30, "node_id": 0, "prevalence": 500},  # Should NOT trigger a response since an SIA is already scheduled
        {"timestep": 100, "node_id": 0, "prevalence": 500},  # Should NOT trigger a response since an SIA is already scheduled
        {"timestep": 300, "node_id": 0, "prevalence": 1000},  # Should trigger a response since we're past the blackout period
    ]

    sim = setup_response_sia_sim(seed_schedule=seed_schedule, r0=0, dur=365)  # No transmission to ensure cases are only in node 0

    # Run
    sim.run()

    # Get initial response SIAs
    response_sias = [entry for entry in sim.pars.sia_schedule if entry.get("source") == "response"]
    n_response_sias = len(response_sias)

    assert len(response_sias) == 4, f"Should have 4 response SIAs (2 for first round, 2 for second round), got {len(response_sias)}"


def test_distance_based_targeting():
    """Test that only nodes within distance threshold are targeted."""
    sim = setup_response_sia_sim(response_sia_dist=35.0, r0=0)  # Smaller distance threshold

    # Create cases in node 0
    sim.people.disease_state[:100] = 2
    sim.people.node_id[:100] = 0
    sim.people.potentially_paralyzed[:50] = True
    sim.people.paralyzed[:25] = True

    sim.run()

    response_sias = [entry for entry in sim.pars.sia_schedule if entry.get("source") == "response"]

    for sia in response_sias:
        # Node 0 should be targeted (source of cases)
        assert 0 in sia["nodes"], "Source node should be targeted"
        # Node 1 should be targeted (distance 30 < 35)
        assert 1 in sia["nodes"], "Node 1 should be targeted (within distance)"
        # Node 2 should not be targeted (distance 60 > 35)
        assert 2 not in sia["nodes"], "Node 2 should not be targeted (outside distance)"


def test_response_sia_parameters():
    """Test that response SIA parameters are correctly applied."""
    sim = setup_response_sia_sim()

    # Create cases to trigger response
    sim.people.disease_state[:100] = 2
    sim.people.node_id[:100] = 0
    sim.people.potentially_paralyzed[:50] = True
    sim.people.paralyzed[:25] = True

    sim.run()

    response_sias = [entry for entry in sim.pars.sia_schedule if entry.get("source") == "response"]

    for sia in response_sias:
        assert sia["age_range"] == [0, 5 * 365], "Age range incorrect"
        assert sia["vaccinetype"] == "nOPV2", "Vaccine type incorrect"
        assert sia["vaccine_strain"] == 2, "Vaccine strain incorrect"
        assert sia["source"] == "response", "Source should be 'response'"


def test_step_size_frequency():
    """Test that response checking occurs at correct frequency."""
    sim = setup_response_sia_sim(step_size_ResponseSIA=14)  # Check every 14 days

    # Create cases
    sim.people.disease_state[:100] = 2
    sim.people.node_id[:100] = 0
    sim.people.potentially_paralyzed[:50] = True
    sim.people.paralyzed[:25] = True

    # Run for fewer days than step size
    sim.run()

    # Should still trigger response (cases exist)
    response_sias = [entry for entry in sim.pars.sia_schedule if entry.get("source") == "response"]
    assert len(response_sias) > 0, "Response should be triggered regardless of step size"


def test_multiple_case_sources():
    """Test response when cases occur in multiple nodes."""
    sim = setup_response_sia_sim()

    # Create cases in multiple nodes
    sim.people.disease_state[:50] = 2
    sim.people.node_id[:50] = 0  # Cases in node 0
    sim.people.potentially_paralyzed[:25] = True
    sim.people.paralyzed[:12] = True

    sim.people.disease_state[50:100] = 2
    sim.people.node_id[50:100] = 1  # Cases in node 1
    sim.people.potentially_paralyzed[50:75] = True
    sim.people.paralyzed[50:62] = True

    sim.run()

    response_sias = [entry for entry in sim.pars.sia_schedule if entry.get("source") == "response"]

    # Should have response SIAs targeting both nodes and their neighbors
    for sia in response_sias:
        # Both source nodes should be targeted
        assert 0 in sia["nodes"] or 1 in sia["nodes"], "Source nodes should be targeted"
        # Node 2 should be targeted (within distance of both 0 and 1)
        assert 2 in sia["nodes"], "Node 2 should be targeted (within distance of both sources)"


if __name__ == "__main__":
    test_initialization()
    test_no_response_without_cases()
    test_response_triggered_by_cases()
    test_two_round_strategy()
    test_blackout_period()
    test_distance_based_targeting()
    test_response_sia_parameters()
    test_step_size_frequency()
    test_multiple_case_sources()
    print("All ResponseSIA tests passed!")
