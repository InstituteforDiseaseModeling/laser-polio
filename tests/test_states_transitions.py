# def test_initial_disease_states(setup_sim):
#     """Verify the correct initial assignment of disease states."""
#     sim = setup_sim
#     total_agents = sim.people.count
#     disease_counts = np.bincount(sim.people.disease_state, minlength=4)
#     assert disease_counts[0] > 0  # Susceptible agents exist
#     assert disease_counts[1] > 0  # Exposed agents exist
#     assert disease_counts[2] > 0  # Infected agents exist
#     assert disease_counts[3] > 0  # Recovered agents exist
