# Test acq_risk changes over time

# # Test Edge Cases
# def test_no_transmission_in_immune_population():
#     """Ensure that if all individuals are immune, no infections occur."""
#     pars = PropertySet(dict(
#         start_date  = lp.date('2020-01-01'),
#         dur   = 30,
#         n_ppl       = np.array([1000, 500]),
#         init_immun  = [1.0],  # 100% immune
#         init_prev   = [0.0],  # No initial infections
#     ))
#     sim = lp.SEIR_ABM(pars)
#     sim.add_component(lp.DiseaseState_ABM(sim))
#     sim.run()
#     assert np.sum(sim.people.disease_state == 2) == 0  # No one should be Infected
