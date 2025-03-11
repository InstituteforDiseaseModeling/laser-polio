"""
Test that disease model timers increment correctly and
don't go below zero.
"""

from laser_core.propertyset import PropertySet

# Setup parameters
params = PropertySet({"population_size": 1000, "init_prev": 0.01, "infection_rate": 0.3, "dur": 100})

# def test_timers():

#     # Initialize the model
#     model = lp.SEIRModel(params)
#     model.add_component(lp.Exposed(model))
#     model.add_component(lp.Infected(model))
#     model.add_component(lp.Recovered(model))

#     # Run the simulation
#     model.run()
#     # model.plot_results()

#     # Tests
#     assert np.mean(model.results.I[1:20]) > model.results.I[0], 'Disease prevalence should increase after t=0'
#     assert (model.population.exposure_timer >= 0).all(), 'Exposure timer should not go below zero'
#     assert (model.population.infection_timer >= 0).all(), 'Infection timer should not go below zero'


# if __name__ == '__main__':
#     test_timers()
#     print('All time tests passed.')
