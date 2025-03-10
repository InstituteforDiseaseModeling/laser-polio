import numpy as np
import pytest
import laser_polio as lp
from laser_core.propertyset import PropertySet

# @pytest.fixture
def setup_sim():
    """Initialize SEIR sim for testing."""
    pars = PropertySet(dict(
        start_date  = lp.date('2020-01-01'),  # Start date of the simulation
        dur   = 10, # Number of timesteps to run the simulation
        n_ppl       = np.array([1000, 500]),  # Two nodes with populations
        cbr         = np.array([30, 25]),  # Birth rate per 1000/year
    ))
    sim = lp.SEIR_ABM(pars)
    return sim, pars

def test_total_pop():
    """Ensure the sim initializes with the correct total population."""
    sim, pars = setup_sim()
    assert sim.people.count == np.sum(sim.pars.n_ppl) == len(sim.people) == pars.n_ppl.sum(), 'Total population does not match expected values from pars.'

def test_capacity():
    """Ensure that the sim initializes with a capacity greater than the total population."""
    sim, pars = setup_sim()
    assert sim.people.capacity > sim.people.count, 'Capacity is not greater than the total population.'

def test_nodes():
    """Check that each individual has a valid node_id."""
    sim, pars = setup_sim()
    unique_nodes = np.unique(sim.people.node_id)
    node_ids = sim.people.node_id[:sim.people.count]

    assert set(unique_nodes).issubset(set(range(len(sim.pars.n_ppl)))), 'Node IDs are not valid.'
    assert len(unique_nodes) == len(sim.pars.n_ppl), 'Length of nodes not equal to number of entries in n_ppl.'

    # Count the number of each unique node ID
    node_counts = np.bincount(node_ids)
    for node_id, count in enumerate(node_counts):
        assert count == sim.pars.n_ppl[node_id] == pars.n_ppl[node_id], f'Node population for node {node_id} does not match n_ppl.'


if __name__ == '__main__':
    test_total_pop()
    test_capacity()
    test_nodes()
    print('All time tests passed.')