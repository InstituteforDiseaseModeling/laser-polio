import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from laser_polio.mymodel import Model
from laser_polio.numpynumba.population import ExtendedLF as Population

# Directories for test data
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"


class TestModel(unittest.TestCase):
    """Unit tests for the Model class."""

    def test_model_initialization(self):
        """Test that Model initializes correctly and loads the manifest."""
        params = MagicMock()
        params.input_dir = str(TEST_DATA_DIR)  # Ensure valid path
        params.ticks = 10
        params.prevalence = 0.1
        params.viz = False

        # Directly load the manifest instead of mocking it
        model = Model(params)

        assert model.manifest is not None
        assert isinstance(model.nn_nodes, dict)
        assert isinstance(model.initial_populations, np.ndarray)
        assert isinstance(model.cbrs, dict)
        assert isinstance(model.input_pop, Population)

    def test_model_get(self):
        """Test the Model factory method to ensure caching behavior."""
        params = MagicMock()
        params.input_dir = str(TEST_DATA_DIR)
        params.ticks = 10
        params.prevalence = 0.1
        params.viz = False

        with patch.object(Model, "_check_for_cached", return_value=False), patch.object(Model, "_init_from_data", return_value=None):
            model = Model.get(params)
            assert isinstance(model, Model)

    @unittest.skip("needs more work")
    def test_model_save(self):
        """Test that Model.save raises an error if age distribution is uninitialized."""
        params = MagicMock()
        params.input_dir = str(TEST_DATA_DIR)

        with (
            patch("os.path.isfile", return_value=True),
            patch("importlib.util.spec_from_file_location"),
            patch("importlib.util.module_from_spec"),
            patch("laser_polio.mods.age_init.age_data_manager.get_data", return_value=None),
        ):
            model = Model(params)

            with pytest.raises(ValueError, match="age_distribution uninitialized while saving"):
                model.save("test.h5")

    def test_propagate_population(self):
        """Test that population propagates correctly."""
        params = MagicMock()
        params.input_dir = str(TEST_DATA_DIR)
        params.ticks = 10
        params.prevalence = 0.1
        params.viz = False

        model = Model(params)
        model.nodes = MagicMock()
        model.nodes.population = np.zeros((11, 5), dtype=np.uint32)
        model.nodes.population[0] = [10, 20, 30, 40, 50]

        Model.propagate_population(model, 0)

        assert np.array_equal(model.nodes.population[1], [10, 20, 30, 40, 50])

    def test_assign_node_ids(self):
        """Test that node IDs are assigned correctly."""
        params = MagicMock()
        params.input_dir = str(TEST_DATA_DIR)

        model = Model(params)
        model.population = MagicMock()
        model.initial_populations = [3, 2, 1]
        model.population.nodeid = np.zeros(sum(model.initial_populations), dtype=np.uint32)

        model._assign_node_ids()

        expected = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint32)
        assert np.array_equal(model.population.nodeid, expected)

    @unittest.skip("needs more work, and for cached file to exist")
    def test_check_for_cached(self):
        """Test the caching mechanism."""
        params = MagicMock()
        params.input_dir = str(TEST_DATA_DIR)

        with (
            patch("os.path.isfile", return_value=True),
            patch("os.listdir", return_value=["cache.h5"]),
            patch("os.makedirs"),
            patch("laser_polio.mods.age_init.age_data_manager.get_data", return_value=np.array([1, 2, 3])),
            patch("laser_polio.numpynumba.population.check_hdf5_attributes", return_value=True),
            patch.object(Model, "_init_from_file"),
        ):
            model = Model(params)
            assert model._check_for_cached()
