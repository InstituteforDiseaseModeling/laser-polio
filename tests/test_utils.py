import unittest

import numpy as np

from laser_polio.utils import pbincount


class TestUtils(unittest.TestCase):
    def test_pbincount(self):
        num_nodes = 1024
        num_people = 1_000_000
        bins = np.random.randint(0, num_nodes, size=num_people)

        expected = np.bincount(bins, minlength=num_nodes)
        result = pbincount(bins, num_nodes)
        assert np.all(result == expected), (
            f"np.bincount(): result != expected ({(result != expected).sum()} mismatches / {num_people:,} people)"
        )

        return

    def test_pbincount_weighted(self):
        num_nodes = 1024
        num_people = 1_000_000
        bins = np.random.randint(0, num_nodes, size=num_people)
        weights = np.random.rand(num_people)

        expected = np.bincount(bins, weights=weights, minlength=num_nodes)
        result = pbincount(bins, num_nodes, weights=weights)
        assert np.allclose(result, expected), "np.bincount(): result != expected"

        return

    def test_pbincount_with_types(self):
        num_nodes = 1024
        num_people = 1_000_000
        bins = np.random.randint(0, num_nodes, size=num_people)

        expected = np.bincount(bins, minlength=num_nodes)
        for in_type, out_type in [(np.int32, np.int32), (np.int32, np.int64), (np.int64, np.int32), (np.int64, np.int64)]:
            result = pbincount(bins.astype(in_type), num_nodes, dtype=out_type)
            assert np.all(result == expected), (
                f"np.bincount(): result != expected ({(result != expected).sum()} mismatches / {num_people} people)"
            )

        return

    def test_pbincount_weighted_with_types(self):
        num_nodes = 1024
        num_people = 1_000_000
        bins = np.random.randint(0, num_nodes, size=num_people)
        weights = np.random.rand(num_people)

        expected = np.bincount(bins, weights=weights, minlength=num_nodes)
        for in_type, out_type in [(np.float32, np.float32), (np.float32, np.float64), (np.float64, np.float32), (np.float64, np.float64)]:
            result = pbincount(bins, num_nodes, weights=weights.astype(in_type), dtype=out_type)
            assert np.allclose(result, expected), f"np.bincount(): result{(in_type, out_type)} != expected."

        return


if __name__ == "__main__":
    unittest.main()
