import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autoresearch_v3 import wilson_lower_bound, ranking_key


class TestAutoresearchV3(unittest.TestCase):
    def test_wilson_bound_is_below_observed_rate(self):
        self.assertLess(wilson_lower_bound(24, 100), 24.0)
        self.assertGreater(wilson_lower_bound(95, 100), 85.0)

    def test_selection_prioritizes_success_over_reward(self):
        good = {
            'success_lcb95': 70.0, 'bingo_fail_rate': 20.0,
            'out_of_bounds_rate': 0.0, 'mean_interception_time': 100.0,
            'mean_reward': -1000.0,
        }
        bad = {
            'success_lcb95': 20.0, 'bingo_fail_rate': 0.0,
            'out_of_bounds_rate': 0.0, 'mean_interception_time': 20.0,
            'mean_reward': 10000.0,
        }
        self.assertGreater(ranking_key(good), ranking_key(bad))


if __name__ == '__main__':
    unittest.main()
