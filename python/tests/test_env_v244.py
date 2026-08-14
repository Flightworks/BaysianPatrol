import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baysian_patrol_env import BaysianPatrolEnv


class TestV244ScenarioContract(unittest.TestCase):
    def test_default_training_scenario_matches_v244(self):
        env = BaysianPatrolEnv()
        self.assertEqual(env.area_width, 60.0)
        self.assertEqual(env.area_height, 60.0)
        self.assertEqual(env.radar_range, 4.0)
        self.assertEqual(env.spatial_sigma, 20.0)
        self.assertEqual(env.time_sigma_minutes, 60.0)

    def test_scenario_parameters_remain_overridable(self):
        env = BaysianPatrolEnv(
            area_width=100.0,
            area_height=80.0,
            radar_range=15.0,
            spatial_sigma=2.0,
            time_sigma_minutes=10.0,
        )
        self.assertEqual(env.area_width, 100.0)
        self.assertEqual(env.area_height, 80.0)
        self.assertEqual(env.radar_range, 15.0)
        self.assertEqual(env.spatial_sigma, 2.0)
        self.assertEqual(env.time_sigma_minutes, 10.0)

    def test_initial_truth_is_conditioned_on_search_area(self):
        env = BaysianPatrolEnv()
        for seed in range(100):
            env.reset(seed=seed)
            self.assertGreaterEqual(env.target_x, env.min_x)
            self.assertLessEqual(env.target_x, env.max_x)
            self.assertGreaterEqual(env.target_y, env.min_y)
            self.assertLessEqual(env.target_y, env.max_y)

    def test_initial_belief_includes_time_uncertainty_along_route(self):
        env = BaysianPatrolEnv(spatial_sigma=1.0, time_sigma_minutes=60.0)
        env.reset(seed=2026)
        env.datum_x = 0.0
        env.datum_y = 0.0
        env.belief_speed = 10.0
        env.belief_heading = 0.0
        belief = env._initial_belief().astype(float)
        mean_x = float((belief * env.mesh_x).sum())
        mean_y = float((belief * env.mesh_y).sum())
        variance_x = float((belief * (env.mesh_x - mean_x) ** 2).sum())
        variance_y = float((belief * (env.mesh_y - mean_y) ** 2).sum())
        self.assertGreater(variance_y, variance_x + 50.0)


if __name__ == '__main__':
    unittest.main()
