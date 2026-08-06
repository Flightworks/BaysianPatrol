import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baysian_patrol_env import BaysianPatrolEnv


class TestBaysianPatrolEnv(unittest.TestCase):
    def test_reset_is_reproducible_with_seed(self):
        a = BaysianPatrolEnv(); b = BaysianPatrolEnv()
        oa, _ = a.reset(seed=42); ob, _ = b.reset(seed=42)
        np.testing.assert_array_equal(oa['grid'], ob['grid'])
        np.testing.assert_array_equal(oa['vector'], ob['vector'])
        self.assertEqual((a.datum_x, a.datum_y, a.target_speed), (b.datum_x, b.datum_y, b.target_speed))

    def test_different_seeds_produce_different_scenarios(self):
        env = BaysianPatrolEnv()
        env.reset(seed=42); first = (env.datum_x, env.datum_y, env.target_speed)
        env.reset(seed=43); second = (env.datum_x, env.datum_y, env.target_speed)
        self.assertNotEqual(first, second)

    def test_step_observation_stays_inside_declared_space(self):
        env = BaysianPatrolEnv(); env.reset(seed=42)
        env.helico_x = 100.0; env.helico_y = 100.0
        obs, _, terminated, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(terminated)

    def test_step_contract_and_probability_normalization(self):
        env = BaysianPatrolEnv(); obs, _ = env.reset(seed=42)
        next_obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0], dtype=np.float32))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertTrue(env.observation_space.contains(next_obs))
        self.assertAlmostEqual(float(env.grid_p.sum()), 1.0, places=5)
        self.assertIn('information_gain', info)

    def test_detection_uses_seeded_randomness(self):
        def rollout(seed):
            env = BaysianPatrolEnv(); env.reset(seed=seed)
            env.target_x, env.target_y = env.helico_x, env.helico_y
            _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
            return terminated, info['intercepted']
        self.assertEqual(rollout(123), rollout(123))


if __name__ == '__main__':
    unittest.main()
