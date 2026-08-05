import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from baysian_patrol_env import BaysianPatrolEnv

class TestBaysianPatrolEnv(unittest.TestCase):
    def setUp(self):
        self.env = BaysianPatrolEnv(grid_dim=32)

    def test_reset(self):
        obs, info = self.env.reset(seed=42)
        self.assertIn("grid", obs)
        self.assertIn("vector", obs)
        self.assertEqual(obs["grid"].shape, (2, 32, 32))
        self.assertEqual(obs["vector"].shape, (10,))

    def test_step(self):
        obs, info = self.env.reset(seed=42)
        action = np.array([0.5, 0.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_boundary(self):
        obs, info = self.env.reset(seed=42)
        # Force helico to move out of bounds
        self.env.helico_x = 100.0
        self.env.helico_y = 100.0
        action = np.array([0.0, 1.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertLess(reward, 0.0)

if __name__ == "__main__":
    unittest.main()
