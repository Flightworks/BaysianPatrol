import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baysian_patrol_env import BaysianPatrolEnv


class TestV231Contract(unittest.TestCase):
    def test_legacy_reward_exploration_keyword_remains_supported(self):
        env = BaysianPatrolEnv(reward_exploration=3.5)
        self.assertEqual(env.reward_probability, 3.5)

    def test_observation_distinguishes_opposite_frigate_bearings(self):
        env = BaysianPatrolEnv()
        env.reset(seed=7)
        env.helico_x = env.helico_y = 0.0
        env.frigate_x, env.frigate_y = 30.0, 0.0
        right = env._get_obs()['vector'].copy()
        env.frigate_x, env.frigate_y = -30.0, 0.0
        left = env._get_obs()['vector'].copy()
        self.assertFalse(np.array_equal(right, left))

    def test_belief_observation_has_unit_peak_and_internal_mass_one(self):
        env = BaysianPatrolEnv()
        obs, _ = env.reset(seed=11)
        self.assertAlmostEqual(float(env.grid_p.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(obs['grid'][0].max()), 1.0, places=6)
        self.assertGreater(float(obs['grid'][0].mean()), 0.0)

    def test_negative_scan_reduces_local_posterior_mass(self):
        env = BaysianPatrolEnv()
        env.reset(seed=13)
        env.helico_x = env.datum_x
        env.helico_y = env.datum_y
        env.target_x = env.max_x
        env.target_y = env.max_y
        yy, xx = np.ogrid[:env.grid_size, :env.grid_size]
        cx = int(np.argmin(np.abs(env.grid_x - env.helico_x)))
        cy = int(np.argmin(np.abs(env.grid_y - env.helico_y)))
        local = (xx - cx) ** 2 + (yy - cy) ** 2 <= 3 ** 2
        before = float(env.grid_p[local].sum())
        _, _, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertFalse(terminated)
        self.assertFalse(info['intercepted'])
        after = float(env.grid_p[local].sum())
        self.assertLess(after, before)

    def test_relative_waypoint_action_drives_deterministic_autopilot(self):
        env = BaysianPatrolEnv()
        env.reset(seed=17)
        env.helico_x = env.helico_y = 0.0
        env.frigate_x = env.frigate_y = 0.0
        env.fuel_remaining = env.max_fuel
        env.target_x, env.target_y = -40.0, -40.0
        env.step(np.array([1.0, 0.0], dtype=np.float32))
        self.assertGreater(env.helico_x, 0.0)
        self.assertAlmostEqual(env.helico_y, 0.0, places=5)
        self.assertAlmostEqual(env.current_speed, env.max_speed, places=5)

    def test_geofence_prevents_waypoint_from_leaving_search_area(self):
        env = BaysianPatrolEnv()
        env.reset(seed=18)
        env.helico_x, env.helico_y = env.max_x-0.1, 0.0
        env.frigate_x, env.frigate_y = env.helico_x, env.helico_y
        env.fuel_remaining = env.max_fuel
        env.target_x, env.target_y = env.min_x, env.min_y
        _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        self.assertLessEqual(env.helico_x, env.max_x)
        self.assertFalse(info['out_of_bounds'])

    def test_safety_shield_returns_and_terminates_safely(self):
        env = BaysianPatrolEnv()
        env.reset(seed=19)
        env.helico_x, env.helico_y = 10.0, 0.0
        env.frigate_x = env.frigate_y = 0.0
        env.target_x, env.target_y = 40.0, 40.0
        return_time = 10.0 / env.max_speed * 60.0
        env.fuel_remaining = return_time + env.bingo_buffer - 0.1
        outcome = None
        for _ in range(20):
            _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            outcome = info['outcome']
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertEqual(outcome, 'safe_rtb')
        self.assertFalse(info['bingo_fail'])
        self.assertLessEqual(np.hypot(env.helico_x-env.frigate_x, env.helico_y-env.frigate_y), env.rtb_radius)

    def test_expert_policy_is_safe_on_fixed_scenarios(self):
        env = BaysianPatrolEnv()
        for seed in range(20):
            obs, _ = env.reset(seed=10_000 + seed)
            final = None
            for _ in range(env.max_steps):
                action = env.expert_action()
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    final = info
                    break
            self.assertIsNotNone(final)
            self.assertFalse(final['bingo_fail'])
            self.assertFalse(final['out_of_bounds'])
            self.assertIn(final['outcome'], ('intercepted', 'safe_rtb'))


if __name__ == '__main__':
    unittest.main()
