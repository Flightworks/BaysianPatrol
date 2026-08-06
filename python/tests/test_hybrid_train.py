import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baysian_patrol_env import BaysianPatrolEnv
from hybrid_train import (
    fixed_evaluation_seeds, evaluate_callable, collect_expert_demonstrations,
    choose_candidate_stage, build_model,
)
from export_onnx import SB3PolicyOnnxWrapper


class TestHybridTrainingContract(unittest.TestCase):
    def test_fixed_evaluation_seeds_are_candidate_independent(self):
        self.assertEqual(fixed_evaluation_seeds(1000, 5), [1000, 1001, 1002, 1003, 1004])
        self.assertEqual(fixed_evaluation_seeds(1000, 5), fixed_evaluation_seeds(1000, 5))

    def test_expert_evaluation_reports_safe_outcomes(self):
        stats = evaluate_callable(lambda env, obs: env.expert_action(), [201, 202, 203])
        self.assertEqual(stats['episodes'], 3)
        self.assertEqual(stats['bingo_failures'], 0)
        self.assertEqual(stats['out_of_bounds'], 0)
        self.assertEqual(stats['successes'] + stats['safe_returns'], 3)

    def test_candidate_selection_keeps_safer_better_bc_stage(self):
        bc = {'success_lcb95': 90.0, 'bingo_failures': 0, 'out_of_bounds': 0, 'mean_interception_time': 12.0}
        ppo = {'success_lcb95': 30.0, 'bingo_failures': 0, 'out_of_bounds': 0, 'mean_interception_time': 9.0}
        self.assertEqual(choose_candidate_stage(bc, ppo), 'bc')
        self.assertEqual(choose_candidate_stage(ppo, bc), 'ppo')

    def test_onnx_wrapper_matches_sb3_clipped_deterministic_action(self):
        env = BaysianPatrolEnv()
        model = build_model(env, seed=99)
        obs, _ = env.reset(seed=99)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.copy_(torch.tensor([2.0, -2.0]))
            exported = SB3PolicyOnnxWrapper(model.policy)(
                torch.from_numpy(obs['grid'][None]),
                torch.from_numpy(obs['vector'][None]),
            ).numpy()[0]
        expected, _ = model.predict(obs, deterministic=True)
        np.testing.assert_allclose(exported, expected, atol=1e-7)

    def test_demonstration_shapes_match_observation_and_action(self):
        data = collect_expert_demonstrations(episodes=2, seed=300)
        self.assertEqual(data['grid'].shape[1:], (2, 32, 32))
        self.assertEqual(data['vector'].shape[1:], (10,))
        self.assertEqual(data['action'].shape[1:], (2,))
        self.assertGreater(len(data['action']), 0)
        env = BaysianPatrolEnv()
        self.assertEqual(data['action'].dtype, env.action_space.dtype)


if __name__ == '__main__':
    unittest.main()
