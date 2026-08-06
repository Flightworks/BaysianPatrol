"""Hybrid expert-imitation-PPO training pipeline for BaysianPatrol v2.3.1.

Candidates are never promoted to the active browser model by this script. The
promotion gate is deliberately separate and requires fixed-seed evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from baysian_patrol_env import BaysianPatrolEnv
from export_onnx import export_onnx_model

ROOT = os.path.dirname(os.path.abspath(__file__))


class HybridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        grid_shape = observation_space.spaces["grid"].shape
        vector_dim = observation_space.spaces["vector"].shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(grid_shape[0], 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.spaces["grid"].sample()[None]).float()
            flat = self.conv(sample).shape[1]
        self.vector_net = nn.Sequential(nn.Linear(vector_dim, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(flat + 64, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.fusion(torch.cat((self.conv(observations["grid"]), self.vector_net(observations["vector"])), dim=1))


def fixed_evaluation_seeds(base_seed: int = 200_000, episodes: int = 500) -> list[int]:
    return list(range(int(base_seed), int(base_seed) + int(episodes)))


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    p = successes / trials
    denominator = 1.0 + z*z/trials
    centre = p + z*z/(2.0*trials)
    margin = z*math.sqrt((p*(1.0-p) + z*z/(4.0*trials))/trials)
    return (centre-margin)/denominator*100.0


def candidate_key(stats: dict) -> tuple:
    """Hard safety first, then Wilson success bound and interception time."""
    mean_time = stats.get("mean_interception_time")
    time_value = float(mean_time) if mean_time is not None else float("inf")
    return (
        -int(stats.get("bingo_failures", 0)),
        -int(stats.get("out_of_bounds", 0)),
        float(stats.get("success_lcb95", 0.0)),
        -time_value,
    )


def choose_candidate_stage(bc_stats: dict, ppo_stats: dict) -> str:
    return "bc" if candidate_key(bc_stats) >= candidate_key(ppo_stats) else "ppo"


def evaluate_callable(policy: Callable, seeds: Sequence[int], curriculum_level: int = 4) -> dict:
    outcomes: list[str] = []
    interception_times: list[float] = []
    rewards: list[float] = []
    for seed in seeds:
        env = BaysianPatrolEnv(curriculum_level=curriculum_level)
        obs, _ = env.reset(seed=int(seed))
        total = 0.0
        final = {"outcome": "time_limit"}
        for _ in range(env.max_steps):
            action = np.asarray(policy(env, obs), dtype=np.float32)
            obs, reward, terminated, truncated, final = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        outcome = str(final.get("outcome", "time_limit"))
        outcomes.append(outcome)
        rewards.append(total)
        if outcome == "intercepted":
            interception_times.append(float(final["elapsed_min"]))
    successes = outcomes.count("intercepted")
    safe_returns = outcomes.count("safe_rtb")
    bingo = outcomes.count("bingo")
    out = outcomes.count("out_of_bounds")
    n = len(outcomes)
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": successes/n*100.0 if n else 0.0,
        "success_lcb95": wilson_lower_bound(successes, n),
        "safe_returns": safe_returns,
        "bingo_failures": bingo,
        "bingo_fail_rate": bingo/n*100.0 if n else 0.0,
        "out_of_bounds": out,
        "out_of_bounds_rate": out/n*100.0 if n else 0.0,
        "mean_interception_time": float(np.mean(interception_times)) if interception_times else None,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "outcomes": {name: outcomes.count(name) for name in sorted(set(outcomes))},
    }


def collect_expert_demonstrations(episodes: int = 500, seed: int = 10_000, curriculum_levels: Sequence[int] = (1, 2, 3, 4)) -> dict[str, np.ndarray]:
    grids: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in range(int(episodes)):
        level = int(curriculum_levels[episode % len(curriculum_levels)])
        env = BaysianPatrolEnv(curriculum_level=level)
        obs, _ = env.reset(seed=int(seed) + episode)
        for _ in range(env.max_steps):
            action = env.expert_action()
            grids.append(obs["grid"].copy())
            vectors.append(obs["vector"].copy())
            actions.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return {
        "grid": np.asarray(grids, dtype=np.float32),
        "vector": np.asarray(vectors, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
    }


def build_model(env, seed: int = 2026) -> PPO:
    return PPO(
        "MultiInputPolicy", env,
        learning_rate=1e-4, n_steps=1024, batch_size=256, n_epochs=5,
        gamma=0.995, gae_lambda=0.95, ent_coef=0.001, seed=seed, verbose=0,
        policy_kwargs={
            "features_extractor_class": HybridExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "normalize_images": False,
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
        },
    )


def behavior_clone(model: PPO, data: dict[str, np.ndarray], epochs: int = 20, batch_size: int = 256, learning_rate: float = 3e-4, seed: int = 2026) -> list[float]:
    rng = np.random.default_rng(seed)
    device = model.device
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    losses: list[float] = []
    count = len(data["action"])
    model.policy.train()
    for _ in range(int(epochs)):
        indices = rng.permutation(count)
        epoch_losses = []
        for start in range(0, count, batch_size):
            idx = indices[start:start+batch_size]
            obs = {
                "grid": torch.as_tensor(data["grid"][idx], device=device),
                "vector": torch.as_tensor(data["vector"][idx], device=device),
            }
            target = torch.as_tensor(data["action"][idx], device=device)
            features = model.policy.extract_features(obs)
            latent = model.policy.mlp_extractor.forward_actor(features)
            predicted = torch.tanh(model.policy.action_net(latent))
            loss = torch.mean((predicted-target)**2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    model.policy.eval()
    return losses


def model_policy(model: PPO) -> Callable:
    def predict(_env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return predict


def _make_env(seed: int, curriculum_level: int):
    def init():
        env = BaysianPatrolEnv(curriculum_level=curriculum_level)
        env.reset(seed=seed)
        return env
    return init


def train_hybrid(demo_episodes: int = 500, bc_epochs: int = 20, ppo_steps_per_level: int = 50_000, n_envs: int = 4, eval_episodes: int = 500, seed: int = 2026) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    initial_vec = SubprocVecEnv([_make_env(seed+i, 1) for i in range(n_envs)], start_method="fork")
    model = build_model(initial_vec, seed)
    demonstrations = collect_expert_demonstrations(demo_episodes, seed=10_000)
    losses = behavior_clone(model, demonstrations, bc_epochs, seed=seed)
    models_dir = os.path.join(ROOT, "models")
    public_dir = os.path.abspath(os.path.join(ROOT, "..", "public", "models"))
    os.makedirs(models_dir, exist_ok=True); os.makedirs(public_dir, exist_ok=True)
    bc_checkpoint = os.path.join(models_dir, "BC_CHECKPOINT_V231")
    model.save(bc_checkpoint)
    validation_seeds = fixed_evaluation_seeds(150_000, min(100, eval_episodes))
    bc_stats = evaluate_callable(model_policy(model), validation_seeds)
    initial_vec.close()

    for level in (1, 2, 3, 4):
        vec = SubprocVecEnv([_make_env(seed+level*1000+i, level) for i in range(n_envs)], start_method="fork")
        model.set_env(vec)
        model.learn(total_timesteps=ppo_steps_per_level, reset_num_timesteps=False, progress_bar=False)
        vec.close()

    test_seeds = fixed_evaluation_seeds(200_000, eval_episodes)
    ppo_stats = evaluate_callable(model_policy(model), test_seeds)
    bc_model = PPO.load(bc_checkpoint+".zip")
    bc_test_stats = evaluate_callable(model_policy(bc_model), test_seeds)
    expert_stats = evaluate_callable(lambda env, obs: env.expert_action(), test_seeds)
    selected_stage = choose_candidate_stage(bc_test_stats, ppo_stats)
    selected_model = bc_model if selected_stage == "bc" else model
    selected_stats = bc_test_stats if selected_stage == "bc" else ppo_stats
    candidate = os.path.join(models_dir, "PPO_CANDIDATE_V231")
    selected_model.save(candidate)
    onnx_path = os.path.join(public_dir, "PPO_CANDIDATE_V231.onnx")
    export_onnx_model(candidate+".zip", onnx_path)
    report = {
        "version": "2.3.1-candidate", "seed": seed,
        "demonstration_transitions": len(demonstrations["action"]),
        "bc_final_loss": losses[-1], "bc_validation": bc_stats,
        "bc_test": bc_test_stats, "ppo_test": ppo_stats,
        "selected_stage": selected_stage, "selected_test": selected_stats,
        "expert_test": expert_stats,
        "promotion_eligible": (
            selected_stats["bingo_failures"] == 0 and selected_stats["out_of_bounds"] == 0
            and selected_stats["success_lcb95"] >= expert_stats["success_lcb95"]-2.0
        ),
        "candidate_zip": candidate+".zip", "candidate_onnx": onnx_path,
    }
    report_path = os.path.join(ROOT, "hybrid_v231_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-episodes", type=int, default=500)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--ppo-steps-per-level", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    train_hybrid(args.demo_episodes, args.bc_epochs, args.ppo_steps_per_level, args.n_envs, args.eval_episodes, args.seed)
