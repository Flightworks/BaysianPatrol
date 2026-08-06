"""Reproducible PPO autoresearch for BaysianPatrol.

Selection is safety-first: success lower confidence bound, then bingo rate,
then interception time. Reward is diagnostic, never the primary objective.
"""
import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from baysian_patrol_env import BaysianPatrolEnv
from export_onnx import export_onnx_model

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(ROOT, "autoresearch_v3_results.json")


class CustomCNNVectorExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        grid_shape = observation_space.spaces["grid"].shape
        vector_dim = observation_space.spaces["vector"].shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(grid_shape[0], 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten()
        )
        with torch.no_grad():
            n_flatten = self.conv(torch.as_tensor(observation_space.spaces["grid"].sample()[None]).float()).shape[1]
        self.vector_fc = nn.Sequential(nn.Linear(vector_dim, 64), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(n_flatten + 64, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(torch.cat((self.conv(observations["grid"]), self.vector_fc(observations["vector"])), dim=1))


def wilson_lower_bound(successes, trials, z=1.96):
    if trials == 0:
        return 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return float((centre - margin) / denom * 100.0)


def ranking_key(stats):
    """Safety-first comparison; reward is only a final tie-breaker."""
    return (
        round(stats["success_lcb95"], 6),
        -round(stats["bingo_fail_rate"], 6),
        -round(stats["out_of_bounds_rate"], 6),
        -round(stats["mean_interception_time"], 6),
        round(stats["mean_reward"], 6),
    )


def load_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if "best_key" in data:
                return data
        except (OSError, ValueError):
            pass
    return {"version": 3, "best_key": None, "best_stats": None, "best_params": None, "history": []}


def save_results(data):
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, RESULTS_FILE)


def make_env(seed, hp):
    def init():
        env = BaysianPatrolEnv(
            reward_detection=hp["r_det"], reward_exploration=hp["r_prob"],
            penalty_time=hp["p_time"], penalty_bingo=hp["p_bingo"]
        )
        env.reset(seed=seed)
        return env
    return init


def evaluate_policy(model, n_episodes=100, seed=50000):
    env = BaysianPatrolEnv()
    rewards, lengths, interception_times = [], [], []
    successes = bingo = out_bounds = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total, steps, done = 0.0, 0, False
        final_info = {}
        while not done and steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total += float(reward); steps += 1; done = terminated or truncated
        rewards.append(total); lengths.append(steps)
        if final_info.get("intercepted"):
            successes += 1; interception_times.append(steps * env.dt)
        bingo += int(final_info.get("bingo_failed", False))
        out_bounds += int(final_info.get("out_of_bounds", False))
    stats = {
        "episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes * 100.0,
        "success_lcb95": wilson_lower_bound(successes, n_episodes),
        "bingo_fail_rate": bingo / n_episodes * 100.0,
        "out_of_bounds_rate": out_bounds / n_episodes * 100.0,
        "mean_reward": float(np.mean(rewards)),
        "mean_flight_length": float(np.mean(lengths)),
        "mean_interception_time": float(np.mean(interception_times)) if interception_times else 300.0,
    }
    stats["ranking_key"] = list(ranking_key(stats))
    return stats


def propose_hyperparams(rng):
    return {
        "lr": float(rng.choice([1e-4, 2e-4, 3e-4, 5e-4])),
        "ent_coef": float(rng.choice([0.0, 0.003, 0.01])),
        "gamma": float(rng.choice([0.99, 0.995, 0.999])),
        "gae_lambda": float(rng.choice([0.90, 0.95, 0.98])),
        "r_det": float(rng.choice([400.0, 800.0, 1200.0])),
        "r_prob": float(rng.choice([10.0, 20.0, 40.0])),
        "p_time": float(rng.choice([0.05, 0.1, 0.2])),
        "p_bingo": float(rng.choice([800.0, 1200.0, 1600.0])),
        "n_steps": int(rng.choice([1024, 2048])),
        "features_dim": int(rng.choice([128, 256])),
    }


def run_autoresearch(experiments=6, timesteps=150000, n_envs=4, eval_episodes=100, seed=2026):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    results = load_results()
    models_dir = os.path.join(ROOT, "models")
    public_dir = os.path.abspath(os.path.join(ROOT, "..", "public", "models"))
    os.makedirs(models_dir, exist_ok=True); os.makedirs(public_dir, exist_ok=True)

    for index in range(1, experiments + 1):
        hp = propose_hyperparams(rng)
        hp["seed"] = seed + index * 10000
        print(f"\n🧪 [V3 {index}/{experiments}] {hp}", flush=True)
        vec = SubprocVecEnv([make_env(hp["seed"] + j, hp) for j in range(n_envs)], start_method="fork")
        policy_kwargs = {
            "features_extractor_class": CustomCNNVectorExtractor,
            "features_extractor_kwargs": {"features_dim": hp["features_dim"]},
            "normalize_images": False,
        }
        model = PPO("MultiInputPolicy", vec, learning_rate=hp["lr"], ent_coef=hp["ent_coef"],
                    gamma=hp["gamma"], gae_lambda=hp["gae_lambda"], n_steps=hp["n_steps"],
                    batch_size=256, n_epochs=10, seed=hp["seed"], policy_kwargs=policy_kwargs,
                    verbose=0)
        start = time.time()
        try:
            model.learn(total_timesteps=timesteps, progress_bar=False)
        finally:
            vec.close()
        stats = evaluate_policy(model, eval_episodes, seed=100000 + index * 1000)
        key = ranking_key(stats)
        entry = {"experiment": index, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "hyperparams": hp, "stats": stats, "ranking_key": list(key), "status": "DISCARDED"}
        old_key = tuple(results["best_key"]) if results["best_key"] is not None else None
        print(f"   résultat: succès={stats['success_rate']:.1f}% (LCB={stats['success_lcb95']:.1f}%) | "
              f"Bingo={stats['bingo_fail_rate']:.1f}% | temps={stats['mean_interception_time']:.1f} min", flush=True)
        if old_key is None or key > old_key:
            entry["status"] = "CHAMPION_KEPT"
            results["best_key"] = list(key); results["best_stats"] = stats; results["best_params"] = hp
            champion = os.path.join(models_dir, "PPO_CHAMPION_V3")
            model.save(champion)
            try:
                onnx = os.path.join(public_dir, "PPO_CHAMPION_V3.onnx")
                export_onnx_model(zip_model_path=champion + ".zip", output_path=onnx)
                shutil.copyfile(onnx, os.path.join(public_dir, "baysian_patrol_policy.onnx"))
            except Exception as exc:
                entry["onnx_error"] = repr(exc)
            print("   🏆 nouveau champion", flush=True)
        results["history"].append(entry); save_results(results)
        print(f"   durée={((time.time()-start)/60):.1f} min", flush=True)
    print(f"\n✅ V3 terminée — meilleur classement: {results['best_key']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=150000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run_autoresearch(args.experiments, args.timesteps, args.n_envs, args.eval_episodes, args.seed)
