import os
import sys
import io

# Force UTF-8 stdout/stderr encoding on Windows to prevent UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import time
import shutil
import random
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from baysian_patrol_env import BaysianPatrolEnv
from export_onnx import export_onnx_model

RESULTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "autoresearch_results.json"))

def load_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"best_score": -999999.0, "best_params": None, "history": []}

def save_results(data):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def evaluate_policy(model, n_episodes=20):
    env = BaysianPatrolEnv()
    rewards = []
    successes = 0
    ep_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_rew = 0.0
        steps = 0
        while not done and steps < 180:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            total_rew += rew
            steps += 1
            if terminated or truncated:
                done = True
                if getattr(env, "intercepted", False):
                    successes += 1
        rewards.append(total_rew)
        ep_lengths.append(steps)

    mean_rew = float(np.mean(rewards))
    success_rate = float(successes / n_episodes) * 100.0
    mean_length = float(np.mean(ep_lengths))
    return mean_rew, success_rate, mean_length

def run_autoresearch(num_experiments=10, timesteps_per_experiment=30000):
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch
    import torch.nn as nn

    class CustomCNNVectorExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=256):
            super().__init__(observation_space, features_dim)
            grid_shape = observation_space.spaces["grid"].shape
            vector_shape = observation_space.spaces["vector"].shape

            self.conv = nn.Sequential(
                nn.Conv2d(grid_shape[0], 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )

            with torch.no_grad():
                dummy_grid = torch.as_tensor(observation_space.spaces["grid"].sample()[None]).float()
                n_flatten = self.conv(dummy_grid).shape[1]

            self.vector_fc = nn.Sequential(
                nn.Linear(vector_shape[0], 64),
                nn.ReLU()
            )

            self.linear = nn.Sequential(
                nn.Linear(n_flatten + 64, features_dim),
                nn.ReLU()
            )

        def forward(self, observations):
            grid_feats = self.conv(observations["grid"])
            vec_feats = self.vector_fc(observations["vector"])
            concatenated = torch.cat([grid_feats, vec_feats], dim=1)
            return self.linear(concatenated)

    results_data = load_results()
    print("==========================================================")
    print("🤖 DEMARRAGE DE L'AUTORESEARCH (KARPATHY LOOP)")
    print(f"Meilleur Score Champion Actuel : {results_data['best_score']:.1f}")
    print("==========================================================")

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    public_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "models"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(public_models_dir, exist_ok=True)

    for i in range(1, num_experiments + 1):
        exp_name = f"PPO_AutoResearch_Exp_{i}"
        
        # Propose Hyperparameters (Mutation / Exploration)
        lr = random.choice([0.0001, 0.0003, 0.0005, 0.001])
        r_det = random.choice([400.0, 600.0, 800.0, 1200.0])
        r_prob = random.choice([5.0, 10.0, 20.0, 30.0])
        p_time = random.choice([0.05, 0.1, 0.2])
        p_bingo = random.choice([500.0, 1000.0, 1500.0])
        gamma = random.choice([0.98, 0.99, 0.995])

        print(f"\n🧪 [Expérimentation {i}/{num_experiments}] Proposition de réglages :")
        print(f"   ► LR={lr}, R_Det={r_det}, R_Prob={r_prob}, P_Time={p_time}, P_Bingo={p_bingo}, Gamma={gamma}")

        env = BaysianPatrolEnv(
            reward_detection=r_det,
            reward_exploration=r_prob,
            penalty_time=p_time,
            penalty_bingo=p_bingo,
        )

        policy_kwargs = dict(
            features_extractor_class=CustomCNNVectorExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )

        tb_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "tensorboard_logs"))
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=lr,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=tb_log_dir
        )

        # Train model
        print(f"   ► Entraînement en cours sur {timesteps_per_experiment} pas...")
        model.learn(total_timesteps=timesteps_per_experiment)

        # Evaluate model performance
        mean_rew, success_rate, mean_len = evaluate_policy(model, n_episodes=20)
        composite_score = mean_rew + (success_rate * 5.0) - (mean_len * 0.5)

        print(f"   ► ÉVALUATION : Récompense={mean_rew:.1f} | Taux Interception={success_rate:.1f}% | Vol Moyen={mean_len:.1f}m | Score Composite={composite_score:.1f}")

        exp_entry = {
            "experiment": i,
            "name": exp_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "score": round(composite_score, 1),
            "mean_reward": round(mean_rew, 1),
            "success_rate": round(success_rate, 1),
            "mean_flight_length": round(mean_len, 1),
            "hyperparams": {
                "lr": lr,
                "r_det": r_det,
                "r_prob": r_prob,
                "p_time": p_time,
                "p_bingo": p_bingo,
                "gamma": gamma
            },
            "status": "DISCARDED"
        }

        # Keep or Discard Decision
        if composite_score > results_data["best_score"]:
            print(f"🏆 NOUVEAU CHAMPION DECOUVERT ! ({composite_score:.1f} > {results_data['best_score']:.1f})")
            results_data["best_score"] = round(composite_score, 1)
            results_data["best_params"] = exp_entry["hyperparams"]
            exp_entry["status"] = "CHAMPION_KEPT"

            # Save champion PyTorch & ONNX model
            champion_zip = os.path.join(models_dir, "PPO_CHAMPION_AUTORESEARCH.zip")
            champion_onnx = os.path.join(public_models_dir, "PPO_CHAMPION_AUTORESEARCH.onnx")
            active_onnx = os.path.join(public_models_dir, "baysian_patrol_policy.onnx")

            model.save(champion_zip)
            export_onnx_model(zip_model_path=champion_zip, output_path=champion_onnx)
            shutil.copyfile(champion_onnx, active_onnx)
            print("   ► Modèle ONNX Champion compilé et activé avec succès dans l'application !")

        results_data["history"].append(exp_entry)
        save_results(results_data)

    print("\n==========================================================")
    print("✅ BOUCLE AUTORESEARCH TERMINEE AVEC SUCCES !")
    print(f"Meilleur Score Final Champion : {results_data['best_score']:.1f}")
    print("==========================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=int, default=5, help="Nombre d'expérimentations")
    parser.add_argument("--timesteps", type=int, default=25000, help="Timesteps par expérimentation")
    args = parser.parse_args()

    run_autoresearch(num_experiments=args.experiments, timesteps_per_experiment=args.timesteps)
