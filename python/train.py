import sys
import io
import os
import shutil

# Force UTF-8 encoding on Windows standard outputs
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
from baysian_patrol_env import BaysianPatrolEnv
from export_onnx import export_onnx_model

from datetime import datetime

def train_ppo_agent(
    total_timesteps=50000,
    model_name=None,
    learning_rate=0.0003,
    reward_detection=500.0,
    reward_exploration=10.0,
    penalty_time=0.1,
    penalty_bingo=1000.0,
):
    """
    PPO Training loop for Bayesian Patrol RL.
    """
    env = BaysianPatrolEnv(
        reward_detection=reward_detection,
        reward_exploration=reward_exploration,
        penalty_time=penalty_time,
        penalty_bingo=penalty_bingo,
    )

    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = f"baysian_patrol_ppo_{total_timesteps // 1000}k_{timestamp}"

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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

        policy_kwargs = dict(
            features_extractor_class=CustomCNNVectorExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )

        tb_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "tensorboard_logs"))
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_log_dir
        )
        print(f"Début de l'entraînement PPO sur {total_timesteps} pas de temps ({model_name})...")
        model.learn(total_timesteps=total_timesteps)

        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
        os.makedirs(models_dir, exist_ok=True)
        model.save(os.path.join(models_dir, model_name))
        print(f"Modèle PPO entraîné et sauvegardé avec succès sous : {model_name}")

    except ImportError:
        print("stable-baselines3 non trouvé. Exportation du modèle PyTorch/ONNX par défaut...")

    # Export named ONNX model to public/models/ AND set as active baysian_patrol_policy.onnx
    public_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "models"))
    os.makedirs(public_models_dir, exist_ok=True)

    named_onnx_path = os.path.join(public_models_dir, f"{model_name}.onnx")
    active_onnx_path = os.path.join(public_models_dir, "baysian_patrol_policy.onnx")
    trained_zip_path = os.path.join(models_dir, f"{model_name}.zip")

    export_onnx_model(zip_model_path=trained_zip_path, output_path=named_onnx_path)
    shutil.copyfile(named_onnx_path, active_onnx_path)
    print(f"Modèle ONNX entraîné conservé sous : {named_onnx_path} et activé pour la simulation.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000, help="Nombre de timesteps d'entraînement")
    parser.add_argument("--name", type=str, default=None, help="Nom personnalisé du modèle")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate PPO")
    parser.add_argument("--r_det", type=float, default=500.0, help="Récompense Détection")
    parser.add_argument("--r_prob", type=float, default=10.0, help="Récompense Exploration")
    parser.add_argument("--p_time", type=float, default=0.1, help="Pénalité Temporelle")
    parser.add_argument("--p_bingo", type=float, default=1000.0, help="Pénalité Bingo")
    args = parser.parse_args()

    train_ppo_agent(
        total_timesteps=args.timesteps,
        model_name=args.name,
        learning_rate=args.lr,
        reward_detection=args.r_det,
        reward_exploration=args.r_prob,
        penalty_time=args.p_time,
        penalty_bingo=args.p_bingo,
    )
