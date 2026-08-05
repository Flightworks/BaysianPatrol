import os
import torch
import numpy as np
from baysian_patrol_env import BaysianPatrolEnv
from export_onnx import export_onnx_model

def train_ppo_agent(total_timesteps=10000):
    """
    PPO Training loop for Bayesian Patrol RL.
    """
    env = BaysianPatrolEnv()

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

                # Calculate conv output dimension dynamically
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

        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        model.learn(total_timesteps=total_timesteps)

        os.makedirs("models", exist_ok=True)
        model.save("models/ppo_baysian_patrol")
        print("Modèle PPO entraîné et sauvegardé avec succès.")

    except ImportError:
        print("stable-baselines3 non trouvé. Exportation du modèle PyTorch/ONNX par défaut...")

    # Export ONNX model for browser client
    export_onnx_model()

if __name__ == "__main__":
    train_ppo_agent(total_timesteps=5000)
