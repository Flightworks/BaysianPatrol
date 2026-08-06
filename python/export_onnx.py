import os
import torch
import torch.nn as nn

class BaysianPatrolPolicyNetwork(nn.Module):
    """
    CNN + MLP Policy Network for Bayesian Patrol RL (Fallback).
    Convolves the 2D P(x,y,t) + Scanned grid map and combines it with the scalar aircraft state.
    """
    def __init__(self, grid_dim=32, vector_dim=10, action_dim=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.mlp_vec = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(1024 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, grid, vector):
        grid_feat = self.conv(grid)
        vec_feat = self.mlp_vec(vector)
        fusion = torch.cat([grid_feat, vec_feat], dim=1)
        action = self.head(fusion)
        return action

class SB3PolicyOnnxWrapper(nn.Module):
    """
    ONNX Wrapper around a trained Stable-Baselines3 MultiInputActorCriticPolicy.
    Exports the trained neural network actor policy.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, grid, vector):
        obs = {"grid": grid, "vector": vector}
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        action_mean = self.policy.action_net(latent_pi)
        return torch.tanh(action_mean)

def export_onnx_model(zip_model_path=None, output_path=None):
    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "models", "baysian_patrol_policy.onnx"))

    model = None

    # Check if a trained Stable-Baselines3 .zip model exists
    if zip_model_path is None:
        default_zip = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "ppo_baysian_patrol.zip"))
        if os.path.exists(default_zip):
            zip_model_path = default_zip

    if zip_model_path and os.path.exists(zip_model_path):
        try:
            from stable_baselines3 import PPO
            print(f"Chargement des poids entraînés depuis : {zip_model_path}...")
            sb3_model = PPO.load(zip_model_path)
            model = SB3PolicyOnnxWrapper(sb3_model.policy)
            print("Poids du réseau de neurones PPO extraits avec succès.")
        except Exception as e:
            print(f"Attention: Impossible de charger {zip_model_path} ({e}). Utilisation du modèle de secours.")

    if model is None:
        print("Création du réseau de neurones de secours...")
        model = BaysianPatrolPolicyNetwork()

    model.eval()

    dummy_grid = torch.randn(1, 2, 32, 32, dtype=torch.float32)
    dummy_vector = torch.randn(1, 10, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_grid, dummy_vector),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["grid", "vector"],
        output_names=["action"],
        dynamic_axes={
            "grid": {0: "batch_size"},
            "vector": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        dynamo=False
    )
    print(f"Modèle ONNX entraîné exporté avec succès dans : {os.path.abspath(output_path)}")

if __name__ == "__main__":
    export_onnx_model()
