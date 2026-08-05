import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaysianPatrolPolicyNetwork(nn.Module):
    """
    CNN + MLP Policy Network for Bayesian Patrol RL.
    Convolves the 2D P(x,y,t) + Scanned grid map and combines it with the scalar aircraft state.
    """
    def __init__(self, grid_dim=32, vector_dim=10, action_dim=2):
        super().__init__()

        # CNN Branch for Grid (2, 32, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1), # -> (16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 4, 4)
            nn.ReLU(),
            nn.Flatten(), # 64 * 4 * 4 = 1024
        )

        # MLP Branch for Vector (10)
        self.mlp_vec = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
        )

        # Combined Fusion Head
        self.head = nn.Sequential(
            nn.Linear(1024 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Continuous action [-1, 1]
        )

    def forward(self, grid, vector):
        grid_feat = self.conv(grid)
        vec_feat = self.mlp_vec(vector)
        fusion = torch.cat([grid_feat, vec_feat], dim=1)
        action = self.head(fusion)
        return action

def export_onnx_model(output_path=None):
    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "models", "baysian_patrol_policy.onnx"))
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
    print(f"Modèle ONNX exporté avec succès dans : {os.path.abspath(output_path)}")

if __name__ == "__main__":
    export_onnx_model()
