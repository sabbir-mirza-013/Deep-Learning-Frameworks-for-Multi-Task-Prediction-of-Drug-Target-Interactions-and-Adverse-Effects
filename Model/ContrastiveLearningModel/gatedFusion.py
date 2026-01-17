import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, dim_1, dim_2, output_dim, dropout):
        super().__init__()
        combined_dim = dim_1 + dim_2
        # Gate: determines "how much" of each feature to let through
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.Sigmoid()
        )
        # Transformation: the actual feature processing
        self.output_layer = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        g = self.gate(combined)
        f = self.output_layer(combined)
        return g * f  # Element-wise gating 

if __name__ == "__main__":
    print("GatedFusion Network")
