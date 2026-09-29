"""Adaptive fusion of semantic and structural representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGatedFusion(nn.Module):
    """Learn sample-specific weights for sequence and structure representations."""

    def __init__(self, representation_dim=256, gate_hidden_dim=256):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(representation_dim * 2, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(gate_hidden_dim, 2),
        )

    def forward(self, sequence_representation, structural_representation):
        gate_input = torch.cat(
            [sequence_representation, structural_representation], dim=-1
        )
        fusion_weights = F.softmax(self.gate_network(gate_input), dim=-1)
        fused_representation = (
            fusion_weights[:, 0:1] * sequence_representation
            + fusion_weights[:, 1:2] * structural_representation
        )
        return fused_representation, fusion_weights
