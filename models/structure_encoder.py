"""BPPM-guided structural context encoder for m5C-SSNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BPPMGINEBlock(nn.Module):
    """Propagate messages over sequence-adjacency and BPPM-derived edges."""

    def __init__(self, hidden_dim, dropout=0.2, top_k=2, min_prob=0.01):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.top_k = int(top_k)
        self.min_prob = float(min_prob)
        self.eps = nn.Parameter(torch.zeros(1))
        self.sequence_edge_embedding = nn.Parameter(torch.zeros(self.hidden_dim))
        self.bppm_edge_embedding = nn.Parameter(torch.zeros(self.hidden_dim))
        nn.init.normal_(self.sequence_edge_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.bppm_edge_embedding, mean=0.0, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def _select_structural_edges(self, bppm):
        if bppm is None:
            return None

        bppm = bppm.to(dtype=torch.float32)
        sequence_length = bppm.size(1)
        diagonal = torch.eye(
            sequence_length, device=bppm.device, dtype=torch.bool
        ).unsqueeze(0)
        bppm = bppm.masked_fill(diagonal, 0.0)

        top_k = min(self.top_k, sequence_length)
        top_values, top_indices = torch.topk(bppm, k=top_k, dim=-1)
        valid_edges = top_values >= self.min_prob
        selected_edges = torch.zeros_like(bppm)
        selected_edges.scatter_(
            -1, top_indices, top_values * valid_edges.to(top_values.dtype)
        )
        return selected_edges

    def forward(self, node_states, bppm=None):
        sequence_edge_bias = self.sequence_edge_embedding.view(1, 1, -1)
        bppm_edge_bias = self.bppm_edge_embedding.view(1, 1, -1)

        left_messages = torch.zeros_like(node_states)
        right_messages = torch.zeros_like(node_states)
        left_messages[:, 1:, :] = F.relu(
            node_states[:, :-1, :] + sequence_edge_bias
        )
        right_messages[:, :-1, :] = F.relu(
            node_states[:, 1:, :] + sequence_edge_bias
        )
        sequence_messages = left_messages + right_messages

        selected_edges = self._select_structural_edges(bppm)
        if selected_edges is None:
            structural_messages = torch.zeros_like(node_states)
        else:
            structural_values = F.relu(node_states + bppm_edge_bias)
            structural_messages = torch.bmm(selected_edges, structural_values)

        aggregated_messages = sequence_messages + structural_messages
        updated_states = self.mlp(
            (1.0 + self.eps) * node_states + aggregated_messages
        )
        return self.output_norm(node_states + updated_states)

class StructuralContextEncoder(nn.Module):
    """Encode a centered RNA window with BPPM-aware GINE layers and a CNN."""

    def __init__(
        self,
        sequence_length=41,
        gine_hidden_dim=64,
        representation_dim=256,
        gine_dropout=0.2,
        num_gine_layers=2,
        structure_cnn_channels=64,
        structure_cnn_dropout=0.2,
        structure_cnn_kernel_size=5,
        bppm_top_k=2,
        minimum_pairing_probability=0.01,
    ):
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.gine_hidden_dim = int(gine_hidden_dim)
        self.representation_dim = int(representation_dim)
        self.num_gine_layers = int(num_gine_layers)
        self.structure_cnn_channels = int(structure_cnn_channels)
        self.structure_cnn_dropout = float(structure_cnn_dropout)
        self.structure_cnn_kernel_size = int(structure_cnn_kernel_size)

        self.node_encoder = nn.Sequential(
            nn.Linear(5, self.gine_hidden_dim),
            nn.LayerNorm(self.gine_hidden_dim),
            nn.GELU(),
            nn.Dropout(gine_dropout),
        )
        self.gine_layers = nn.ModuleList(
            [
                BPPMGINEBlock(
                    self.gine_hidden_dim,
                    dropout=gine_dropout,
                    top_k=bppm_top_k,
                    min_prob=minimum_pairing_probability,
                )
                for _ in range(self.num_gine_layers)
            ]
        )

        padding = self.structure_cnn_kernel_size // 2
        self.conv1 = self._convolution_block(
            self.gine_hidden_dim, self.structure_cnn_channels, padding
        )
        self.conv2 = self._convolution_block(
            self.structure_cnn_channels, self.structure_cnn_channels, padding
        )
        self.conv3 = self._convolution_block(
            self.structure_cnn_channels, self.structure_cnn_channels, padding
        )
        self.fc = nn.Sequential(
            nn.Linear(self._infer_flattened_dim(), self.representation_dim),
            nn.ReLU(),
            nn.Dropout(self.structure_cnn_dropout),
        )

    def _convolution_block(self, input_channels, output_channels, padding):
        return nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=self.structure_cnn_kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.structure_cnn_dropout),
        )

    def _extract_center_window(self, sequence):
        sequence = str(sequence).upper().replace("U", "T")
        if len(sequence) >= self.sequence_length:
            start = (len(sequence) - self.sequence_length) // 2
            return sequence[start : start + self.sequence_length]
        padding = self.sequence_length - len(sequence)
        left_padding = padding // 2
        right_padding = padding - left_padding
        return "N" * left_padding + sequence + "N" * right_padding

    def _infer_flattened_dim(self):
        with torch.no_grad():
            tensor = torch.zeros(
                1, self.gine_hidden_dim, self.sequence_length, dtype=torch.float32
            )
            tensor = self.conv1(tensor)
            tensor = self.conv2(tensor)
            tensor = self.conv3(tensor)
            return tensor.shape[1] * tensor.shape[2]

    def _encode_node_features(self, sequences):
        device = next(self.parameters()).device
        batch_size = len(sequences)
        one_hot = torch.zeros(
            (batch_size, self.sequence_length, 4),
            dtype=torch.float32,
            device=device,
        )
        relative_positions = torch.linspace(
            -1.0, 1.0, steps=self.sequence_length, device=device
        ).view(1, self.sequence_length, 1)
        relative_positions = relative_positions.expand(batch_size, -1, -1)
        nucleotide_indices = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

        for row_index, sequence in enumerate(sequences):
            centered_sequence = self._extract_center_window(sequence)
            for column_index, nucleotide in enumerate(centered_sequence):
                nucleotide_index = nucleotide_indices.get(nucleotide)
                if nucleotide_index is not None:
                    one_hot[row_index, column_index, nucleotide_index] = 1.0

        return torch.cat([one_hot, relative_positions], dim=-1)

    def _prepare_bppm(self, bppm, batch_size, device):
        if bppm is None:
            return None
        bppm = torch.as_tensor(bppm, dtype=torch.float32, device=device)
        if bppm.dim() == 2:
            bppm = bppm.unsqueeze(0)
        if bppm.dim() != 3:
            raise ValueError("BPPM input must have shape [batch, length, length].")

        if bppm.size(0) == 1 and batch_size > 1:
            bppm = bppm.expand(batch_size, -1, -1)
        elif bppm.size(0) != batch_size:
            raise ValueError(
                f"BPPM batch size {bppm.size(0)} does not match sequence batch size "
                f"{batch_size}."
            )

        target_length = self.sequence_length
        row_length, column_length = bppm.shape[1:]
        if row_length >= target_length:
            row_start = (row_length - target_length) // 2
            bppm = bppm[:, row_start : row_start + target_length, :]
        else:
            top = (target_length - row_length) // 2
            bottom = target_length - row_length - top
            bppm = F.pad(bppm, (0, 0, top, bottom))

        if column_length >= target_length:
            column_start = (column_length - target_length) // 2
            bppm = bppm[:, :, column_start : column_start + target_length]
        else:
            left = (target_length - column_length) // 2
            right = target_length - column_length - left
            bppm = F.pad(bppm, (left, right, 0, 0))
        return bppm

    def forward(self, sequences, bppm=None):
        node_states = self.node_encoder(self._encode_node_features(sequences))
        bppm = self._prepare_bppm(
            bppm, batch_size=node_states.size(0), device=node_states.device
        )
        for gine_layer in self.gine_layers:
            node_states = gine_layer(node_states, bppm=bppm)

        tensor = node_states.transpose(1, 2)
        tensor = self.conv1(tensor)
        tensor = self.conv2(tensor)
        tensor = self.conv3(tensor)
        return self.fc(tensor.reshape(tensor.size(0), -1))
