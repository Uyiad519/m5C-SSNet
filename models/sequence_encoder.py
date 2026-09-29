"""SpliceBERT-based semantic sequence encoder."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from utils.runtime import _masked_mean_pool, _strip_special_token_features

class SpliceBERTSequenceEncoder(nn.Module):
    """Encode RNA sequence semantics with a frozen SpliceBERT model."""

    def __init__(self, pretrained_model_path, representation_dim=256, max_length=256):
        super().__init__()
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        try:
            masked_language_model = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_path
            )
            self.splicebert = masked_language_model.base_model
        except (OSError, ValueError):
            self.splicebert = AutoModel.from_pretrained(
                pretrained_model_path, add_pooling_layer=False
            )

        if self.tokenizer.pad_token is None:
            fallback_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback_token is None:
                raise ValueError("The SpliceBERT tokenizer does not define a padding token.")
            self.tokenizer.pad_token = fallback_token

        self.splicebert.requires_grad_(False)
        hidden_size = int(self.splicebert.config.hidden_size)
        self.sequence_projection = nn.Sequential(
            nn.Linear(hidden_size, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, sequences):
        """Return one semantic representation for each input sequence."""
        tokenized_sequences = [
            " ".join(str(sequence).upper().replace("U", "T"))
            for sequence in sequences
        ]
        tokenized = self.tokenizer(
            tokenized_sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        tokenized = {name: value.to(device) for name, value in tokenized.items()}
        attention_mask = tokenized.get("attention_mask")
        special_tokens_mask = tokenized.pop("special_tokens_mask", None)

        self.splicebert.eval()
        with torch.no_grad():
            outputs = self.splicebert(**tokenized)
        nucleotide_embeddings, valid_mask = _strip_special_token_features(
            outputs.last_hidden_state.float(), attention_mask, special_tokens_mask
        )
        pooled_representation = _masked_mean_pool(
            nucleotide_embeddings, valid_mask
        )
        return self.sequence_projection(pooled_representation)
