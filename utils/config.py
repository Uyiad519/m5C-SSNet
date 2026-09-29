"""Default configuration for m5C-SSNet training and evaluation."""

import os

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models_folder", "SpliceBERT.1024nt"
)
DEFAULT_BPPM_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "bppm_cache")

def get_config():
    """Return the default m5C-SSNet configuration."""
    return {
        "train_data_path": os.path.join(
            PROJECT_ROOT, "data", "data_split", "train-5mC-mm.fasta"
        ),
        "validation_data_path": os.path.join(
            PROJECT_ROOT, "data", "data_split", "val-5mC-mm.fasta"
        ),
        "test_data_path": os.path.join(
            PROJECT_ROOT, "data", "data_split", "ind-5mC-mm.fasta"
        ),
        "output_dir": os.path.join(PROJECT_ROOT, "logs"),
        "sequence_length": 41,
        "representation_dim": 256,
        "classifier_hidden_dim": 256,
        "num_classes": 2,
        "classifier_dropout": 0.6,
        "learning_rate": 1e-4,
        "gine_hidden_dim": 64,
        "gine_dropout": 0.2,
        "num_gine_layers": 2,
        "structure_cnn_channels": 64,
        "structure_cnn_dropout": 0.2,
        "structure_cnn_kernel_size": 5,
        "bppm_top_k": 2,
        "minimum_pairing_probability": 0.01,
        "pretrained_model_path": PRETRAINED_MODEL_PATH,
        "splicebert_max_length": 256,
        "use_bppm_cache": True,
        "bppm_cache_dir": DEFAULT_BPPM_CACHE_DIR,
        "batch_size": 32,
        "num_workers": 4,
        "num_epochs": 60,
        "early_stopping_patience": 10,
        "gradient_clip_value": 1.0,
        "seed": None,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "gpu_devices": 0,
        "evaluation_gpu_devices": None,
        "strategy": "auto",
        "precision": "16-mixed" if torch.cuda.is_available() else "32-true",
        "cpu_devices": 1,
    }
