"""Training and evaluation workflow for m5C-SSNet."""

import gc
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from data.dataset import RNADataset
from models.m5c_ssnet import M5CSSNet
from utils.runtime import (
    _allocate_shared_csv_logger_version,
    _build_trainer_device_config,
    _cleanup_shared_csv_logger_version,
    _format_trainer_device_config,
    _rank_zero_print,
)

class LearningCurve(pl.Callback):
    """Save training and validation learning curves after each epoch."""

    def __init__(self, figsize=(10, 6), names=None):
        super().__init__()
        self.figsize = figsize
        self.names = list(names or ["loss", "acc", "f1"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            trainer.sanity_checking
            or not trainer.is_global_zero
            or not hasattr(pl_module, "history")
        ):
            return

        save_dir = trainer.logger.log_dir
        os.makedirs(save_dir, exist_ok=True)
        for name in self.names:
            train_values = pl_module.history.get(name, [])
            validation_values = pl_module.history.get(f"val_{name}", [])
            if not train_values:
                continue

            plt.figure(figsize=self.figsize)
            plt.plot(train_values, label=f"Train {name}")
            if validation_values:
                plt.plot(validation_values, label=f"Validation {name}")
            plt.title(f"{name.upper()} by epoch")
            plt.xlabel("Epoch")
            plt.ylabel(name.upper())
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}_curve.png"))
            plt.close()

def _release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _score_to_float(score):
    return float(score.item()) if score is not None else None

def train(config):
    """Train m5C-SSNet and evaluate the best validation-AUC checkpoint."""
    _rank_zero_print("Starting m5C-SSNet training")
    os.makedirs(config["output_dir"], exist_ok=True)

    seed = config.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
        config["seed"] = seed
    _rank_zero_print(f"Random seed: {seed}")
    pl.seed_everything(seed)

    dataset_options = {
        "use_bppm_cache": config.get("use_bppm_cache", True),
        "bppm_cache_dir": config.get("bppm_cache_dir"),
        "bppm_input_len": config.get("sequence_length", 41),
    }
    datasets = {}
    dataset_paths = {
        "train": config["train_data_path"],
        "validation": config["validation_data_path"],
        "test": config["test_data_path"],
    }
    for split_name, data_path in dataset_paths.items():
        _rank_zero_print(f"\nLoading {split_name} data: {data_path}")
        dataset = RNADataset(data_path, **dataset_options)
        datasets[split_name] = dataset
        stats = dataset.get_length_stats()
        _rank_zero_print(
            f"Loaded {len(dataset)} {split_name} samples; length range: "
            f"{stats['min_length']}-{stats['max_length']} nt"
        )

    _rank_zero_print("\n=== Dataset sizes ===")
    for split_name, dataset in datasets.items():
        _rank_zero_print(f"{split_name.capitalize()}: {len(dataset)} samples")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )

    logger_version, logger_state_path = _allocate_shared_csv_logger_version(
        save_dir=config["output_dir"],
        name="m5C-SSNet",
        configured_version=config.get("logger_version"),
    )
    logger = CSVLogger(
        save_dir=config["output_dir"],
        name="m5C-SSNet",
        version=logger_version,
    )
    experiment_dir = logger.log_dir
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    _rank_zero_print(f"\nExperiment directory: {experiment_dir}")

    model = M5CSSNet(
        sequence_length=config.get("sequence_length", 41),
        representation_dim=config["representation_dim"],
        classifier_hidden_dim=config["classifier_hidden_dim"],
        num_classes=config["num_classes"],
        classifier_dropout=config["classifier_dropout"],
        learning_rate=config["learning_rate"],
        gine_hidden_dim=config.get("gine_hidden_dim", 64),
        gine_dropout=config.get("gine_dropout", 0.2),
        num_gine_layers=config.get("num_gine_layers", 2),
        structure_cnn_channels=config.get("structure_cnn_channels", 64),
        structure_cnn_dropout=config.get("structure_cnn_dropout", 0.2),
        structure_cnn_kernel_size=config.get("structure_cnn_kernel_size", 5),
        bppm_top_k=config.get("bppm_top_k", 2),
        minimum_pairing_probability=config.get(
            "minimum_pairing_probability", 0.01
        ),
        splicebert_max_length=config.get("splicebert_max_length", 256),
        pretrained_model_path=config["pretrained_model_path"],
    )
    model.set_class_weights(datasets["train"].labels)

    best_auc_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_auc-{epoch:02d}-{val_auc:.3f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    best_aupr_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_aupr-{epoch:02d}-{val_aupr:.3f}",
        monitor="val_aupr",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        best_auc_checkpoint,
        best_aupr_checkpoint,
        LearningCurve(figsize=(15, 5), names=["loss", "auc", "aupr"]),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=config.get("early_stopping_patience", 10),
            min_delta=0.001,
            verbose=True,
        ),
    ]

    training_device_config = _build_trainer_device_config(config, stage="train")
    evaluation_device_config = _build_trainer_device_config(config, stage="eval")
    _rank_zero_print(
        "Training devices: "
        f"{_format_trainer_device_config(training_device_config)}"
    )
    _rank_zero_print(
        "Evaluation devices: "
        f"{_format_trainer_device_config(evaluation_device_config)}"
    )

    lightning_trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        gradient_clip_val=config["gradient_clip_value"],
        **training_device_config,
    )

    with open(os.path.join(experiment_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    start_time = datetime.now()
    _rank_zero_print(f"\nTraining started at {start_time:%Y-%m-%d %H:%M:%S}")
    lightning_trainer.fit(model, train_loader, validation_loader)
    if not lightning_trainer.is_global_zero:
        return None

    best_auc_path = best_auc_checkpoint.best_model_path
    best_aupr_path = best_aupr_checkpoint.best_model_path
    if not best_auc_path:
        raise RuntimeError("No validation-AUC checkpoint was created.")

    training_summary = {
        "best_val_auc": _score_to_float(best_auc_checkpoint.best_model_score),
        "best_val_aupr": _score_to_float(best_aupr_checkpoint.best_model_score),
        "best_val_acc": max(model.history["val_acc"]),
        "best_val_f1": max(model.history["val_f1"]),
        "final_val_auc": model.history["val_auc"][-1],
        "final_val_aupr": model.history["val_aupr"][-1],
        "final_val_acc": model.history["val_acc"][-1],
        "final_val_f1": model.history["val_f1"][-1],
        "total_epochs": len(model.history["val_auc"]),
    }

    del model, lightning_trainer
    _release_memory()

    _rank_zero_print("\nEvaluating the best validation-AUC checkpoint")
    test_model = M5CSSNet.load_from_checkpoint(
        best_auc_path,
        strict=True,
        pretrained_model_path=config["pretrained_model_path"],
    )
    test_trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **evaluation_device_config,
    )
    test_results = test_trainer.test(test_model, test_loader)[0]
    fusion_weights = test_model.get_fusion_weights()
    _rank_zero_print(f"Test results: {test_results}")

    del test_model, test_trainer
    _release_memory()

    results = {
        "training": training_summary,
        "testing": {"best_auc_model": test_results},
        "paths": {
            "output_dir": experiment_dir,
            "dataset_dir": os.path.dirname(config["train_data_path"]),
            "best_auc_model_path": best_auc_path,
            "best_aupr_model_path": best_aupr_path,
        },
        "fusion_weights": fusion_weights,
    }
    results_path = os.path.join(experiment_dir, "full_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    end_time = datetime.now()
    _rank_zero_print(f"\nTraining finished at {end_time:%Y-%m-%d %H:%M:%S}")
    _rank_zero_print(
        "Training summary\n"
        f"Best validation AUC: {training_summary['best_val_auc']:.4f}\n"
        f"Best validation AUPR: {training_summary['best_val_aupr']:.4f}\n"
        f"Best validation accuracy: {training_summary['best_val_acc']:.4f}\n"
        f"Best validation F1: {training_summary['best_val_f1']:.4f}\n"
        f"Test accuracy: {test_results.get('test_acc', 0):.4f}\n"
        f"Test AUC: {test_results.get('test_auc', 0):.4f}\n"
        f"Test AUPR: {test_results.get('test_aupr', 0):.4f}\n"
        f"Test MCC: {test_results.get('test_mcc', 0):.4f}\n"
        f"Test F1: {test_results.get('test_f1', 0):.4f}\n"
        f"Completed epochs: {training_summary['total_epochs']}\n"
        f"Detailed results: {results_path}"
    )
    _cleanup_shared_csv_logger_version(logger_state_path)
    return results
