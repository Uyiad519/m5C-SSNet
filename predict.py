#!/usr/bin/env python
"""Run m5C-SSNet inference on labeled or unlabeled FASTA records."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from utils.bppm import load_or_build_bppm_cache
from utils.config import (
    DEFAULT_BPPM_CACHE_DIR,
    PRETRAINED_MODEL_PATH,
    PROJECT_ROOT,
)

class PredictionFastaDataset(Dataset):
    """Load FASTA records and optionally parse binary labels from their headers."""

    def __init__(
        self,
        file_path,
        use_bppm_cache=True,
        bppm_cache_dir=None,
        bppm_input_len=41,
    ):
        self.file_path = file_path
        self.use_bppm_cache = bool(use_bppm_cache)
        self.bppm_cache_dir = bppm_cache_dir
        self.bppm_input_len = int(bppm_input_len)
        self.headers = []
        self.sequences = []
        self.labels = []
        self.bppm_matrices = None

        self._read_fasta()
        self._load_bppm()

    def _read_fasta(self):
        current_header = None
        current_sequence = []

        with open(self.file_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_header is not None:
                        self._append_record(current_header, current_sequence)
                    current_header = line[1:]
                    current_sequence = []
                else:
                    if current_header is None:
                        raise ValueError("A FASTA sequence was found before its header.")
                    current_sequence.append(line.upper().replace("U", "T"))

        if current_header is not None:
            self._append_record(current_header, current_sequence)
        if not self.sequences:
            raise ValueError(f"No valid FASTA records were found in {self.file_path}.")

    def _append_record(self, header, sequence_parts):
        sequence = "".join(sequence_parts)
        if not sequence:
            raise ValueError(f"FASTA record has an empty sequence: {header}")
        label_match = re.search(
            r"(?<![A-Za-z0-9_])label\s*=\s*([01])(?!\d)",
            header,
            flags=re.IGNORECASE,
        )
        self.headers.append(header)
        self.sequences.append(sequence)
        self.labels.append(int(label_match.group(1)) if label_match else -1)

    def _load_bppm(self):
        if not self.use_bppm_cache:
            print(
                "BPPM input is disabled; the structural encoder will use sequence edges only."
            )
            return
        cache_dir = self.bppm_cache_dir or DEFAULT_BPPM_CACHE_DIR
        self.bppm_matrices = load_or_build_bppm_cache(
            self.sequences,
            dataset_path=self.file_path,
            cache_dir=cache_dir,
            input_length=self.bppm_input_len,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        item = {
            "index": index,
            "header": self.headers[index],
            "sequence": self.sequences[index],
            "label": self.labels[index],
        }
        if self.bppm_matrices is not None:
            item["bppm"] = self.bppm_matrices[index]
        return item

def resolve_device(device_name):
    """Resolve an explicit device or select the first available GPU."""
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)

def compute_binary_metrics(labels, probabilities):
    """Compute binary classification metrics from class probabilities."""
    predictions = probabilities.argmax(axis=1)
    positive_scores = probabilities[:, 1]
    confusion = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, _, _ = confusion.ravel()
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0.0
    )

    try:
        auc = float(roc_auc_score(labels, positive_scores))
    except ValueError:
        auc = None
    try:
        aupr = float(average_precision_score(labels, positive_scores))
    except ValueError:
        aupr = None

    return {
        "acc": float(accuracy_score(labels, predictions)),
        "auc": auc,
        "aupr": aupr,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "confusion_matrix": confusion.tolist(),
    }

def resolve_output_paths(
    checkpoint_path,
    test_data_path,
    output_csv=None,
    metrics_json=None,
):
    """Resolve output paths and create their parent directories."""
    checkpoint_path = Path(checkpoint_path).resolve()
    test_stem = Path(test_data_path).stem
    checkpoint_stem = checkpoint_path.stem
    default_dir = checkpoint_path.parent.parent / "predictions"

    csv_path = (
        Path(output_csv)
        if output_csv
        else default_dir / f"{test_stem}_{checkpoint_stem}_scores.csv"
    )
    json_path = (
        Path(metrics_json)
        if metrics_json
        else default_dir / f"{test_stem}_{checkpoint_stem}_metrics.json"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, json_path

def run_prediction(config):
    """Load a checkpoint, run inference, and export scores and optional metrics."""
    from models.m5c_ssnet import M5CSSNet

    checkpoint_path = config["checkpoint_path"]
    test_data_path = config["test_data_path"]
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(test_data_path):
        raise FileNotFoundError(f"FASTA file not found: {test_data_path}")

    device = resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"Loading prediction data: {test_data_path}")
    dataset = PredictionFastaDataset(
        test_data_path,
        use_bppm_cache=config.get("use_bppm_cache", True),
        bppm_cache_dir=config.get("bppm_cache_dir"),
        bppm_input_len=config.get("bppm_input_len", 41),
    )
    print(f"Prediction samples: {len(dataset)}")

    data_loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    model = M5CSSNet.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,
        pretrained_model_path=PRETRAINED_MODEL_PATH,
    ).to(device)
    model.eval()
    model._reset_gate_statistics()

    rows = []
    probability_batches = []
    label_batches = []
    with torch.inference_mode():
        for batch in data_loader:
            model_inputs = {"sequence": batch["sequence"]}
            if "bppm" in batch:
                model_inputs["bppm"] = batch["bppm"].to(device)

            probabilities = F.softmax(model(model_inputs), dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            batch_labels = batch["label"].cpu().numpy()

            for index, header, sequence, label, prediction, scores in zip(
                batch["index"].cpu().numpy(),
                batch["header"],
                batch["sequence"],
                batch_labels,
                predictions,
                probabilities,
            ):
                row = {
                    "sample_index": int(index),
                    "header": header,
                    "sequence": sequence,
                    "label": int(label),
                    "pred_label": int(prediction),
                }
                for class_index, score in enumerate(scores):
                    row[f"score_class_{class_index}"] = float(score)
                if len(scores) == 2:
                    row["score"] = float(scores[1])
                rows.append(row)

            probability_batches.append(probabilities)
            label_batches.append(batch_labels)

    output_csv, metrics_json = resolve_output_paths(
        checkpoint_path,
        test_data_path,
        output_csv=config.get("output_csv"),
        metrics_json=config.get("metrics_json"),
    )
    num_classes = len(
        [key for key in rows[0] if key.startswith("score_class_")]
    )
    fieldnames = ["sample_index", "header", "sequence", "label", "pred_label"]
    fieldnames.extend(f"score_class_{index}" for index in range(num_classes))
    if num_classes == 2:
        fieldnames.append("score")

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prediction scores saved to: {output_csv}")

    labels = np.concatenate(label_batches)
    probabilities = np.concatenate(probability_batches)
    valid_labels = labels >= 0
    metrics = None
    if valid_labels.any() and probabilities.shape[1] == 2:
        metrics = compute_binary_metrics(
            labels[valid_labels], probabilities[valid_labels]
        )
        with open(metrics_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Evaluation metrics saved to: {metrics_json}")
        print(
            f"AUC={metrics['auc']}, AUPR={metrics['aupr']}, "
            f"MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}, "
            f"ACC={metrics['acc']:.4f}"
        )

    return {
        "prediction_scores_path": str(output_csv),
        "metrics_path": str(metrics_json) if metrics is not None else None,
        "metrics": metrics,
    }

def parse_args():
    """Parse command-line arguments for prediction."""
    parser = argparse.ArgumentParser(
        description="Run m5C-SSNet and export per-sample prediction scores."
    )
    parser.add_argument("checkpoint_path", help="Path to an m5C-SSNet checkpoint.")
    parser.add_argument(
        "--test-data-path",
        default=os.path.join(
            PROJECT_ROOT, "data", "data_split", "ind-5mC-mm.fasta"
        ),
        help="Path to a labeled or unlabeled FASTA file.",
    )
    parser.add_argument("--output-csv", help="Optional prediction CSV path.")
    parser.add_argument("--metrics-json", help="Optional metrics JSON path.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="Device such as auto, cpu, or cuda:0.")
    parser.add_argument("--bppm-cache-dir", default=DEFAULT_BPPM_CACHE_DIR)
    parser.add_argument("--bppm-input-len", type=int, default=41)
    parser.add_argument(
        "--no-bppm-cache",
        action="store_true",
        help="Disable BPPM input and use sequence edges only.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    run_prediction(
        {
            "checkpoint_path": args.checkpoint_path,
            "test_data_path": args.test_data_path,
            "output_csv": args.output_csv,
            "metrics_json": args.metrics_json,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "use_bppm_cache": not args.no_bppm_cache,
            "bppm_cache_dir": args.bppm_cache_dir,
            "bppm_input_len": args.bppm_input_len,
        }
    )

if __name__ == "__main__":
    main()
