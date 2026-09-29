"""Classification metrics for m5C site prediction."""

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

class ClassificationMetric:
    """Accumulate logits and compute epoch-level classification metrics."""

    def __init__(self, num_classes=2, positive_label=1):
        self.num_classes = int(num_classes)
        self.positive_label = int(positive_label)
        self.logits = []
        self.targets = []

    def reset(self):
        self.logits.clear()
        self.targets.clear()

    def update(self, logits, targets):
        self.logits.append(logits.detach().cpu().float())
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)
        self.targets.append(targets.detach().cpu().long())

    def _compute_binary_metrics(self, true_labels, predictions, scores):
        confusion = confusion_matrix(true_labels, predictions, labels=[0, 1])
        true_negative, false_positive, false_negative, true_positive = confusion.ravel()
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive > 0
            else 0.0
        )
        class_accuracy = {
            0: float(specificity),
            1: float(
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative > 0
                else 0.0
            ),
        }

        if len(np.unique(true_labels)) > 1:
            auc = float(roc_auc_score(true_labels, scores))
            aupr = float(average_precision_score(true_labels, scores))
        else:
            auc = 0.0
            aupr = 0.0

        recall = float(
            recall_score(
                true_labels,
                predictions,
                pos_label=self.positive_label,
                zero_division=0,
            )
        )
        precision = float(
            precision_score(
                true_labels,
                predictions,
                pos_label=self.positive_label,
                zero_division=0,
            )
        )
        return {
            "acc": float(accuracy_score(true_labels, predictions)),
            "recall": recall,
            "sn": recall,
            "specificity": float(specificity),
            "sp": float(specificity),
            "precision": precision,
            "pre": precision,
            "f1": float(f1_score(true_labels, predictions, zero_division=0)),
            "mcc": float(matthews_corrcoef(true_labels, predictions)),
            "auc": auc,
            "aupr": aupr,
            "confusion_matrix": confusion.tolist(),
            "class_accuracy": class_accuracy,
        }

    def _compute_multiclass_metrics(self, true_labels, predictions, probabilities):
        labels = list(range(self.num_classes))
        confusion = confusion_matrix(true_labels, predictions, labels=labels)
        total = confusion.sum()
        specificities = []
        class_accuracy = {}
        for class_index in labels:
            true_positive = confusion[class_index, class_index]
            false_positive = confusion[:, class_index].sum() - true_positive
            false_negative = confusion[class_index, :].sum() - true_positive
            true_negative = total - true_positive - false_positive - false_negative
            specificities.append(
                true_negative / (true_negative + false_positive)
                if true_negative + false_positive > 0
                else 0.0
            )
            class_total = confusion[class_index, :].sum()
            class_accuracy[class_index] = float(
                true_positive / class_total if class_total > 0 else 0.0
            )

        try:
            auc = float(
                roc_auc_score(
                    true_labels,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            auc = 0.0
        try:
            aupr = float(
                average_precision_score(true_labels, probabilities, average="macro")
            )
        except ValueError:
            aupr = 0.0

        recall = float(
            recall_score(true_labels, predictions, average="macro", zero_division=0)
        )
        precision = float(
            precision_score(
                true_labels, predictions, average="macro", zero_division=0
            )
        )
        specificity = float(np.mean(specificities))
        return {
            "acc": float(accuracy_score(true_labels, predictions)),
            "recall": recall,
            "sn": recall,
            "specificity": specificity,
            "sp": specificity,
            "precision": precision,
            "pre": precision,
            "f1": float(
                f1_score(true_labels, predictions, average="macro", zero_division=0)
            ),
            "mcc": float(matthews_corrcoef(true_labels, predictions)),
            "auc": auc,
            "aupr": aupr,
            "confusion_matrix": confusion.tolist(),
            "class_accuracy": class_accuracy,
        }

    def compute(self):
        if not self.logits or not self.targets:
            return {
                "acc": 0.0,
                "recall": 0.0,
                "sn": 0.0,
                "specificity": 0.0,
                "sp": 0.0,
                "precision": 0.0,
                "pre": 0.0,
                "f1": 0.0,
                "mcc": 0.0,
                "auc": 0.0,
                "aupr": 0.0,
                "confusion_matrix": [],
                "class_accuracy": {},
            }

        logits = torch.cat(self.logits)
        true_labels = torch.cat(self.targets).numpy()
        probabilities = F.softmax(logits, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
        if self.num_classes == 2:
            metrics = self._compute_binary_metrics(
                true_labels, predictions, probabilities[:, 1]
            )
        else:
            metrics = self._compute_multiclass_metrics(
                true_labels, predictions, probabilities
            )
        self.reset()
        return metrics
