"""m5C-SSNet model for interpretable RNA m5C site prediction."""

from collections import OrderedDict
import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW

from models.fusion import AdaptiveGatedFusion
from models.sequence_encoder import SpliceBERTSequenceEncoder
from models.structure_encoder import StructuralContextEncoder
from utils.config import PRETRAINED_MODEL_PATH
from utils.metrics import ClassificationMetric
from utils.runtime import _rank_zero_print

class M5CSSNet(pl.LightningModule):
    """Integrate RNA sequence semantics and structural context for m5C prediction."""

    def __init__(
        self,
        sequence_length=41,
        representation_dim=256,
        classifier_hidden_dim=256,
        num_classes=2,
        classifier_dropout=0.6,
        learning_rate=1e-4,
        gine_hidden_dim=64,
        gine_dropout=0.2,
        num_gine_layers=2,
        structure_cnn_channels=64,
        structure_cnn_dropout=0.2,
        structure_cnn_kernel_size=5,
        bppm_top_k=2,
        minimum_pairing_probability=0.01,
        splicebert_max_length=256,
        pretrained_model_path=PRETRAINED_MODEL_PATH,
        **legacy_parameters,
    ):
        super().__init__()
        supported_legacy_parameters = {
            "RNA_type", "ablation_mode", "cnn_channels", "cnn_dropout",
            "dropout", "evo_dim", "fusion_dim", "fusion_dropout",
            "graph_dropout", "graph_hidden_dim", "graph_kernel_size",
            "graph_layers", "hidden_dim", "max_length", "ncp_dim",
            "ncp_input_len", "seqcnn_dim", "seqcnn_input_len",
            "splicebert_path",
        }
        unknown_parameters = set(legacy_parameters) - supported_legacy_parameters
        if unknown_parameters:
            names = ", ".join(sorted(unknown_parameters))
            raise TypeError(f"Unexpected model parameters: {names}")

        sequence_length = legacy_parameters.get("seqcnn_input_len", sequence_length)
        representation_dim = legacy_parameters.get("fusion_dim", representation_dim)
        classifier_hidden_dim = legacy_parameters.get("hidden_dim", classifier_hidden_dim)
        classifier_dropout = legacy_parameters.get("dropout", classifier_dropout)
        gine_hidden_dim = legacy_parameters.get("graph_hidden_dim", gine_hidden_dim)
        gine_dropout = legacy_parameters.get("graph_dropout", gine_dropout)
        num_gine_layers = legacy_parameters.get("graph_layers", num_gine_layers)
        structure_cnn_channels = legacy_parameters.get("cnn_channels", structure_cnn_channels)
        structure_cnn_dropout = legacy_parameters.get("cnn_dropout", structure_cnn_dropout)
        structure_cnn_kernel_size = legacy_parameters.get(
            "graph_kernel_size", structure_cnn_kernel_size
        )
        splicebert_max_length = legacy_parameters.get("max_length", splicebert_max_length)
        legacy_model_path = legacy_parameters.get("splicebert_path")
        if legacy_model_path and not os.path.isdir(pretrained_model_path):
            pretrained_model_path = legacy_model_path
        self.save_hyperparameters(ignore=["legacy_parameters"])

        self.sequence_encoder = SpliceBERTSequenceEncoder(
            pretrained_model_path=pretrained_model_path,
            representation_dim=representation_dim,
            max_length=splicebert_max_length,
        )
        self.structural_context_encoder = StructuralContextEncoder(
            sequence_length=sequence_length,
            gine_hidden_dim=gine_hidden_dim,
            representation_dim=representation_dim,
            gine_dropout=gine_dropout,
            num_gine_layers=num_gine_layers,
            structure_cnn_channels=structure_cnn_channels,
            structure_cnn_dropout=structure_cnn_dropout,
            structure_cnn_kernel_size=structure_cnn_kernel_size,
            bppm_top_k=bppm_top_k,
            minimum_pairing_probability=minimum_pairing_probability,
        )
        self.adaptive_gated_fusion = AdaptiveGatedFusion(
            representation_dim=representation_dim,
            gate_hidden_dim=classifier_hidden_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(representation_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        self.register_buffer("gate_weight_sum", torch.zeros(2), persistent=False)
        self.register_buffer("gate_weight_count", torch.zeros(1), persistent=False)
        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics = ClassificationMetric(num_classes=num_classes)
        self.val_metrics = ClassificationMetric(num_classes=num_classes)
        self.test_metrics = ClassificationMetric(num_classes=num_classes)
        metric_names = [
            "loss", "acc", "auc", "aupr", "mcc", "f1",
            "precision", "recall", "specificity",
        ]
        self.history = {name: [] for name in metric_names}
        self.history.update({f"val_{name}": [] for name in metric_names})
        self.train_batch_outputs = []
        self.val_batch_outputs = []
        self.test_batch_outputs = []

    @staticmethod
    def _upgrade_legacy_state_dict(state_dict):
        """Map historical parameter prefixes to the public m5C-SSNet names."""
        prefix_map = {
            "splicebert.": "sequence_encoder.splicebert.",
            "evo_reducer.": "sequence_encoder.sequence_projection.",
            "graph_local_branch.": "structural_context_encoder.",
            "gate_network.": "adaptive_gated_fusion.gate_network.",
        }
        upgraded = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in prefix_map.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            new_key = new_key.replace(
                ".chain1_edge_embedding", ".sequence_edge_embedding"
            )
            upgraded[new_key] = value
        if hasattr(state_dict, "_metadata"):
            upgraded._metadata = state_dict._metadata
        return upgraded

    def load_state_dict(self, state_dict, strict=True, assign=False):
        upgraded = self._upgrade_legacy_state_dict(state_dict)
        saved_class_weights = upgraded.get("criterion.weight")
        if saved_class_weights is not None and self.criterion.weight is None:
            self.criterion = nn.CrossEntropyLoss(
                weight=saved_class_weights.detach().clone()
            )
        try:
            return super().load_state_dict(upgraded, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(upgraded, strict=strict)

    def _encode_and_fuse(self, batch):
        sequence_representation = self.sequence_encoder(batch["sequence"])
        structural_representation = self.structural_context_encoder(
            batch["sequence"], bppm=batch.get("bppm")
        )
        fused_representation, fusion_weights = self.adaptive_gated_fusion(
            sequence_representation, structural_representation
        )
        return (
            sequence_representation, structural_representation,
            fused_representation, fusion_weights,
        )

    def forward(self, batch):
        _, _, fused_representation, fusion_weights = self._encode_and_fuse(batch)
        with torch.no_grad():
            self.gate_weight_sum += fusion_weights.detach().sum(dim=0)
            self.gate_weight_count += fusion_weights.size(0)
        return self.classifier(fused_representation)

    @torch.no_grad()
    def extract_embeddings(self, batch):
        was_training = self.training
        self.eval()
        sequence_representation, structural_representation, fused_representation, fusion_weights = (
            self._encode_and_fuse(batch)
        )
        logits = self.classifier(fused_representation)
        if was_training:
            self.train()
        return {
            "sequence_representation": sequence_representation,
            "structural_representation": structural_representation,
            "fusion_weights": fusion_weights,
            "fused_representation": fused_representation,
            "logits": logits,
        }

    def _reset_gate_statistics(self):
        self.gate_weight_sum.zero_()
        self.gate_weight_count.zero_()

    def set_class_weights(self, train_labels):
        """Set mildly upweighted cross-entropy loss for an imbalanced class."""
        labels = np.asarray(train_labels, dtype=np.int64)
        num_classes = int(self.hparams.num_classes)
        if labels.size == 0:
            raise ValueError("Training labels cannot be empty.")
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

        class_counts = np.bincount(labels, minlength=num_classes)
        total_samples = int(labels.size)
        _rank_zero_print("\n=== Class distribution ===")
        for class_index, count in enumerate(class_counts):
            percentage = count / total_samples * 100
            _rank_zero_print(
                f"Class {class_index}: {count} samples ({percentage:.2f}%)"
            )

        weights = np.ones(num_classes, dtype=np.float32)
        present_counts = class_counts[class_counts > 0]
        if present_counts.size < num_classes:
            _rank_zero_print(
                "Warning: at least one class is absent from the training set; "
                "class weights remain unchanged."
            )
        else:
            majority_class = int(np.argmax(class_counts))
            minority_class = int(np.argmin(class_counts))
            imbalance_ratio = float(class_counts[majority_class] / class_counts[minority_class])
            _rank_zero_print(
                f"Majority-to-minority ratio: {imbalance_ratio:.2f}"
            )
            if imbalance_ratio > 1.3:
                weights[minority_class] = min(
                    np.sqrt(imbalance_ratio) * 1.5, 2.5
                )

        class_weights = torch.tensor(
            weights, dtype=torch.float32, device=self.device
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        _rank_zero_print("Class weights: " + ", ".join(
            f"{index}={weight:.4f}" for index, weight in enumerate(weights)
        ))
        return class_weights

    def _sync_epoch_scalar(self, value):
        if getattr(self.trainer, 'world_size', 1) <= 1:
            return value
        gathered = self.all_gather(value.detach())
        return gathered.float().mean()

    def _gather_epoch_tensors(self, logits, labels):
        if getattr(self.trainer, 'world_size', 1) <= 1:
            return logits, labels

        gathered_logits = self.all_gather(logits.detach())
        gathered_logits = gathered_logits.reshape(-1, gathered_logits.size(-1))

        gathered_labels = self.all_gather(labels.detach())
        if labels.dim() > 1:
            gathered_labels = gathered_labels.reshape(-1, labels.size(-1))
        else:
            gathered_labels = gathered_labels.reshape(-1)

        return gathered_logits, gathered_labels

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.criterion(logits, batch['label'])

        self.train_batch_outputs.append({
            'loss': loss.detach(),
            'logits': logits.detach(),
            'labels': batch['label'].detach()
        })

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.criterion(logits, batch['label'])

        self.val_batch_outputs.append({
            'loss': loss.detach(),
            'logits': logits.detach(),
            'labels': batch['label'].detach()
        })

        preds = torch.argmax(logits, dim=1)
        if (
            batch_idx == 0
            and not self.trainer.sanity_checking
            and getattr(self.trainer, "is_global_zero", True)
        ):
            label_counts = torch.bincount(batch['label'], minlength=self.hparams.num_classes)
            pred_counts = torch.bincount(preds, minlength=self.hparams.num_classes)
            _rank_zero_print(f"Validation label distribution: {label_counts.cpu().numpy()}")
            _rank_zero_print(f"Validation prediction distribution: {pred_counts.cpu().numpy()}")

        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.criterion(logits, batch['label'])

        self.test_batch_outputs.append({
            'loss': loss.detach(),
            'logits': logits.detach(),
            'labels': batch['label'].detach()
        })

        preds = torch.argmax(logits, dim=1)
        if (
            batch_idx == 0
            and not self.trainer.sanity_checking
            and getattr(self.trainer, "is_global_zero", True)
        ):
            label_counts = torch.bincount(batch['label'], minlength=self.hparams.num_classes)
            pred_counts = torch.bincount(preds, minlength=self.hparams.num_classes)
            _rank_zero_print(f"Test label distribution: {label_counts.cpu().numpy()}")
            _rank_zero_print(f"Test prediction distribution: {pred_counts.cpu().numpy()}")

        return loss

    def on_train_epoch_end(self):

        if not self.train_batch_outputs:
            return

        train_loss = torch.stack([x['loss'] for x in self.train_batch_outputs]).mean()
        train_loss = self._sync_epoch_scalar(train_loss)

        all_logits = torch.cat([x['logits'] for x in self.train_batch_outputs])
        all_labels = torch.cat([x['labels'] for x in self.train_batch_outputs])
        all_logits, all_labels = self._gather_epoch_tensors(all_logits, all_labels)

        self.train_metrics.update(all_logits, all_labels)

        metrics = self.train_metrics.compute()
        train_acc = metrics['acc']
        train_auc = metrics['auc']
        train_aupr = metrics['aupr']
        train_mcc = metrics['mcc']
        train_recall = metrics['recall']
        train_sp = metrics['specificity']
        train_prec = metrics['precision']
        train_f1 = metrics['f1']

        self.history['loss'].append(train_loss.item())
        self.history['acc'].append(train_acc)
        self.history['auc'].append(train_auc)
        self.history['aupr'].append(train_aupr)
        self.history['mcc'].append(train_mcc)
        self.history['recall'].append(train_recall)
        self.history['specificity'].append(train_sp)
        self.history['precision'].append(train_prec)
        self.history['f1'].append(train_f1)

        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_auc', train_auc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_aupr', train_aupr, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_mcc', train_mcc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_f1', train_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_precision', train_prec, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_pre', train_prec, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_recall', train_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_sn', train_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_specificity', train_sp, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_sp', train_sp, on_step=False, on_epoch=True, sync_dist=True)

        self.train_batch_outputs.clear()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_batch_outputs.clear()
            self.val_metrics.reset()
            return

        if not self.val_batch_outputs:
            return

        val_loss = torch.stack([x['loss'] for x in self.val_batch_outputs]).mean()
        val_loss = self._sync_epoch_scalar(val_loss)

        all_logits = torch.cat([x['logits'] for x in self.val_batch_outputs])
        all_labels = torch.cat([x['labels'] for x in self.val_batch_outputs])
        all_logits, all_labels = self._gather_epoch_tensors(all_logits, all_labels)

        self.val_metrics.update(all_logits, all_labels)

        metrics = self.val_metrics.compute()
        val_acc = metrics['acc']
        val_auc = metrics['auc']
        val_aupr = metrics['aupr']
        val_mcc = metrics['mcc']
        val_recall = metrics['recall']
        val_sp = metrics['specificity']
        val_prec = metrics['precision']
        val_f1 = metrics['f1']

        self.history['val_loss'].append(val_loss.item())
        self.history['val_acc'].append(val_acc)
        self.history['val_auc'].append(val_auc)
        self.history['val_aupr'].append(val_aupr)
        self.history['val_mcc'].append(val_mcc)
        self.history['val_recall'].append(val_recall)
        self.history['val_specificity'].append(val_sp)
        self.history['val_precision'].append(val_prec)
        self.history['val_f1'].append(val_f1)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_auc', val_auc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_aupr', val_aupr, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mcc', val_mcc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_f1', val_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_precision', val_prec, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_pre', val_prec, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_recall', val_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_sn', val_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_specificity', val_sp, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_sp', val_sp, on_step=False, on_epoch=True, sync_dist=True)

        current_epoch = self.current_epoch
        if getattr(self.trainer, 'is_global_zero', True):
            _rank_zero_print(
                f"Epoch {current_epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f} | Val MCC: {val_mcc:.4f}"
            )

        self.val_batch_outputs.clear()

        self.val_metrics.reset()

    def on_test_start(self):
        """Reset fusion-gate statistics before test evaluation."""
        self._reset_gate_statistics()

    def on_test_epoch_end(self):

        if not self.test_batch_outputs:
            return

        test_loss = torch.stack([x['loss'] for x in self.test_batch_outputs]).mean()
        test_loss = self._sync_epoch_scalar(test_loss)

        all_logits = torch.cat([x['logits'] for x in self.test_batch_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_batch_outputs])
        all_logits, all_labels = self._gather_epoch_tensors(all_logits, all_labels)

        self.test_metrics.update(all_logits, all_labels)

        metrics = self.test_metrics.compute()
        test_acc = metrics['acc']
        test_auc = metrics['auc']
        test_aupr = metrics['aupr']
        test_mcc = metrics['mcc']
        test_recall = metrics['recall']
        test_sp = metrics['specificity']
        test_prec = metrics['precision']
        test_f1 = metrics['f1']

        confusion = metrics['confusion_matrix']

        self.log('test_loss', test_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc', test_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_auc', test_auc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_aupr', test_aupr, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_mcc', test_mcc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_f1', test_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_precision', test_prec, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_pre', test_prec, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_recall', test_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_sn', test_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_specificity', test_sp, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_sp', test_sp, on_step=False, on_epoch=True, sync_dist=True)

        test_results = {
            'test_loss': test_loss.item(),
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_aupr': test_aupr,
            'test_mcc': test_mcc,
            'test_f1': test_f1,
            'test_precision': test_prec,
            'test_pre': test_prec,
            'test_recall': test_recall,
            'test_sn': test_recall,
            'test_specificity': test_sp,
            'test_sp': test_sp,
            'confusion_matrix': confusion
        }

        test_results['class_accuracy'] = metrics['class_accuracy']

        fusion_weights = self.get_fusion_weights()
        test_results['fusion_weights'] = fusion_weights

        self.test_batch_outputs.clear()

        self.test_metrics.reset()

        self.test_results = test_results

    def configure_optimizers(self):
        projection_params = []
        task_specific_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (
                'classifier' in name or
                'structural_context_encoder' in name or
                'adaptive_gated_fusion' in name
            ):
                task_specific_params.append(param)
            else:
                projection_params.append(param)

        optimizer = AdamW([
            {'params': projection_params, 'lr': self.hparams.learning_rate / 10},
            {'params': task_specific_params, 'lr': self.hparams.learning_rate}
        ], weight_decay=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def get_fusion_weights(self):
        """Return mean sample-wise gate weights collected during evaluation."""
        return {
            "fusion_type": "adaptive_gated_fusion",
            "branch_weights": {
                "sequence_semantics": (self.gate_weight_sum[0] / self.gate_weight_count.clamp_min(1)).item(),
                "structural_context": (self.gate_weight_sum[1] / self.gate_weight_count.clamp_min(1)).item(),
            },
        }

    @property
    def device(self):
        return next(self.parameters()).device

