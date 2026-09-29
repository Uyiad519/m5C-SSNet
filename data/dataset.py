"""Dataset loading for cytosine-centered RNA fragments."""

import os
import re

import numpy as np
from torch.utils.data import Dataset

from utils.bppm import load_or_build_bppm_cache
from utils.runtime import _rank_zero_print

class RNADataset(Dataset):
    """Load labeled RNA fragments and their precomputed structural matrices."""

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
        self.sequences = []
        self.labels = []
        self.bppm_matrices = None

        self._read_fasta()
        if not self.labels:
            raise ValueError(f"No labeled sequences were found in {file_path}.")
        self.length_stats = self._analyze_sequence_lengths()
        self._load_bppm()

    def _read_fasta(self):
        _rank_zero_print(f"Reading FASTA file: {self.file_path}")
        current_sequence = []
        current_label = None

        with open(self.file_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_sequence and current_label is not None:
                        self.sequences.append("".join(current_sequence))
                        self.labels.append(current_label)
                    header = line[1:]
                    label_match = re.search(
                        r"(?<![A-Za-z0-9_])label\s*=\s*([01])(?!\d)",
                        header,
                        flags=re.IGNORECASE,
                    )
                    if label_match is None:
                        raise ValueError(
                            f"FASTA header does not contain label=0 or label=1: {header}"
                        )
                    current_label = int(label_match.group(1))
                    current_sequence = []
                else:
                    if current_label is None:
                        raise ValueError("A FASTA sequence was found before its header.")
                    current_sequence.append(line.upper().replace("U", "T"))

        if current_sequence and current_label is not None:
            self.sequences.append("".join(current_sequence))
            self.labels.append(current_label)

        label_counts = np.bincount(self.labels, minlength=2)
        _rank_zero_print(f"Loaded {len(self.sequences)} sequences")
        for label, count in enumerate(label_counts):
            _rank_zero_print(f"Class {label}: {count} sequences")

    def _load_bppm(self):
        if not self.use_bppm_cache:
            _rank_zero_print(
                "BPPM input is disabled; the structural encoder will use sequence edges only."
            )
            return
        cache_dir = self.bppm_cache_dir or os.path.join("data", "bppm_cache")
        self.bppm_matrices = load_or_build_bppm_cache(
            self.sequences,
            dataset_path=self.file_path,
            cache_dir=cache_dir,
            input_length=self.bppm_input_len,
            log=_rank_zero_print,
        )

    def _analyze_sequence_lengths(self):
        lengths = [len(sequence) for sequence in self.sequences]
        lengths_by_class = {0: [], 1: []}
        for sequence, label in zip(self.sequences, self.labels):
            lengths_by_class[label].append(len(sequence))

        stats = {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "percentiles": {
                percentile: int(np.percentile(lengths, percentile))
                for percentile in (10, 25, 50, 75, 90)
            },
            "length_by_class": lengths_by_class,
        }
        _rank_zero_print("\n=== Sequence length statistics ===")
        _rank_zero_print(f"Minimum: {stats['min_length']} nt")
        _rank_zero_print(f"Maximum: {stats['max_length']} nt")
        _rank_zero_print(f"Mean: {stats['avg_length']:.2f} nt")
        return stats

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        item = {
            "sequence": self.sequences[index],
            "label": self.labels[index],
        }
        if self.bppm_matrices is not None:
            item["bppm"] = self.bppm_matrices[index]
        return item

    def get_length_stats(self):
        """Return sequence-length summary statistics."""
        return self.length_stats
