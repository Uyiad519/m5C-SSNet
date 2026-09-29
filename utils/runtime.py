"""Runtime, distributed-training, and tensor utilities."""

import fcntl
import os
import re
from datetime import datetime

import torch

def _parse_gpu_devices(gpu_devices):
    """Normalize GPU device settings for PyTorch Lightning."""
    if gpu_devices is None:
        return None

    if isinstance(gpu_devices, int):
        if gpu_devices < 0:
            raise ValueError("gpu_devices cannot be negative")
        return [gpu_devices]

    if isinstance(gpu_devices, str):
        value = gpu_devices.strip().replace("，", ",")
        if not value:
            return None
        if value.lower() == "auto":
            return "auto"
        gpu_devices = [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(gpu_devices, (list, tuple, set)):
        parsed_devices = []
        seen = set()
        for item in gpu_devices:
            if isinstance(item, str):
                item = item.strip()
                if not item:
                    continue
                item = int(item)
            if not isinstance(item, int) or item < 0:
                raise ValueError(f"Invalid gpu_devices setting: {gpu_devices}")
            if item not in seen:
                parsed_devices.append(item)
                seen.add(item)
        return parsed_devices or None

    raise TypeError(
        "gpu_devices must be an int, string, list, tuple, or set; "
        f"received {type(gpu_devices).__name__}"
    )

def _build_trainer_device_config(config, stage="train"):
    """Build stage-specific PyTorch Lightning device settings."""
    accelerator = str(config.get('accelerator', 'auto')).lower()
    if accelerator == 'auto':
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    if accelerator == 'cuda':
        accelerator = 'gpu'

    precision = config.get(
        'precision',
        '16-mixed' if accelerator == 'gpu' and torch.cuda.is_available() else '32-true'
    )

    trainer_device_config = {
        'accelerator': accelerator,
        'precision': precision
    }

    if accelerator == 'gpu':
        if not torch.cuda.is_available():
            _rank_zero_print("CUDA is unavailable; falling back to CPU.")
            return {
                'accelerator': 'cpu',
                'devices': 1,
                'precision': '32-true'
            }

        gpu_key = 'evaluation_gpu_devices' if stage != "train" and config.get('evaluation_gpu_devices') is not None else 'gpu_devices'
        devices = _parse_gpu_devices(config.get(gpu_key, 0))
        trainer_device_config['devices'] = devices if devices is not None else [0]

        uses_multi_gpu = (
            trainer_device_config['devices'] == 'auto' or
            (isinstance(trainer_device_config['devices'], list) and len(trainer_device_config['devices']) > 1)
        )

        strategy = config.get('strategy', 'auto')
        if strategy and strategy != 'auto':
            if uses_multi_gpu or not str(strategy).startswith('ddp'):
                trainer_device_config['strategy'] = strategy
            else:
                _rank_zero_print(f"Ignoring multi-GPU strategy for a single-device configuration: {strategy}")
        elif uses_multi_gpu:
            trainer_device_config['strategy'] = 'ddp'

        return trainer_device_config

    trainer_device_config['devices'] = int(config.get('cpu_devices', 1))
    if trainer_device_config['precision'] == '16-mixed':
        trainer_device_config['precision'] = '32-true'
    return trainer_device_config

def _format_trainer_device_config(trainer_device_config):
    parts = [
        f"accelerator={trainer_device_config.get('accelerator')}",
        f"devices={trainer_device_config.get('devices')}",
        f"precision={trainer_device_config.get('precision')}"
    ]
    if 'strategy' in trainer_device_config:
        parts.append(f"strategy={trainer_device_config['strategy']}")
    return ", ".join(parts)

def _get_process_rank():
    for key in ("RANK", "GLOBAL_RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0

def _is_global_zero_process():
    return _get_process_rank() == 0

def _rank_zero_print(*args, **kwargs):
    if _is_global_zero_process():
        print(*args, **kwargs)

def _get_distributed_run_token():
    existing = os.environ.get("M5C_SSNET_RUN_TOKEN")
    if existing:
        return existing

    suffix = "single"
    for key in ("TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "MASTER_PORT"):
        value = os.environ.get(key)
        if value:
            suffix = str(value)
            break

    token = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
    os.environ["M5C_SSNET_RUN_TOKEN"] = token
    return token

def _next_csv_logger_version(logger_root):
    max_version = -1
    if os.path.isdir(logger_root):
        for entry in os.listdir(logger_root):
            match = re.fullmatch(r"version_(\d+)", entry)
            if match:
                max_version = max(max_version, int(match.group(1)))
    return max_version + 1

def _allocate_shared_csv_logger_version(save_dir, name, configured_version=None):
    if configured_version not in (None, "auto"):
        return configured_version, None

    logger_root = os.path.join(save_dir, name)
    os.makedirs(logger_root, exist_ok=True)

    run_token = _get_distributed_run_token()
    state_path = os.path.join(logger_root, f".version_state_{run_token}")

    with open(state_path, "a+", encoding="utf-8") as state_file:
        fcntl.flock(state_file.fileno(), fcntl.LOCK_EX)
        state_file.seek(0)
        current_value = state_file.read().strip()
        if current_value:
            try:
                version = int(current_value)
            except ValueError:
                version = current_value
        else:
            version = _next_csv_logger_version(logger_root)
            state_file.seek(0)
            state_file.truncate()
            state_file.write(str(version))
            state_file.flush()
            os.fsync(state_file.fileno())
        fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)

    return version, state_path

def _cleanup_shared_csv_logger_version(state_path):
    if not state_path or not _is_global_zero_process():
        return
    try:
        os.remove(state_path)
    except FileNotFoundError:
        pass

def _strip_special_token_features(token_features, attention_mask, special_tokens_mask):
    """Remove special-token positions while preserving padded batch structure."""
    if attention_mask is None:
        attention_mask = torch.ones(
            token_features.shape[:2],
            dtype=torch.bool,
            device=token_features.device
        )
    else:
        attention_mask = attention_mask.bool()

    if special_tokens_mask is None:
        special_tokens_mask = torch.zeros_like(attention_mask)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    valid_mask = attention_mask & (~special_tokens_mask)
    valid_counts = valid_mask.sum(dim=1)
    max_valid_len = int(valid_counts.max().item()) if valid_counts.numel() > 0 else 0

    batch_size, _, hidden_dim = token_features.shape
    if max_valid_len <= 0:
        return (
            token_features.new_zeros((batch_size, 1, hidden_dim)),
            torch.zeros((batch_size, 1), dtype=torch.bool, device=token_features.device)
        )

    compact_features = token_features.new_zeros((batch_size, max_valid_len, hidden_dim))
    compact_mask = torch.zeros((batch_size, max_valid_len), dtype=torch.bool, device=token_features.device)

    for row_idx in range(batch_size):
        row_features = token_features[row_idx][valid_mask[row_idx]]
        row_len = row_features.size(0)
        if row_len == 0:
            continue
        compact_features[row_idx, :row_len] = row_features
        compact_mask[row_idx, :row_len] = True

    return compact_features, compact_mask

def _masked_mean_pool(token_features, attention_mask):
    """Mean-pool valid token representations."""
    if attention_mask is None:
        return torch.mean(token_features, dim=1)

    valid_mask = attention_mask.unsqueeze(-1).to(token_features.dtype)
    return (token_features * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1e-8)

def _extract_center_window(sequence, target_len=41):
    """Crop or pad a sequence to a centered fixed-length window."""
    seq = str(sequence).upper().replace("U", "T")
    target_len = int(target_len)
    if len(seq) >= target_len:
        start = (len(seq) - target_len) // 2
        end = start + target_len
        return seq[start:end]
    pad_total = target_len - len(seq)
    left_pad = pad_total // 2
    right_pad = pad_total - left_pad
    return ("N" * left_pad) + seq + ("N" * right_pad)
