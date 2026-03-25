from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 30
DEFAULT_LR = 0.1
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_MOMENTUM = 0.9
DEFAULT_INPUT_NOISE_STD = 0.1
DEFAULT_FORWARD_BITS = 5
DEFAULT_GRAD_BITS = 5
DEFAULT_GRADIENT_NOISE_STD = 0.1
DEFAULT_CALIBRATION_BATCHES = 16

MODEL_ARCHS = ("resnet18", "vgg11")
TRAINING_MODES = ("software", "hardware")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def build_checkpoint_name(arch: str, training_mode: str) -> str:
    return f"{training_mode}_{arch}_cifar10.pth"


def build_checkpoint_path(arch: str, training_mode: str) -> Path:
    return CHECKPOINT_DIR / build_checkpoint_name(arch, training_mode)


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state)

    missing_keys = [key for key in model.state_dict().keys() if key not in loaded_keys]
    return {
        "loaded": loaded_keys,
        "missing": missing_keys,
        "skipped": skipped_keys,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
