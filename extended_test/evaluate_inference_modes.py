from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    CHECKPOINT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CALIBRATION_BATCHES,
    DEFAULT_FORWARD_BITS,
    DEFAULT_INPUT_NOISE_STD,
    MODEL_ARCHS,
    TRAINING_MODES,
    build_checkpoint_path,
    get_device,
    load_checkpoint,
    load_compatible_state_dict,
    set_seed,
    write_json,
)
from data_utils import build_cifar10_loaders
from models import (
    build_model,
    disable_quant_measure_calibration,
    enable_quant_measure_calibration,
)
from quantized_pytorch_master.models.modules.quantize import set_gradient_noise_std


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def calibrate_hardware_model(model, loader, device, max_batches):
    enable_quant_measure_calibration(model)
    for batch_index, (inputs, _labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        model(inputs.to(device))
    disable_quant_measure_calibration(model)


def evaluate_checkpoint(
    checkpoint_path: Path,
    software_testloader,
    hardware_trainloader,
    hardware_testloader,
    device,
    num_bits: int,
    calibration_batches: int,
):
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    arch = checkpoint["arch"]
    training_mode = checkpoint["training_mode"]
    state_dict = checkpoint["state_dict"]

    software_model = build_model(arch=arch, hardware_aware=False, num_classes=10).to(device)
    software_load_info = load_compatible_state_dict(software_model, state_dict)
    software_acc = evaluate(software_model, software_testloader, device)

    hardware_model = build_model(
        arch=arch,
        hardware_aware=True,
        num_classes=10,
        num_bits=num_bits,
        num_bits_grad=None,
    ).to(device)
    hardware_load_info = load_compatible_state_dict(hardware_model, state_dict)
    if training_mode == "software":
        calibrate_hardware_model(
            hardware_model,
            hardware_trainloader,
            device,
            max_batches=calibration_batches,
        )
    hardware_acc = evaluate(hardware_model, hardware_testloader, device)

    return {
        "checkpoint": str(checkpoint_path),
        "arch": arch,
        "training_mode": training_mode,
        "software_inference_accuracy": software_acc,
        "hardware_inference_accuracy": hardware_acc,
        "accuracy_gap": hardware_acc - software_acc,
        "software_load_missing_keys": len(software_load_info["missing"]),
        "hardware_load_missing_keys": len(hardware_load_info["missing"]),
        "hardware_load_skipped_keys": len(hardware_load_info["skipped"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare software and hardware inference on all trained CIFAR-10 models.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--input-noise-std", type=float, default=DEFAULT_INPUT_NOISE_STD)
    parser.add_argument("--num-bits", type=int, default=DEFAULT_FORWARD_BITS)
    parser.add_argument("--calibration-batches", type=int, default=DEFAULT_CALIBRATION_BATCHES)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    set_gradient_noise_std(0.0)

    software_trainloader, software_testloader = build_cifar10_loaders(
        batch_size=args.batch_size,
        hardware_train=False,
        hardware_test=False,
    )
    hardware_trainloader, hardware_testloader = build_cifar10_loaders(
        batch_size=args.batch_size,
        hardware_train=True,
        hardware_test=True,
        input_noise_std=args.input_noise_std,
    )

    results = []
    for training_mode in TRAINING_MODES:
        for arch in MODEL_ARCHS:
            checkpoint_path = build_checkpoint_path(arch, training_mode)
            if not checkpoint_path.exists():
                print(f"[skip] missing checkpoint: {checkpoint_path}")
                continue

            result = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                software_testloader=software_testloader,
                hardware_trainloader=hardware_trainloader,
                hardware_testloader=hardware_testloader,
                device=device,
                num_bits=args.num_bits,
                calibration_batches=args.calibration_batches,
            )
            results.append(result)
            print(
                f"[{result['training_mode']}][{result['arch']}] "
                f"software_acc={result['software_inference_accuracy']:.4f} "
                f"hardware_acc={result['hardware_inference_accuracy']:.4f} "
                f"gap={result['accuracy_gap']:.4f}"
            )

    summary = {
        "results": results,
        "evaluated_checkpoints": len(results),
    }
    summary_path = CHECKPOINT_DIR / "inference_comparison.json"
    write_json(summary_path, summary)
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
