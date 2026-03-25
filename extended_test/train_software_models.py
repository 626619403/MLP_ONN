from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_WEIGHT_DECAY,
    MODEL_ARCHS,
    build_checkpoint_path,
    get_device,
    save_checkpoint,
    set_seed,
)
from data_utils import build_cifar10_loaders
from models import build_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        running_total += labels.size(0)

    return running_loss / running_total, running_correct / running_total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_correct = 0
    running_total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        running_total += labels.size(0)

    return running_correct / running_total


def train_software_model(args, arch: str) -> Path:
    device = get_device()
    trainloader, testloader = build_cifar10_loaders(
        batch_size=args.batch_size,
        hardware_train=False,
        hardware_test=False,
    )

    model = build_model(
        arch=arch,
        hardware_aware=False,
        num_classes=10,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    checkpoint_path = build_checkpoint_path(arch, "software")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        test_acc = evaluate(model, testloader, device)
        scheduler.step()

        if test_acc >= best_acc:
            best_acc = test_acc
            save_checkpoint(
                checkpoint_path,
                {
                    "arch": arch,
                    "training_mode": "software",
                    "num_classes": 10,
                    "state_dict": model.state_dict(),
                    "best_test_accuracy": best_acc,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                },
            )

        print(
            f"[software][{arch}] epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    print(f"[software][{arch}] best_test_acc={best_acc:.4f} saved_to={checkpoint_path}")
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train pure software CIFAR-10 models.")
    parser.add_argument("--arch", choices=[*MODEL_ARCHS, "all"], default="all")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    archs = MODEL_ARCHS if args.arch == "all" else (args.arch,)
    for arch in archs:
        train_software_model(args, arch)


if __name__ == "__main__":
    main()
