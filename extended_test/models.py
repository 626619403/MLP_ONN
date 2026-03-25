from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg11

from quantized_pytorch_master.models.modules.quantize import QConv2d, QLinear, QuantMeasure


def _convert_conv(module: nn.Conv2d, num_bits: int, num_bits_grad: int | None) -> QConv2d:
    new_module = QConv2d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        num_bits=num_bits,
        num_bits_weight=num_bits,
        num_bits_grad=num_bits_grad,
    )
    new_module.weight.data.copy_(module.weight.data)
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)
    return new_module


def _convert_linear(module: nn.Linear, num_bits: int, num_bits_grad: int | None) -> QLinear:
    new_module = QLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        num_bits=num_bits,
        num_bits_weight=num_bits,
        num_bits_grad=num_bits_grad,
    )
    new_module.weight.data.copy_(module.weight.data)
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)
    return new_module


def convert_to_hardware_layers(module: nn.Module, num_bits: int, num_bits_grad: int | None) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, _convert_conv(child, num_bits=num_bits, num_bits_grad=num_bits_grad))
        elif isinstance(child, nn.Linear):
            setattr(module, name, _convert_linear(child, num_bits=num_bits, num_bits_grad=num_bits_grad))
        else:
            convert_to_hardware_layers(child, num_bits=num_bits, num_bits_grad=num_bits_grad)
    return module


def build_resnet18(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vgg11(num_classes: int = 10) -> nn.Module:
    model = vgg11(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def build_model(
    arch: str,
    hardware_aware: bool = False,
    num_classes: int = 10,
    num_bits: int = 5,
    num_bits_grad: int | None = 5,
) -> nn.Module:
    builders: dict[str, Type[nn.Module] | callable] = {
        "resnet18": build_resnet18,
        "vgg11": build_vgg11,
    }
    if arch not in builders:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = builders[arch](num_classes=num_classes)
    if hardware_aware:
        model = convert_to_hardware_layers(model, num_bits=num_bits, num_bits_grad=num_bits_grad)
    return model


def enable_quant_measure_calibration(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, QuantMeasure):
            module.train()


def disable_quant_measure_calibration(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, QuantMeasure):
            module.eval()
