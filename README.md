# README

This repository contains the support code for the KAUST Integrated Photonics Lab publication in *Advanced Photonics*, "Integrated Quantum Dot Lasers for Parallelized Photonic Edge Computing."

## Overview

This repository contains a compact MNIST training pipeline built around:

- A `ResNet18` teacher model.
- A quantization-aware `MLP` student model.
- Student pruning and fine-tuning.
- Knowledge distillation from the teacher to the student.
- A separate noise-scanning script for input-noise robustness analysis.
- A separate notebook for backward-gradient quantization/noise approximation experiments.

The current main path trains the student with noisy inputs and explicit 5-bit quantization-aware training.

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Main Pipeline

Run the main training pipeline with:

```bash
python main.py [--argument_name argument_value]
```

`main.py` performs the following steps:

1. Trains a `ResNet18` teacher on MNIST.
2. Builds a student `MLP` and prepares it for quantization-aware training.
3. Trains the student on MNIST with Gaussian input noise added to the training images.
4. Evaluates the undistilled student.
5. Prunes the student.
6. Runs knowledge distillation from the teacher to the pruned student.
7. Evaluates the distilled student.
8. Converts the final student to a quantized model and saves the weights.

## Main Files

- `main.py`: Main training entry point.
- `args.py`: Command-line arguments for the main pipeline.
- `dataloader.py`: MNIST loaders with noisy-input training.
- `resnet.py`: Teacher network definition.
- `MLP.py`: Student network definition and QAT config.
- `train.py`: Teacher/student train and test helpers.
- `distill.py`: Knowledge distillation loop.
- `prune.py`: Student pruning and short retraining.

## Noise Robustness Scan

Run the separate noise robustness script with:

```bash
python noise_scan.py
```

This script:

1. Reuses the same teacher/student setup as the main path.
2. Trains or loads a student checkpoint.
3. Evaluates the student under multiple input-noise levels.
4. Saves checkpoints, CSV results, and a robustness plot under `Noise_analysis/`.

## Notebook Experiment

The notebook `backward-gradient-5bit-noise-simulation.ipynb` is a separate experiment for approximating noisy and quantized backward-gradient propagation.

## Command-Line Arguments

The main pipeline accepts these arguments:

- `--image_size`: Input image size. Choices: `7`, `14`, `28`.
- `--layer_num`: Number of student hidden layers. Choices: `1`, `2`, `3`, `4`.
- `--train_epoch`: Number of teacher/student training epochs before distillation.
- `--hidden_layer_size`: Width of the hidden layers in the student MLP.
- `--distill_epoch`: Number of distillation epochs.
- `--prune_amount`: Fraction of linear-layer weights removed during pruning.
- `--link_calibrated_noise`: Standard deviation of Gaussian noise added to training inputs.

## Example

Example main run:

```bash
python main.py --image_size 14 --layer_num 4 --train_epoch 20 --hidden_layer_size 14 --distill_epoch 5 --prune_amount 0.5 --link_calibrated_noise 0.1
```

Example noise scan:

```bash
python noise_scan.py
```

## License

This repository includes MIT-licensed components. See the bundled license files for details.
