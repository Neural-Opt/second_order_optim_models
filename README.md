# Second Order Optimization Models

![GitHub License](https://img.shields.io/github/license/Neural-Opt/second_order_optim_models)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

<div align="center">
   <img src="optim.gif" alt="Demo of the feature" width="500"/>
</div>
Second Order Optimization Models is a project designed to evaluate and experiment with advanced optimization algorithms, particularly those involving second-order methods. This repository includes implementations, benchmarks, and utilities to analyze performance and efficiency for various optimizers applied in machine learning contexts.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)

- [License](#license)

## Features

- **Second-order optimizers**: Implementations of advanced optimization algorithms that utilize second-order information for more efficient convergence.
- **Benchmarking Tools**: Built-in support for benchmarking various optimizers on different models.
- **Flexible Configuration**: Easily configurable via YAML files.
- **GPU Support**: Supports GPU-based memory management and optimization tracking.
- **Utilities for Post-Processing**: Includes tools for processing and visualizing results after training or benchmarking runs.

## Installation

### Prerequisites

- Python 3.8 or higher
- [PyTorch](https://pytorch.org/get-started/locally/) (ensure compatibility with your GPU if applicable)
- `pip` for package management

### Steps

Clone the repository and install dependencies:

```bash
git clone https://github.com/Neural-Opt/second_order_optim_models.git
cd second_order_optim_models
```



### Configuration (`config/config.yaml`)
```yaml
# Example configuration
runs:
  epochs: 30
  name: cifar10-cosine
  dir: "./runs"
  lr_schedule_per_epoch: true
dataset:
  name: cifar10-cosine
  path: "./data/cifar10-cosine"
kpi:
- name: tps
  unit: ms
  fqn: Time per step
  plot: box
- name: ttc
  unit: epochs
  fqn: Time till convergence
  plot: graph
- name: acc_train
  unit: "%"
  fqn: Training Accuracy
  plot: graph
- name: acc_test
  unit: "%"
  fqn: Test Accuracy
  plot: graph
- name: test_loss
  unit: ''
  fqn: Test Loss
  plot: graph
- name: train_loss
  unit: ''
  fqn: Train Loss
  plot: graph
- name: gpu_mem
  unit: MB
  fqn: GPU Memory usage
  plot: box
- name: bleu
  unit: ''
  fqn: BLEU Score
  plot: graph
lr_scheduler:
  type: CosineAnnealingLR
  params:
    T_max: 200
optim:
  Apollo:
    name: Apollo
    params:
      init_lr: None
      lr: 0.01
      beta: 0.9
      warmup: 500
      weight_decay_type: L2
      weight_decay: 0.00025
      rebound: constant
  ApolloW:
    name: Apollo
    params:
      init_lr: None
      lr: 0.01
      beta: 0.9
      warmup: 500
      weight_decay_type: decoupled
      weight_decay: 0.025
      rebound: constant
  AdaHessian:
    name: AdaHessian
    params:
      lr: 0.15
      beta1: 0.9
      beta2: 0.999
      weight_decay: 0.001
      warmup: 500
  Adam:
    name: Adam
    params:
      lr: 0.001
      beta1: 0.9
      beta2: 0.999
      eps: 1.0e-08
      weight_decay: 0.00025
  AdamJan:
    name: "AdamJan"
    params:
      lr: 0.0003
      beta1: 0.9
      beta2: 0.98
      eps: 0.00000001
      weight_decay: 0
      weight_decouple: True
  AdamW:
    name: Adam
    params:
      lr: 0.001
      beta1: 0.9
      beta2: 0.999
      eps: 1.0e-08
      weight_decay: 0.025
      weight_decouple: true
  AdaBelief:
    name: AdaBelief
    params:
      lr: 0.001
      beta1: 0.9
      beta2: 0.999
      eps: 1.0e-08
      weight_decay: 0.00025
      weight_decouple: true
  RMSprop:
    name: RMSprop
    params:
      lr: 0.001
      alpha: 0.99
      eps: 1.0e-08
      weight_decay: 0.00025
  SGD:
    name: SGD
    params:
      lr: 0.1
      momentum: 0.9
      weight_decay: 0.00025
```
### Usage
To run our model with the respective configuration just do
```bash
python3 train.py 
```
