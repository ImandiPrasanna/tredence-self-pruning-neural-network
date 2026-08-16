# Self-Pruning Neural Network for CIFAR-10

## Overview

This project implements a self-pruning neural network using learnable
gates and L1-style sparsity regularization.

The model learns both:
1. The network weights
2. A learnable gate associated with every weight

The gates determine the effective contribution of each connection.
During training, sparsity regularization encourages the gates toward zero,
allowing the network to suppress unnecessary connections.

## Objective

The objective is to investigate whether a neural network can learn to
prune its own connections during training while maintaining reasonable
classification accuracy.

The experiments were performed on the CIFAR-10 image classification
dataset.

## Methodology

Each custom linear layer uses a learnable gate for every weight.

The effective weight is calculated as:

Effective Weight = Weight × sigmoid(Gate Score)

The training objective is:

Total Loss = Classification Loss + λ × Sparsity Loss

where:

Sparsity Loss = Sum of all gate values

The coefficient λ controls the strength of the sparsity regularization.

A larger λ applies stronger pressure toward smaller gate values.

## Model Architecture

The network consists of three prunable fully connected layers:

- Input: 3072 features (3 × 32 × 32 CIFAR-10 image)
- Prunable Linear: 3072 → 256
- Prunable Linear: 256 → 128
- Prunable Linear: 128 → 10
- ReLU activation between hidden layers

Every weight in the three linear layers has an associated learnable gate.

## Dataset

CIFAR-10 was used for training and evaluation.

- Training images: 50,000
- Test images: 10,000
- Number of classes: 10
- Image size: 32 × 32 RGB

## Sparsity Definition

A gate is considered pruned when:

gate < 0.01

Sparsity is calculated as:

Sparsity (%) =
(Number of gates below 0.01 / Total number of gates) × 100

## Experimental Setup

All three experiments used:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Training epochs: 20
- Pruning threshold: 0.01

Three values of λ were evaluated:

- 0.00001
- 0.00005
- 0.00020

## Experimental Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|------------|------------------:|------------:|
| 0.00001    | 55.86             | 0.30        |
| 0.00005    | 55.12             | 5.85        |
| 0.00020    | 55.37             | 34.62       |

## Results Analysis

The experiments show that increasing the sparsity regularization
coefficient increases the sparsity of the network.

With λ = 0.00001, the model achieved 55.86% test accuracy with only
0.30% sparsity.

Increasing λ to 0.00005 increased sparsity to 5.85%, while test accuracy
was 55.12%.

With λ = 0.00020, the model achieved 34.62% sparsity while maintaining
55.37% test accuracy.

Therefore, stronger sparsity regularization can substantially reduce
the number of active connections while maintaining comparable
classification performance.

## Gate Distribution

The gate distribution of the high-λ model shows a strong concentration
of gate values near zero.

This indicates that the sparsity regularization successfully suppresses
many connections.

The pruning threshold used in the experiment is 0.01.

The high-λ model achieved:

- Test Accuracy: 55.37%
- Sparsity: 34.62%

## Project Structure

```text
tredence-self-pruning-neural-network/
│
├── train.py
├── README.md
├── requirements.txt
│
└── results/
    └── gate_distribution.png
