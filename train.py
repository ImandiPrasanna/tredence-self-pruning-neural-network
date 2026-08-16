import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 20

# Lambda values used for the experiments
LAMBDA_VALUES = [0.00001, 0.00005, 0.0002]

# Gates below this value are considered pruned
PRUNING_THRESHOLD = 1e-2

DATA_DIR = "./data"
RESULTS_DIR = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Prunable Linear Layer
# ============================================================

class PrunableLinear(nn.Module):
    """
    Linear layer with a learnable sigmoid gate for every weight.

    Effective weight:
        W_eff = W * sigmoid(gate_score)
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Standard linear-layer weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )

        # Learnable gate score for every weight
        self.gate_scores = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

        # Bias
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

    def forward(self, x):

        # Convert gate scores into values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)

        # Apply gates to the weights
        effective_weight = self.weight * gates

        # Standard linear transformation
        return F.linear(
            x,
            effective_weight,
            self.bias
        )


# ============================================================
# Self-Pruning Neural Network
# ============================================================

class SelfPruningNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # CIFAR-10 image:
        # 32 x 32 x 3 = 3072 input features
        self.fc1 = PrunableLinear(32 * 32 * 3, 256)

        self.fc2 = PrunableLinear(256, 128)

        # 10 output classes for CIFAR-10
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x):

        # Flatten image
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x


# ============================================================
# Sparsity Loss
# ============================================================

def calculate_sparsity_loss(model):
    """
    Calculate the sparsity penalty as the sum of all
    sigmoid gate values across PrunableLinear layers.
    """

    sparsity_loss = 0.0

    for module in model.modules():

        if isinstance(module, PrunableLinear):

            gates = torch.sigmoid(
                module.gate_scores
            )

            sparsity_loss += gates.sum()

    return sparsity_loss


# ============================================================
# Calculate Model Sparsity
# ============================================================

def calculate_sparsity(
    model,
    threshold=PRUNING_THRESHOLD
):
    """
    Calculate percentage of gates below the pruning threshold.
    """

    all_gates = []

    for module in model.modules():

        if isinstance(module, PrunableLinear):

            gates = torch.sigmoid(
                module.gate_scores
            )

            all_gates.append(
                gates.detach().cpu().flatten()
            )

    all_gates = torch.cat(all_gates)

    sparsity = (
        (all_gates < threshold)
        .float()
        .mean()
        .item()
        * 100
    )

    return sparsity


# ============================================================
# Get All Gate Values
# ============================================================

def get_all_gates(model):

    all_gates = []

    for module in model.modules():

        if isinstance(module, PrunableLinear):

            gates = torch.sigmoid(
                module.gate_scores
            )

            all_gates.append(
                gates.detach()
                .cpu()
                .flatten()
            )

    return torch.cat(all_gates)


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    lambda_
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Classification loss
        classification_loss = criterion(
            outputs,
            labels
        )

        # Sparsity loss
        sparsity_loss = calculate_sparsity_loss(
            model
        )

        # Combined objective
        loss = (
            classification_loss
            + lambda_ * sparsity_loss
        )

        # Backpropagation
        loss.backward()

        # Update weights and gate scores
        optimizer.step()

        # Statistics
        total_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = outputs.max(1)

        total_correct += (
            predicted.eq(labels)
            .sum()
            .item()
        )

        total_samples += labels.size(0)

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        100.0
        * total_correct
        / total_samples
    )

    return average_loss, accuracy


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    test_loader
):

    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total_correct += (
                predicted.eq(labels)
                .sum()
                .item()
            )

            total_samples += labels.size(0)

    accuracy = (
        100.0
        * total_correct
        / total_samples
    )

    return accuracy


# ============================================================
# Run One Lambda Experiment
# ============================================================

def run_experiment(
    lambda_,
    train_loader,
    test_loader,
    epochs=EPOCHS
):

    print("\n" + "=" * 60)
    print(
        f"Starting experiment with lambda = {lambda_}"
    )
    print("=" * 60)

    # Fresh model for every lambda
    model = SelfPruningNetwork().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(epochs):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                lambda_
            )
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

    test_accuracy = evaluate_model(
        model,
        test_loader
    )

    sparsity = calculate_sparsity(
        model
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Sparsity: "
        f"{sparsity:.2f}%"
    )

    return (
        model,
        test_accuracy,
        sparsity
    )


# ============================================================
# Plot Gate Distribution
# ============================================================

def plot_gate_distribution(
    model,
    lambda_
):

    gates = get_all_gates(model).numpy()

    plt.figure(figsize=(10, 6))

    plt.hist(
        gates,
        bins=50,
        edgecolor="black"
    )

    plt.axvline(
        PRUNING_THRESHOLD,
        linestyle="--",
        label=(
            f"Pruning threshold = "
            f"{PRUNING_THRESHOLD}"
        )
    )

    plt.xlabel("Gate Value")
    plt.ylabel("Number of Gates")

    plt.title(
        "Distribution of Gate Values "
        f"- High Lambda Model"
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "gate_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Gate distribution saved to: "
        f"{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # CIFAR-10 preprocessing
    # --------------------------------------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # --------------------------------------------------------
    # Load CIFAR-10
    # --------------------------------------------------------

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"\nTraining images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    trained_models = {}

    for lambda_ in LAMBDA_VALUES:

        (
            model,
            test_accuracy,
            sparsity
        ) = run_experiment(
            lambda_,
            train_loader,
            test_loader,
            EPOCHS
        )

        results.append({
            "lambda": lambda_,
            "test_accuracy": test_accuracy,
            "sparsity": sparsity
        })

        trained_models[lambda_] = model

    # --------------------------------------------------------
    # Print final results table
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL EXPERIMENTAL RESULTS")
    print("=" * 60)

    print(
        f"{'Lambda':<15}"
        f"{'Test Accuracy':<20}"
        f"{'Sparsity':<15}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{result['lambda']:<15.5f}"
            f"{result['test_accuracy']:<20.2f}"
            f"{result['sparsity']:<15.2f}"
        )

    # --------------------------------------------------------
    # Select highest-lambda model for visualization
    # --------------------------------------------------------

    high_lambda = max(
        LAMBDA_VALUES
    )

    high_model = trained_models[
        high_lambda
    ]

    plot_gate_distribution(
        high_model,
        high_lambda
    )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()
