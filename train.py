import os
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

FINAL_LAMBDA = 0.0001
PRUNING_THRESHOLD = 0.01

DATA_DIR = "./data"
RESULTS_DIR = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)
print("PyTorch version:", torch.__version__)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CIFAR-10 NORMALIZATION
# ============================================================

CIFAR_MEAN = (
    0.4914,
    0.4822,
    0.4465
)

CIFAR_STD = (
    0.2470,
    0.2435,
    0.2616
)


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])


# ============================================================
# LOAD CIFAR-10 DATASET
# ============================================================

print("\nDownloading/loading CIFAR-10 dataset...")

train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=test_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))


# ============================================================
# PRUNABLE LINEAR LAYER
# ============================================================

class PrunableLinear(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Learnable network weights
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Learnable gate scores
        self.gate_scores = nn.Parameter(
            torch.full(
                (out_features, in_features),
                3.0
            )
        )

        # Kaiming initialization
        nn.init.kaiming_uniform_(
            self.weight,
            a=np.sqrt(5)
        )

    def forward(self, x):

        # Convert gate scores into values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)

        # Effective weight
        effective_weight = self.weight * gates

        return F.linear(
            x,
            effective_weight
        )


# ============================================================
# SELF-PRUNING NETWORK
# ============================================================

class SelfPruningNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        # CIFAR-10:
        # 3 channels × 32 × 32 = 3072 input features

        self.fc1 = PrunableLinear(
            3 * 32 * 32,
            1024
        )

        self.fc2 = PrunableLinear(
            1024,
            512
        )

        self.fc3 = PrunableLinear(
            512,
            10
        )

    def forward(self, x):

        # Flatten image
        x = x.view(x.size(0), -1)

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        x = self.fc3(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = SelfPruningNetwork().to(device)

print("\nModel architecture:")
print(model)


# ============================================================
# SPARSITY LOSS
# ============================================================

def calculate_sparsity_loss(model):

    sparsity_loss = 0.0

    for module in model.modules():

        if isinstance(module, PrunableLinear):

            gates = torch.sigmoid(
                module.gate_scores
            )

            sparsity_loss = sparsity_loss + gates.sum()

    return sparsity_loss


# ============================================================
# GET ALL GATES
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
# CALCULATE SPARSITY
# ============================================================

def calculate_sparsity(
    model,
    threshold=PRUNING_THRESHOLD
):

    all_gates = get_all_gates(model)

    sparsity = (
        (all_gates < threshold)
        .float()
        .mean()
        .item()
        * 100
    )

    return sparsity, all_gates


# ============================================================
# LAMBDA SCHEDULE
# ============================================================

def get_lambda(epoch):

    # Epochs 1-10:
    # No sparsity pressure

    if epoch <= 10:
        return 0.0

    # Epochs 11-50:
    # Gradually increase lambda

    progress = (epoch - 10) / (EPOCHS - 10)

    return FINAL_LAMBDA * progress


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total_correct += (
                predicted
                .eq(labels)
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
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    lambda_
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

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

        # Total loss
        loss = (
            classification_loss
            + lambda_ * sparsity_loss
        )

        # Backpropagation
        loss.backward()

        # Update weights and gates
        optimizer.step()

        # Statistics
        total_loss += (
            loss.item()
            * images.size(0)
        )

        _, predicted = outputs.max(1)

        total_correct += (
            predicted
            .eq(labels)
            .sum()
            .item()
        )

        total_samples += labels.size(0)

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        100.0
        * total_correct
        / total_samples
    )

    return average_loss, accuracy


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_accuracy": [],
    "lambda": [],
    "sparsity": []
}


for epoch in range(
    1,
    EPOCHS + 1
):

    lambda_ = get_lambda(epoch)

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        lambda_
    )

    test_accuracy = evaluate_model(
        model,
        test_loader
    )

    sparsity, _ = calculate_sparsity(
        model
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["test_accuracy"].append(
        test_accuracy
    )

    history["lambda"].append(
        lambda_
    )

    history["sparsity"].append(
        sparsity
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"λ: {lambda_:.8f} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Test Accuracy: {test_accuracy:.2f}% | "
        f"Sparsity: {sparsity:.2f}%"
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

final_accuracy = evaluate_model(
    model,
    test_loader
)

final_sparsity, all_gates = calculate_sparsity(
    model,
    PRUNING_THRESHOLD
)

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Test Accuracy : {final_accuracy:.2f}%"
)

print(
    f"Sparsity      : {final_sparsity:.2f}%"
)

print(
    f"Total Gates   : {all_gates.numel():,}"
)

print(
    f"Minimum Gate  : {all_gates.min().item():.6f}"
)

print(
    f"Maximum Gate  : {all_gates.max().item():.6f}"
)

print(
    f"Mean Gate     : {all_gates.mean().item():.6f}"
)

print("=" * 60)


# ============================================================
# GATE STATISTICS
# ============================================================

below_01 = (
    (all_gates < 0.1)
    .float()
    .mean()
    .item()
    * 100
)

below_005 = (
    (all_gates < 0.05)
    .float()
    .mean()
    .item()
    * 100
)

below_001 = (
    (all_gates < 0.01)
    .float()
    .mean()
    .item()
    * 100
)

print("\nGate distribution statistics:")

print(
    f"Below 0.10 : {below_01:.2f}%"
)

print(
    f"Below 0.05 : {below_005:.2f}%"
)

print(
    f"Below 0.01 : {below_001:.2f}%"
)


# ============================================================
# SAVE GATE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    all_gates.numpy(),
    bins=50,
    edgecolor="black"
)

plt.axvline(
    PRUNING_THRESHOLD,
    linestyle="--",
    label="Pruning threshold = 0.01"
)

plt.xlabel("Gate Value")
plt.ylabel("Number of Gates")

plt.title(
    "Distribution of Gate Values - Final Model"
)

plt.legend()

plt.tight_layout()

plot_path = os.path.join(
    RESULTS_DIR,
    "gate_distribution.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nGate distribution saved to: {plot_path}"
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "final_results.txt"
)

with open(
    results_path,
    "w"
) as f:

    f.write(
        "Self-Pruning Neural Network - Final Results\n"
    )

    f.write(
        "=" * 50 + "\n"
    )

    f.write(
        f"Test Accuracy: {final_accuracy:.2f}%\n"
    )

    f.write(
        f"Sparsity: {final_sparsity:.2f}%\n"
    )

    f.write(
        f"Total Gates: {all_gates.numel():,}\n"
    )

    f.write(
        f"Minimum Gate: {all_gates.min().item():.6f}\n"
    )

    f.write(
        f"Maximum Gate: {all_gates.max().item():.6f}\n"
    )

    f.write(
        f"Mean Gate: {all_gates.mean().item():.6f}\n"
    )

    f.write(
        f"Below 0.10: {below_01:.2f}%\n"
    )

    f.write(
        f"Below 0.05: {below_005:.2f}%\n"
    )

    f.write(
        f"Below 0.01: {below_001:.2f}%\n"
    )

print(
    f"Final results saved to: {results_path}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    RESULTS_DIR,
    "self_pruning_cifar10.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print(
    f"Model saved to: {model_path}"
)

print("\nTraining and evaluation completed successfully!")
