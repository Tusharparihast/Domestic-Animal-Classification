from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import create_dataloaders
from models import create_resnet50

# Configuration

num_classes = 9
learning_rate = 1e-4
num_epochs = 10
early_stopping = 3

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

stage1_checkpoint_path = model_dir / "resnet50_stage1_best.pth"
stage2_checkpoint_path = model_dir / "resnet50_stage2_best.pth"
history_path = results_dir / "resnet50_stage2_history.csv"

# Device configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data

(
    train_loader,
    validation_loader,
    test_loader,
    train_dataset,
    validation_dataset,
    test_dataset
) = create_dataloaders()

# Create Model
model = create_resnet50(num_classes)
# Load stage 1 checkpoint

checkpoint = torch.load(
    stage1_checkpoint_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint['model_state_dict'])

print(
    f"Loaded stage 1 checkppoint from: {stage1_checkpoint_path}"
)

# Freeze the pre-trained backbone
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze ResNet layer4 and classifier layer
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer

optimizer = optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad, model.parameters()
    ),
    lr = learning_rate
)

# Training state
best_val_f1 = 0.0
epoch_without_improvement = 0

training_history = []

# Training Loop
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1} / {num_epochs}]")

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
        train_loader, 
        desc="Training", 
        leave=False
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_loss = (running_loss / len(train_loader))

    train_accuracy = (100 * correct / total)

    # Validation

    model.eval()
    val_loss = 0.0

    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for images, labels in tqdm(
            validation_loader,
            desc="Validation",
            leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            val_predictions.extend(predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(validation_loader)

    # Validation metrics
    val_f1 = f1_score(
        val_labels,
        val_predictions,
        average="macro"
    )

    val_accuracy = (
        100 * sum(
            prediction == label
            for prediction, label in zip(val_predictions, val_labels)
        )
        / len(val_labels)
    )

    # Print results
    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.2f}% | "
        f"Validation Macro F1: {val_f1:.4f}"
    )

    # Save training history
    training_history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1
    })

    # Checkpointing

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1
            },
            stage2_checkpoint_path
        )

        print(
            f"Best checkpoint saved"
            f"(Macro F1: {val_f1:.4f})"
        )
    else:
        epochs_without_improvement += 1
        print(
            f"No improvement in Macro F1 for "
            f"{epochs_without_improvement} epoch(s)."
        )

    # Early stopping
    if epochs_without_improvement >= early_stopping:
        print(
            f"Early stopping triggered after "
            f"{early_stopping} epochs without improvement."
        )
        break

    # Training complete

    print("\nResNet50 Stage 2 Training Complete.")
    print(
        f"Best Validation Macro F1: {best_val_f1:.4f}"
    )
    print(
        f"Best Checkpoint:"
        f" {stage2_checkpoint_path}"
    )

    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(history_path, index=False)
    print(
        f"Training history saved to: {history_path}"
    )