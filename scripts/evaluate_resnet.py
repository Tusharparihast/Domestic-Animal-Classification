from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import create_dataloaders
from models import create_resnet50


# --------------------------------------------------
# Configuration
# --------------------------------------------------

num_classes = 9

checkpoint_path = Path(
    "models/resnet50_stage2_best.pth"
)

results_dir = Path("results")
results_dir.mkdir(
    parents=True,
    exist_ok=True
)

report_path = (
    results_dir / "resnet50_test_report.txt"
)

confusion_matrix_path = (
    results_dir / "resnet50_confusion_matrix.csv"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if device.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

(
    train_loader,
    validation_loader,
    test_loader,
    train_dataset,
    validation_dataset,
    test_dataset
) = create_dataloaders()


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = create_resnet50(num_classes)


# --------------------------------------------------
# Load best Stage 2 checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print(
    f"Loaded checkpoint from: "
    f"{checkpoint_path}"
)

print(
    f"Checkpoint epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation Macro F1: "
    f"{checkpoint['val_f1']:.4f}"
)


# --------------------------------------------------
# Test evaluation
# --------------------------------------------------

test_predictions = []
test_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_labels.extend(
            labels.numpy()
        )


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision = precision_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_macro_f1 = f1_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_weighted_f1 = f1_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

class_names = test_dataset.classes

report = classification_report(
    test_labels,
    test_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    test_labels,
    test_predictions
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nResNet50 Test Results")

print(
    f"Accuracy:       {test_accuracy:.4f}"
)

print(
    f"Macro Precision: {test_precision:.4f}"
)

print(
    f"Macro Recall:    {test_recall:.4f}"
)

print(
    f"Macro F1:        {test_macro_f1:.4f}"
)

print(
    f"Weighted F1:     {test_weighted_f1:.4f}"
)

print("\nClassification Report:")
print(report)


# --------------------------------------------------
# Save report
# --------------------------------------------------

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ResNet50 Test Evaluation\n"
    )

    file.write(
        "=========================\n\n"
    )

    file.write(
        f"Accuracy: {test_accuracy:.4f}\n"
    )

    file.write(
        f"Macro Precision: "
        f"{test_precision:.4f}\n"
    )

    file.write(
        f"Macro Recall: "
        f"{test_recall:.4f}\n"
    )

    file.write(
        f"Macro F1: "
        f"{test_macro_f1:.4f}\n"
    )

    file.write(
        f"Weighted F1: "
        f"{test_weighted_f1:.4f}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "---------------------\n"
    )

    file.write(report)


# --------------------------------------------------
# Save confusion matrix
# --------------------------------------------------

import pandas as pd

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    confusion_matrix_path
)


# --------------------------------------------------
# Complete
# --------------------------------------------------

print(
    f"\nTest report saved to: "
    f"{report_path}"
)

print(
    f"Confusion matrix saved to: "
    f"{confusion_matrix_path}"
)