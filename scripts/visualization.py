from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = Path("results")

# ResNet50 CSV result files
RESNET_RESULTS_DIR = (
    RESULTS_DIR / "ResNet_csv"
)

# Folder for generated visualization images
VISUALIZATION_DIR = (RESULTS_DIR / "visualizations")

VISUALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Input files

STAGE1_HISTORY_PATH = (
    RESNET_RESULTS_DIR
    / "resnet50_stage1_history.csv"
)

STAGE2_HISTORY_PATH = (
    RESNET_RESULTS_DIR
    / "resnet50_stage2_history.csv"
)

CONFUSION_MATRIX_PATH = (
    RESNET_RESULTS_DIR
    / "resnet50_confusion_matrix.csv"
)

# Load training histories
stage1 = pd.read_csv(
    STAGE1_HISTORY_PATH
)

stage2 = pd.read_csv(
    STAGE2_HISTORY_PATH
)

# Stage 1 — Loss
plt.figure(figsize=(10, 6))

plt.plot(
    stage1["epoch"],
    stage1["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    stage1["epoch"],
    stage1["val_loss"],
    marker="o",
    label="Validation Loss"
)
plt.title("ResNet50 Stage 1 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    VISUALIZATION_DIR
    / "resnet50_stage1_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Stage 1 — Accuracy
plt.figure(figsize=(10, 6))
plt.plot(
    stage1["epoch"],
    stage1["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    stage1["epoch"],
    stage1["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.title("ResNet50 Stage 1 Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    VISUALIZATION_DIR
    / "resnet50_stage1_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Stage 1 — Validation Macro F1
plt.figure(figsize=(10, 6))
plt.plot(
    stage1["epoch"],
    stage1["val_macro_f1"],
    marker="o",
    label="Validation Macro F1"
)

plt.title("ResNet50 Stage 1 Validation Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    VISUALIZATION_DIR
    / "resnet50_stage1_macro_f1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Stage 2 — Loss
plt.figure(figsize=(10, 6))

plt.plot(
    stage2["epoch"],
    stage2["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    stage2["epoch"],
    stage2["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("ResNet50 Stage 2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR
    / "resnet50_stage2_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Stage 2 — Accuracy
plt.figure(figsize=(10, 6))

plt.plot(
    stage2["epoch"],
    stage2["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    stage2["epoch"],
    stage2["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.title("ResNet50 Stage 2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    VISUALIZATION_DIR
    / "resnet50_stage2_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Stage 2 — Validation Macro F1
plt.figure(figsize=(10, 6))

plt.plot(
    stage2["epoch"],
    stage2["val_f1"],
    marker="o",
    label="Validation  F1"
)

plt.title(
    "ResNet50 Stage 2 Validation F1"
)

plt.xlabel("Epoch")
plt.ylabel("F1")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR
    / "resnet50_stage2_f1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Load confusion matrix

confusion_matrix = pd.read_csv(
    CONFUSION_MATRIX_PATH,
    index_col=0
)

# Confusion Matrix Visualization

plt.figure(figsize=(10, 8))

plt.imshow(
    confusion_matrix,
    interpolation="nearest"
)

plt.title("ResNet50 Test Set Confusion Matrix")
plt.colorbar()

classes = confusion_matrix.columns

plt.xticks(
    range(len(classes)),
    classes,
    rotation=45,
    ha="right"
)

plt.yticks(range(len(classes)), classes)

# Add numerical values inside the matrix
for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix.columns)):

        plt.text(
            j,
            i,
            confusion_matrix.iloc[i, j],
            ha="center",
            va="center"
        )

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig(
    VISUALIZATION_DIR
    / "resnet50_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Completion message

print("\nResNet50 visualizations created successfully.")
print(f"Visualization directory: {VISUALIZATION_DIR}")
print("\nGenerated files:")
print("- resnet50_stage1_loss.png")
print("- resnet50_stage1_accuracy.png")
print("- resnet50_stage1_f1.png")
print("- resnet50_stage2_loss.png")
print("- resnet50_stage2_accuracy.png")
print("- resnet50_stage2_f1.png")
print("- resnet50_confusion_matrix.png")