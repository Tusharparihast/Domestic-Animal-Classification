from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = Path("results")

RESNET_DIR = RESULTS_DIR / "ResNet_csv"
EFFICIENTNET_DIR = RESULTS_DIR / "EfficientNet_csv"

VISUALIZATION_DIR = (RESULTS_DIR / "visualizations")

VISUALIZATION_DIR.mkdir(parents=True,exist_ok=True)

# Helper function
def save_plot(filename):
    """
    Save the current matplotlib figure
    and close it.
    """
    output_path = (
        VISUALIZATION_DIR / filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

# Load training histories
resnet_stage1 = pd.read_csv(
    RESNET_DIR / "resnet50_stage1_history.csv"
)

resnet_stage2 = pd.read_csv(
    RESNET_DIR / "resnet50_stage2_history.csv"
)

efficientnet_stage1 = pd.read_csv(
    EFFICIENTNET_DIR / "efficientnet_b0_stage1_history.csv"
)

efficientnet_stage2 = pd.read_csv(
    EFFICIENTNET_DIR / "efficientnet_b0_stage2_history.csv"
)

# ResNet50 Stage 1 - Loss
plt.figure(figsize=(8, 5))

plt.plot(
    resnet_stage1["epoch"],
    resnet_stage1["train_loss"],
    label="Training Loss"
)

plt.plot(
    resnet_stage1["epoch"],
    resnet_stage1["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet50 Stage 1 Loss")
plt.legend()
plt.grid(alpha=0.3)

save_plot("resnet50_stage1_loss.png")

# ResNet50 Stage 1 - Accuracy
plt.figure(figsize=(8, 5))
plt.plot(
    resnet_stage1["epoch"],
    resnet_stage1["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    resnet_stage1["epoch"],
    resnet_stage1["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ResNet50 Stage 1 Accuracy")
plt.legend()
plt.grid(alpha=0.3)

save_plot("resnet50_stage1_accuracy.png")

# ResNet50 Stage 1 - Macro F1
plt.figure(figsize=(8, 5))

plt.plot(
    resnet_stage1["epoch"],
    resnet_stage1["val_macro_f1"],
    label="Validation Macro F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("ResNet50 Stage 1 Validation Macro F1")
plt.legend()
plt.grid(alpha=0.3)

save_plot("resnet50_stage1_macro_f1.png")

# ResNet50 Stage 2 - Loss
plt.figure(figsize=(8, 5))

plt.plot(
    resnet_stage2["epoch"],
    resnet_stage2["train_loss"],
    label="Training Loss"
)

plt.plot(
    resnet_stage2["epoch"],
    resnet_stage2["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet50 Stage 2 Loss")
plt.legend()
plt.grid(alpha=0.3)

save_plot("resnet50_stage2_loss.png")

# ResNet50 Stage 2 - Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    resnet_stage2["epoch"],
    resnet_stage2["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    resnet_stage2["epoch"],
    resnet_stage2["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ResNet50 Stage 2 Accuracy")
plt.legend()
plt.grid(alpha=0.3)

save_plot("resnet50_stage2_accuracy.png")

# ResNet50 Stage 2 - Macro F1
plt.figure(figsize=(8, 5))
plt.plot(
    resnet_stage2["epoch"],
    resnet_stage2["val_f1"],
    label="Validation Macro F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("ResNet50 Stage 2 Validation Macro F1")
plt.legend()
plt.grid(alpha=0.3)

save_plot("resnet50_stage2_macro_f1.png")

# EfficientNet-B0 Stage 1 - Loss
plt.figure(figsize=(8, 5))

plt.plot(
    efficientnet_stage1["epoch"],
    efficientnet_stage1["train_loss"],
    label="Training Loss"
)

plt.plot(
    efficientnet_stage1["epoch"],
    efficientnet_stage1["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNet-B0 Stage 1 Loss")
plt.legend()
plt.grid(alpha=0.3)

save_plot("efficientnet_b0_stage1_loss.png")

# EfficientNet-B0 Stage 1 - Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    efficientnet_stage1["epoch"],
    efficientnet_stage1["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    efficientnet_stage1["epoch"],
    efficientnet_stage1["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("EfficientNet-B0 Stage 1 Accuracy")
plt.legend()
plt.grid(alpha=0.3)
save_plot("efficientnet_b0_stage1_accuracy.png")

# EfficientNet-B0 Stage 1 - Macro F1
plt.figure(figsize=(8, 5))

plt.plot(
    efficientnet_stage1["epoch"],
    efficientnet_stage1["val_macro_f1"],
    label="Validation Macro F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("EfficientNet-B0 Stage 1 Validation Macro F1")
plt.legend()
plt.grid(alpha=0.3)
save_plot("efficientnet_b0_stage1_macro_f1.png")

# EfficientNet-B0 Stage 2 - Loss
plt.figure(figsize=(8, 5))

plt.plot(
    efficientnet_stage2["epoch"],
    efficientnet_stage2["train_loss"],
    label="Training Loss"
)

plt.plot(
    efficientnet_stage2["epoch"],
    efficientnet_stage2["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNet-B0 Stage 2 Loss")
plt.legend()
plt.grid(alpha=0.3)
save_plot("efficientnet_b0_stage2_loss.png")

# EfficientNet-B0 Stage 2 - Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    efficientnet_stage2["epoch"],
    efficientnet_stage2["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    efficientnet_stage2["epoch"],
    efficientnet_stage2["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("EfficientNet-B0 Stage 2 Accuracy")
plt.legend()
plt.grid(alpha=0.3)
save_plot("efficientnet_b0_stage2_accuracy.png")

# EfficientNet-B0 Stage 2 - Macro F1
plt.figure(figsize=(8, 5))

plt.plot(
    efficientnet_stage2["epoch"],
    efficientnet_stage2["val_macro_f1"],
    label="Validation Macro F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("EfficientNet-B0 Stage 2 Validation Macro F1")
plt.legend()
plt.grid(alpha=0.3)

save_plot("efficientnet_b0_stage2_macro_f1.png")

# Final model comparison
comparison_path = (RESULTS_DIR /"model_comparison.csv")

if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    metrics = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
    ]

    resnet = comparison[
        comparison["Model"] ==
        "resnet50"
    ].iloc[0]

    efficientnet = comparison[
        comparison["Model"] ==
        "efficientnet_b0"
    ].iloc[0]

    # Comparison chart
    x = range(len(metrics))
    width = 0.35
    resnet_positions = [
        value - width / 2
        for value in x
    ]

    efficientnet_positions = [
        value + width / 2
        for value in x
    ]

    resnet_values = [
        resnet[metric]
        for metric in metrics
    ]

    efficientnet_values = [
        efficientnet[metric]
        for metric in metrics
    ]

    plt.figure(figsize=(11, 6))
    plt.bar(
        resnet_positions,
        resnet_values,
        width=width,
        label="ResNet50"
    )

    plt.bar(
        efficientnet_positions,
        efficientnet_values,
        width=width,
        label="EfficientNet-B0"
    )

    plt.xticks(
        list(x),
        metrics,
        rotation=20
    )

    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")
    plt.title(
        "ResNet50 vs EfficientNet-B0 "
        "Test Performance"
    )

    plt.ylim(0,1.0)
    plt.legend()
    plt.grid(axis="y",alpha=0.3)

    # Add values
    for positions, values in [
        (
            resnet_positions,
            resnet_values
        ),

        (
            efficientnet_positions,
            efficientnet_values
        ),

    ]:
        for position, value in zip(
            positions,
            values
        ):

            plt.text(
                position,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    save_plot(
        "model_comparison.png"
    )

else:
    print(
        "\nModel comparison CSV not found."
    )

    print(
        "Run evaluate.py first to create "
        "the final model comparison."
    )

# Final message
print(
    "\nAll training and evaluation "
    "visualizations generated successfully."
)