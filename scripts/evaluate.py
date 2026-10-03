from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from dataset import create_dataloaders
from models import (create_resnet50,create_efficientnet_b0,)


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

VISUALIZATION_DIR = (RESULTS_DIR / "visualizations")

VISUALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Model configuration
MODEL_CONFIGS = {

    "resnet50": {
        "create":   create_resnet50,
        "checkpoint":   MODEL_DIR /"resnet50_stage2_best.pth",
        "results_dir":  RESULTS_DIR /"ResNet_csv",
        "report_name":  "resnet50_test_report.txt",
        "matrix_name":  "resnet50_confusion_matrix.csv",
        "figure_name":  "resnet50_confusion_matrix.png",
    },

    "efficientnet_b0": {
        "create":   create_efficientnet_b0,
        "checkpoint":   MODEL_DIR / "efficientnet_b0_stage2_best.pth",
        "results_dir":  RESULTS_DIR / "EfficientNet_csv",
        "report_name":  "efficientnet_b0_test_report.txt",
        "matrix_name":  "efficientnet_b0_confusion_matrix.csv",
        "figure_name":  "efficientnet_b0_confusion_matrix.png",
    },
}

# Evaluate one model
def evaluate_model(model_name,test_loader,class_names,):
    """
    Evaluate one final trained model.

    The best Stage 2 checkpoint is loaded and
    evaluated on the untouched test set.
    """
    config = MODEL_CONFIGS[model_name]

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} - TEST EVALUATION")
    print("=" * 60)

    # Load best Stage 2 checkpoint
    checkpoint = torch.load(
        config["checkpoint"],
        map_location=DEVICE,
        weights_only=False
    )
    model = config["create"](num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint: {config['checkpoint']}")
    print(f"Best epoch: {checkpoint.get('epoch', 'unknown')}")

    # Generate predictions
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = torch.argmax(outputs,dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Overall metrics
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    # Print metrics
    print("\nOverall Metrics")
    print("-" * 40)
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print( f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")
    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report")
    print("-" * 60)
    print(report)

    # Create results directory
    results_dir = config["results_dir"]
    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save classification report
    report_path = (results_dir / config["report_name"])

    with open(report_path,"w",encoding="utf-8") as file:

        file.write(f"{model_name} Test Evaluation\n")
        file.write("=" * 60 +"\n\n")
        file.write(f"Best checkpoint epoch: {checkpoint.get('epoch', 'unknown')}\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro Precision: {macro_precision:.4f}\n")
        file.write(f"Macro Recall: {macro_recall:.4f}\n")
        file.write(f"Macro F1:  {macro_f1:.4f}\n")
        file.write(f"Weighted F1: {weighted_f1:.4f}\n")
        file.write("\nClassification Report\n")
        file.write(
            "=" * 60 +
            "\n"
        )
        file.write(report)

    print(f"Saved report: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels,all_predictions)

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_path = (results_dir / config["matrix_name"])
    cm_df.to_csv(cm_path)
    print(f"Saved confusion matrix CSV: {cm_path}")

    # Confusion matrix visualization
    plt.figure(figsize=(9, 8))
    plt.imshow(cm,interpolation="nearest")
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(class_names))

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right"
    )
    plt.yticks(tick_marks,class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add values inside confusion matrix
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()
    figure_path = (
        VISUALIZATION_DIR /
        config["figure_name"]
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    print(f"Saved confusion matrix image: {figure_path}")

    # Return metrics for model comparison
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    }

# Comparative Visualization
def create_comparison_visual( comparison_df):
    """
    Create a grouped bar chart comparing
    ResNet50 and EfficientNet-B0.
    """
    metrics = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
    ]

    resnet_row = comparison_df[
        comparison_df["Model"] ==
        "resnet50"
    ].iloc[0]

    efficientnet_row = comparison_df[
        comparison_df["Model"] ==
        "efficientnet_b0"
    ].iloc[0]

    resnet_values = [
        resnet_row[metric]
        for metric in metrics
    ]

    efficientnet_values = [
        efficientnet_row[metric]
        for metric in metrics
    ]

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

    # Create chart
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

    plt.ylim(0,1.05)
    plt.legend()
    plt.grid(axis="y",alpha=0.3)

    # Add metric values
    for positions, values in [
        (resnet_positions,resnet_values),
        (efficientnet_positions,efficientnet_values),

    ]:
        for position, value in zip(positions,values):
            plt.text(
                position,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    comparison_figure = (VISUALIZATION_DIR / "model_comparison.png")
    plt.savefig(
        comparison_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    print(
        f"Saved comparative visualization: {comparison_figure}")

# Main
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Load final test dataset
    (
        _,
        _,
        test_loader,
        _,
        _,
        test_dataset,
    ) = create_dataloaders()

    class_names = test_dataset.classes

    print(f"Test images: {len(test_dataset)}")
    print(f"Classes: {class_names}")

    # Evaluate both final models
    models_to_evaluate = ["resnet50","efficientnet_b0",]
    results = []

    for model_name in models_to_evaluate:
        result = evaluate_model(
            model_name=model_name,
            test_loader=test_loader,
            class_names=class_names,
        )
        results.append(result)

    # Final Model Comparison
    comparison_df = pd.DataFrame(results)

    # Save comparison CSV
    comparison_path = (RESULTS_DIR / "model_comparison.csv")
    comparison_df.to_csv(comparison_path,index=False)
    print(f"\nSaved model comparison: {comparison_path}")

    # Display comparison
    print("\nFinal Model Comparison")
    print("=" * 80)
    print(
        comparison_df.to_string(
            index=False
        )
    )

    # Create comparison visualization
    create_comparison_visual(comparison_df)
    print(
        "\nAll model evaluations and "
        "comparison visualizations "
        "completed successfully."
    )