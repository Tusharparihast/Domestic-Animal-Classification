from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from tqdm import tqdm

from dataset import create_dataloaders
from models import (
    create_resnet50,
    create_efficientnet_b0,
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

STAGE1_LEARNING_RATE = 1e-3
STAGE2_LEARNING_RATE = 1e-4

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Model configuration
MODEL_CONFIGS = {
    "resnet50": {
        "create": create_resnet50,

        "stage1_checkpoint":
            MODEL_DIR / "resnet50_stage1_best.pth",

        "stage2_checkpoint":
            MODEL_DIR / "resnet50_stage2_best.pth",

        "stage1_history":
            RESULTS_DIR /
            "ResNet_csv" /
            "resnet50_stage1_history.csv",

        "stage2_history":
            RESULTS_DIR /
            "ResNet_csv" /
            "resnet50_stage2_history.csv",
    },

    "efficientnet_b0": {
        "create": create_efficientnet_b0,

        "stage1_checkpoint":
            MODEL_DIR / "efficientnet_b0_stage1_best.pth",

        "stage2_checkpoint":
            MODEL_DIR / "efficientnet_b0_stage2_best.pth",

        "stage1_history":
            RESULTS_DIR /
            "EfficientNet_csv" /
            "efficientnet_b0_stage1_history.csv",

        "stage2_history":
            RESULTS_DIR /
            "EfficientNet_csv" /
            "efficientnet_b0_stage2_history.csv",
    },
}

# Model creation
def create_model(model_name, num_classes):
    """Create the selected ImageNet-pretrained model."""

    return MODEL_CONFIGS[model_name]["create"](
        num_classes=num_classes
    )

# Stage 1 configuration
def configure_stage1(model, model_name):
    """
    Freeze the pretrained backbone and train only
    the newly created classification head.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_name == "resnet50":

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif model_name == "efficientnet_b0":

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    return model

# Stage 2 configuration
def configure_stage2(model, model_name):
    """
    Fine-tune the later layers of the pretrained model
    together with the classification head.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_name == "resnet50":

        # Fine-tune ResNet50 layer4 and classifier.
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif model_name == "efficientnet_b0":

        # Fine-tune the later EfficientNet feature blocks and classifier.
        for parameter in model.features[6:].parameters():
            parameter.requires_grad = True

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    return model

# Trainable parameters
def get_trainable_parameters(model):
    """Return only parameters selected for training."""

    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


# Validation
def validate(model,validation_loader,criterion):
    # Evaluate the model on the validation set
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in validation_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs,labels)
            predictions = torch.argmax(outputs,dim=1)

            total_loss += (
                loss.item() *
                images.size(0)
            )

            all_labels.extend(labels.cpu().numpy())

            all_predictions.extend(predictions.cpu().numpy())

    validation_loss = (
        total_loss /
        len(validation_loader.dataset)
    )

    validation_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    validation_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        validation_loss,
        validation_accuracy,
        validation_macro_f1,
    )

# Train one stage
def train_stage(model_name,stage,model,train_loader,validation_loader,num_classes):
    """
    Train one stage of one model.

    Stage 1:
        Train only the classifier.

    Stage 2:
        Fine-tune the later backbone layers
        and classifier.
    """

    config = MODEL_CONFIGS[model_name]

    # Stage configuration
    if stage == 1:
        learning_rate = STAGE1_LEARNING_RATE
        checkpoint_path = (config["stage1_checkpoint"])
        history_path = (config["stage1_history"])
        model = configure_stage1(
            model,
            model_name
        )

    else:

        learning_rate = STAGE2_LEARNING_RATE
        checkpoint_path = (config["stage2_checkpoint"])
        history_path = (config["stage2_history"])
        model = configure_stage2(
            model,
            model_name
        )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        get_trainable_parameters(model),
        lr=learning_rate
    )

    # Training state
    history = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    # Display configuration
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} - STAGE {stage}")

    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Learning rate: {learning_rate}")

    # Epoch loop
    for epoch in range(1,NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        all_labels = []
        all_predictions = []

        progress = tqdm(
            train_loader,
            desc=(
                f"{model_name} "
                f"Stage {stage} "
                f"Epoch {epoch}/{NUM_EPOCHS}"
            )
        )

        for images, labels in progress:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(
                outputs,
                dim=1
            )
            running_loss += (loss.item() *images.size(0))

            all_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            progress.set_postfix(loss=f"{loss.item():.4f}")


        # Training metrics
        train_loss = (running_loss / len(train_loader.dataset))

        train_accuracy = accuracy_score(all_labels,all_predictions)

        # Validation metrics
        (
            validation_loss,
            validation_accuracy,
            validation_macro_f1,
        ) = validate(
            model,
            validation_loader,
            criterion
        )

        history_row = {
            "epoch": epoch,
            "train_loss":train_loss,
            "train_accuracy":train_accuracy * 100,
            "val_loss":validation_loss,
            "val_accuracy":validation_accuracy * 100,
        }

        if (
            stage == 2
            and model_name == "resnet50"
        ):

            history_row["val_f1"] = (
                validation_macro_f1
            )

        else:
            history_row["val_macro_f1"] = (
                validation_macro_f1
            )

        history.append(
            history_row
        )

        # Print epoch results
        print(
            f"Epoch {epoch}:Train Loss={train_loss:.4f}, "
            f"Train Acc="
            f"{train_accuracy * 100:.2f}%, "
            f"Val Loss="
            f"{validation_loss:.4f}, "
            f"Val Acc="
            f"{validation_accuracy * 100:.2f}%, "
            f"Val Macro F1="
            f"{validation_macro_f1:.4f}"
        )

        # Save best checkpoint
        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = (validation_macro_f1)
            epochs_without_improvement = 0

            checkpoint_path.parent.mkdir(parents=True,exist_ok=True)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_f1":
                        validation_macro_f1,
                    "classes":
                        num_classes,
                },
                checkpoint_path
            )

            print(f"Saved best checkpoint: {checkpoint_path}")

        else:
            epochs_without_improvement += 1

        # Early stopping
        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(f"Early stopping after epoch {epoch}.")
            break

    # Save history
    history_df = pd.DataFrame(history)

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print(f"Saved training history: {history_path}")
    return model


# Train one complete model
def train_model(model_name):
    """
    Train one model through Stage 1 and Stage 2.
    """
    print("\n" + "#" * 60)

    print(f"STARTING {model_name.upper()}")
    print("#" * 60)

    # Load dataset
    (
        train_loader,
        validation_loader,
        _,
        train_dataset,
        _,
        _,
    ) = create_dataloaders()

    num_classes = len(train_dataset.classes)

    # Stage 1
    model = create_model(model_name,num_classes)
    train_stage(
        model_name=model_name,
        stage=1,
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_classes=num_classes,
    )

    # Load the best Stage 1 checkpoint
    stage1_checkpoint = (
        MODEL_CONFIGS[model_name]
        ["stage1_checkpoint"]
    )

    checkpoint = torch.load(
        stage1_checkpoint,
        map_location=DEVICE,
        weights_only=False
    )

    model = create_model(model_name,num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nLoaded best Stage 1 checkpoint: {stage1_checkpoint}")

    # Stage 2
    train_stage(
        model_name=model_name,
        stage=2,
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_classes=num_classes,
    )

    print("\n" + "#" * 60)
    print(f"COMPLETED {model_name.upper()}")
    print("#" * 60)

# Main
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("\nTraining both models sequentially:")
    print("1. ResNet50 Stage 1")
    print("2. ResNet50 Stage 2")
    print("3. EfficientNet-B0 Stage 1")
    print("4. EfficientNet-B0 Stage 2")

    # Train ResNet50
    train_model("resnet50")

    # Train EfficientNet-B0
    train_model("efficientnet_b0")
    print("\nAll four training stages completed.")
