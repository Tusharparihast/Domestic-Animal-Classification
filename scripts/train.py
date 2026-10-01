import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from models import create_resnet50

# Configuration
num_classes = 9
Learning_rate = 0.001
num_epochs = 10

model_path = "models/resnet50_best.pth"

# Use the GPU if CUDA is available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data

create_dataloaders() = (
    train_loader,
    validation_loader,
    test_loader,
    train_dataset,
    validation_dataset,
    test_dataset
)

# Create Model
model = create_resnet50(num_classes)

# Move the model to the selected device.
model = model.to(device)

# Freeze the pre-trained backbone
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze only the new classifier layer
for parameter in model.fc.parameters():
    parameter.requires_grad = True

# Loss Function

criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    filter(lambda parameter: parameter.requires_grad, model.parameters()),
    lr = Learning_rate
)

# Training

best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Move images and labels to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()    # Clear previous gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()         # Update weights

        # Track training statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # Validation Phase
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation for validation
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(validation_loader)
    val_accuracy = 100 * correct / total

    # Print results

    print(
        f"Epoch [{epoch + 1} / {num_epochs}]"
        f"Train Loss: {train_loss:.4f}"
        f"Train Accuracy: {train_accuracy:.2f}%"
        f"Validation Loss: {val_loss:.4f}"
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Save the best model

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), model_path)

        print(
            f"Best model saved with "
            f" Validation accuracy: {val_accuracy:.2f}%"
        )

print("\nTraining Complete!")
print(
    f"Best Validation Accuracy:"
    f"{best_val_accuracy:.2f}%"
)