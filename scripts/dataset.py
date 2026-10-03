from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# Configuration

DATASET_ROOT = Path("data")

TRAIN_DIR = DATASET_ROOT / "train"
VALIDATION_DIR = DATASET_ROOT / "validation"
TEST_DIR = DATASET_ROOT / "test"

Image_size = 224
Batch_size = 32
Num_workers = 0  # 0 separate processes will load and preprocess images in parallel while the model is training.

# Image Transformations
Imagenet_mean = [0.485, 0.456, 0.406]
Imagenet_std = [0.229, 0.224, 0.225]
# Training transformations include random augmentation.

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop( Image_size, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=Imagenet_mean,
        std=Imagenet_std
    )
])

# Validation and test data should not use random augmentation

eval_transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop(Image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=Imagenet_mean,
        std=Imagenet_std
    )
])

# Dataset Loading

def create_datasets():
    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform = train_transform
    )

    validation_dataset = datasets.ImageFolder(
        VALIDATION_DIR,
        transform = eval_transform
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform = eval_transform
    )

    return train_dataset, validation_dataset, test_dataset

''' 
Weighted Sampling:
so that smaller classes are sampled more frequently during training. 
'''

def create_wt_sampler(train_dataset):
    # Get the class labels for every training images
    targets = torch.tensor(train_dataset.targets)

    # Count the nimber of images in each class
    class_counts = torch.bincount(targets)

    # Give smaller classes larger weights
    class_weights = 1.0 / class_counts.float()

    #Assign a weight to each training image based on its class
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )

    return sampler

# DataLoaders

def create_dataloaders():
    train_dataset, validation_dataset, test_dataset = create_datasets()

    # Use wt. sampling only for the training dataset
    train_sampler = create_wt_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size = Batch_size,
        sampler = train_sampler,
        num_workers = Num_workers,
        pin_memory = torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size = Batch_size,
        shuffle = False,
        num_workers = Num_workers,
        pin_memory = torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = Batch_size,
        shuffle = False,
        num_workers = Num_workers,
        pin_memory = torch.cuda.is_available()
    )

    return (train_loader, validation_loader, test_loader, 
            train_dataset, validation_dataset, test_dataset 
    )

# Quick Test

if __name__ == "__main__":
    (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset,
        validation_dataset,
        test_dataset
    ) = create_dataloaders()

    print("Dataset loaded successfully.")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    print(f"Training Batches: {len(train_loader)}")
    print(f"Validation Batches: {len(validation_loader)}")
    print(f"Test Batches: {len(test_loader)}")