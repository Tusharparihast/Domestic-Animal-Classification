# Domestic Animal Classification

A deep learning project for classifying domestic animals into 9 categories using transfer learning and pretrained CNN architectures.

The project focuses not only on model training, but also on **dataset quality, duplicate detection, data leakage prevention, class imbalance handling, and reliable evaluation** before and during model development.

---

## Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Project Structure](#project-structure)
- [Features](#features)
- [Dataset](#dataset)
- [Dataset Preparation](#dataset-preparation)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Model Development](#model-development)
- [Training Strategy](#training-strategy)
- [Current Results](#current-results)
- [Results and Outputs](#results-and-outputs)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Modules Overview](#modules-overview)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)

---

## Overview

This project develops an image classification system capable of identifying domestic animals from images.

The dataset contains nine animal classes:

- Buffalo
- Camel
- Cat
- Chicken
- Donkey
- Goat
- Horse
- Pig
- Sheep

Two pretrained convolutional neural network architectures are being investigated:

- **ResNet50**
- **EfficientNet-B0**

The models use ImageNet pretrained weights and are adapted for the nine-class classification problem through transfer learning and fine-tuning.

A major focus of the project is ensuring that the dataset is reliable before model training. This includes checking for corrupted images, duplicate images, near-duplicate images, and possible data leakage between training, validation, and test sets.

---

## Project Status

The project is currently in the **model development and training stage**.

### Completed

- Dataset organization and split verification
- Class distribution analysis
- Image quality inspection
- Corrupted image detection
- Image format analysis
- Image dimension and color-mode analysis
- Visual inspection
- Exact duplicate detection
- Near-duplicate detection using pHash
- Cross-split near-duplicate investigation
- ORB-based verification of strong duplicate candidates
- Empirical ORB threshold selection
- Duplicate relationship grouping
- Dataset cleanup
- Dataset redistribution
- GPU-enabled PyTorch environment
- Data loading pipeline
- Image augmentation pipeline
- Weighted sampling for class imbalance
- ResNet50 model implementation
- EfficientNet-B0 model implementation
- ResNet50 Stage 1 transfer learning
- Best-model checkpointing
- Validation Macro F1 monitoring
- Early stopping logic

### In Progress

- Improved training progress visualization
- ResNet50 fine-tuning
- EfficientNet-B0 training
- Final model evaluation
- Model comparison

### Planned

- Confusion matrix
- Per-class evaluation
- Final test-set comparison
- Error analysis
- Explainability analysis such as Grad-CAM

---

# Project Structure

```text
Domestic-Animal-Classification/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── pre-processing/
│
├── results/
│
├── scripts/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
│
├── models/
│
├── .gitignore
├── requirements.txt
└── README.md
```

The project structure will be expanded as additional training and evaluation components are implemented.

---

# Features

The project currently includes or is designed to include:

- 9-class domestic animal classification
- Dataset quality assurance
- Exact duplicate detection
- Perceptual near-duplicate detection
- Cross-split leakage investigation
- ORB feature-based duplicate verification
- Dataset cleanup and redistribution
- Class imbalance handling
- Image augmentation
- ImageNet normalization
- Transfer learning
- CNN fine-tuning
- GPU acceleration using CUDA
- Validation-based model checkpointing
- Macro F1-based model selection
- Early stopping
- Final test-set evaluation
- Confusion matrix analysis
- Per-class performance analysis
- Model comparison

---

# Dataset

The original dataset contained **10,778 images** distributed across nine animal classes.

### Original Split

| Split | Images |
|---|---:|
| Train | 5,597 |
| Validation | 2,317 |
| Test | 2,864 |
| **Total** | **10,778** |

The original dataset was not perfectly balanced across classes.

### Original Class Distribution

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Buffalo | 371 | 371 | 371 |
| Camel | 401 | 401 | 401 |
| Horse | 975 | 140 | 278 |
| Pig | 644 | 91 | 183 |
| Cat | 611 | 611 | 613 |
| Donkey | 773 | 110 | 221 |
| Sheep | 692 | 99 | 196 |
| Chicken | 387 | 387 | 389 |
| Goat | 743 | 107 | 212 |

---

## Classes

The dataset contains:

| Class |
|---|
| Buffalo |
| Camel |
| Cat |
| Chicken |
| Donkey |
| Goat |
| Horse |
| Pig |
| Sheep |

---

# Dataset Preparation

Dataset preparation was treated as an important stage before model training.

The dataset was checked for:

- Folder organization
- Train/validation/test split structure
- Class counts
- Corrupted images
- File formats
- Image dimensions
- Color modes
- Visual quality
- Exact duplicates
- Near duplicates
- Cross-split leakage

The detailed dataset investigation and preprocessing documentation is maintained separately in:

**[`notebooks/README.md`](notebooks/README.md)**

The preprocessing documentation contains the detailed methodology, observations, dataset statistics, duplicate-detection process, leakage investigation, and dataset-cleaning decisions.

---

# Dataset Quality and Leakage Prevention

## Exact Duplicate Detection

Exact duplicate detection was performed before model training.

The final exact-duplicate analysis produced:

| Metric | Result |
|---|---:|
| Total images analyzed | 10,778 |
| Unique images | 10,723 |
| Duplicate groups | 37 |
| Cross-split exact duplicate groups | 0 |

After cleanup, no exact duplicate groups remained across the train, validation, and test sets.

---

## Near-Duplicate Detection

Exact duplicate detection alone is not sufficient because the same underlying image can appear with:

- Different resolutions
- Different compression
- Different file formats
- Different filenames
- Cropping
- Minor transformations

Therefore, perceptual hashing was used to identify visually similar images.

The project used **pHash** with a Hamming-distance threshold of **5**.

The initial near-duplicate analysis identified:

- 766 near-duplicate pairs
- 382 cross-split near-duplicate pairs
- 704 unique images involved in cross-split relationships

The detailed near-duplicate analysis is documented in:

**[`notebooks/README.md`](notebooks/README.md)**

---

## ORB Verification

The strongest pHash candidates were further investigated using ORB feature matching.

The verification process used:

- Up to 1000 ORB features
- Hamming-distance descriptor matching
- Lowe ratio test of 0.75
- RANSAC homography verification
- Reprojection threshold of 5.0 pixels

Five borderline examples were manually inspected.

The observed ORB inlier counts were:

```text
11
16
20
26
47
```

All five inspected examples represented the same underlying image.

Based on this empirical investigation, **11 ORB inliers** was adopted as the project's working threshold.

The final duplicate criterion used:

```text
pHash distance = 0
AND
ORB inliers >= 11
```

---

## Confirmed Duplicate Relationships

Using the combined pHash and ORB verification process, the analysis identified:

**293 confirmed duplicate relationships**

| Split Combination | Relationships |
|---|---:|
| Train ↔ Test | 148 |
| Train ↔ Validation | 93 |
| Validation ↔ Test | 52 |
| **Total** | **293** |

These relationships were grouped into connected duplicate groups.

A total of:

**264 duplicate groups**

were identified.

The largest groups contained up to six related images.

---

# Dataset Cleanup

The confirmed duplicate relationships were not blindly deleted.

Instead, a retention policy was established:

```text
Train > Test > Validation
```

The goal was to retain one representative copy while preventing the same underlying image from appearing across different dataset splits.

The final quarantine process resulted in:

| Action | Images |
|---|---:|
| Test removed when Train retained | 138 |
| Validation removed when Train retained | 83 |
| Validation removed when Test retained | 49 |
| Train duplicates removed within Train | 15 |
| **Total quarantined** | **285** |

The quarantine operation produced:

- 285 images moved
- 0 missing files
- 0 file-operation errors

The quarantined images were retained as a backup rather than permanently deleted.

---

# Final Clean Dataset

After duplicate cleanup and manual redistribution, the working dataset contains:

**10,387 images**

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Buffalo | 571 | 271 | 271 | 1,113 |
| Camel | 611 | 292 | 288 | 1,191 |
| Cat | 611 | 610 | 612 | 1,833 |
| Chicken | 583 | 227 | 269 | 1,079 |
| Donkey | 759 | 85 | 192 | 1,036 |
| Goat | 742 | 98 | 192 | 1,032 |
| Horse | 761 | 233 | 234 | 1,228 |
| Pig | 644 | 90 | 182 | 916 |
| Sheep | 689 | 88 | 182 | 959 |
| **Total** | **5,971** | **1,994** | **2,422** | **10,387** |

The training dataset remains somewhat imbalanced.

Instead of deleting valid images simply to force equal class counts, the project uses sampling and augmentation strategies to reduce the impact of class imbalance during training.

---

# Data Loading and Preprocessing

The data-loading pipeline is implemented in:

```text
scripts/dataset.py
```

The pipeline uses PyTorch and `torchvision.datasets.ImageFolder`.

For the complete preprocessing methodology and dataset-quality investigation, see:

**[`notebooks/README.md`](notebooks/README.md)**

---

## Image Size

Images are processed at:

```text
224 × 224
```

This size is compatible with the pretrained ResNet50 and EfficientNet-B0 architectures used in the project.

---

## Training Augmentation

The current training pipeline uses:

- Random resized crop
- Random horizontal flip
- Random rotation
- Color jitter
- Conversion to tensor
- ImageNet normalization

The transformations introduce moderate variation while preserving the animal's visual characteristics.

---

## Validation and Test Preprocessing

Validation and test images use deterministic preprocessing:

```text
Resize → Center Crop → Tensor → ImageNet Normalization
```

Random augmentation is not applied to validation or test images.

This keeps evaluation consistent.

---

## ImageNet Normalization

The pretrained models use ImageNet normalization:

```text
Mean = [0.485, 0.456, 0.406]

Std = [0.229, 0.224, 0.225]
```

---

## Class Imbalance Handling

The training dataset is not perfectly balanced.

A `WeightedRandomSampler` is therefore used during training.

Classes with fewer training images receive larger sampling weights.

This allows smaller classes to be sampled more frequently without creating physical duplicate files in the dataset.

Validation and test datasets are evaluated using their natural distributions.

---

# Model Development

Two pretrained CNN architectures are being investigated.

## ResNet50

ResNet50 is initialized using:

```python
ResNet50_Weights.IMAGENET1K_V2
```

The original 1000-class ImageNet classifier is replaced with a classifier containing nine output classes.

The model is implemented in:

```text
scripts/models.py
```

---

## EfficientNet-B0

EfficientNet-B0 is initialized using:

```python
EfficientNet_B0_Weights.IMAGENET1K_V1
```

Its original ImageNet classifier is replaced with a nine-class classifier.

The model is also implemented in:

```text
scripts/models.py
```

The V1/V2 difference refers to the pretrained weight versions provided by torchvision, not different model architectures.

---

# Training Strategy

The training process is divided into two major stages.

---

## Stage 1 — Transfer Learning

In Stage 1:

1. Load ImageNet pretrained weights.
2. Freeze the pretrained backbone.
3. Replace the original ImageNet classifier.
4. Train only the new nine-class classification head.
5. Monitor validation performance.
6. Save the best checkpoint.

This allows the newly initialized classifier to adapt to the animal dataset before modifying the pretrained feature extractor.

---

## Stage 2 — Fine-Tuning

The next stage will:

1. Load the best Stage 1 checkpoint.
2. Unfreeze selected later backbone layers.
3. Use a lower learning rate.
4. Continue training.
5. Monitor validation Macro F1.
6. Save the best fine-tuned checkpoint.
7. Use early stopping when validation performance stops improving.

Stage 2 has not yet been completed.

---

# Training Configuration

The current baseline uses:

| Parameter | Value |
|---|---:|
| Image size | 224 × 224 |
| Batch size | 32 |
| Stage 1 epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Early stopping patience | 3 |
| Number of classes | 9 |

Training is performed using the available NVIDIA GPU when CUDA is available.

---

# GPU Environment

The project was moved to a dedicated Python 3.12 environment:

```text
classification-gpu
```

The working GPU configuration is:

```text
GPU:
NVIDIA GeForce RTX 4060 Laptop GPU

PyTorch:
2.14.0+cu126

CUDA available:
True

PyTorch CUDA:
12.6
```

The NVIDIA driver supports the CUDA runtime required by the PyTorch installation.

---

# Current Results

## ResNet50 Stage 1

The first completed model-training stage used a frozen ImageNet-pretrained ResNet50 and trained only its classification head.

Training was performed for 10 epochs.

| Epoch | Train Accuracy | Validation Accuracy | Validation Macro F1 |
|---:|---:|---:|---:|
| 1 | 85.20% | 93.58% | 0.8874 |
| 2 | 92.40% | 95.29% | 0.9170 |
| 3 | 94.17% | 95.24% | 0.9176 |
| 4 | 94.86% | 95.19% | 0.9151 |
| 5 | 95.34% | 95.49% | 0.9199 |
| 6 | 96.28% | 95.34% | 0.9172 |
| 7 | 95.86% | 95.49% | 0.9219 |
| **8** | **96.70%** | **95.89%** | **0.9273** |
| 9 | 96.60% | 95.69% | 0.9249 |
| 10 | 96.97% | 95.69% | 0.9250 |

The best Stage 1 checkpoint was obtained at:

```text
Epoch: 8

Validation Accuracy:
95.89%

Validation Macro F1:
0.9273
```

Checkpoint:

```text
models/resnet50_stage1_best.pth
```

The test set was not used during this training stage.

---

# Results and Outputs

The project stores generated results and trained models in dedicated directories:

```text
results/
models/
```

Current important output:

```text
models/
└── resnet50_stage1_best.pth
```

The checkpoint stores:

- Epoch number
- Model state
- Optimizer state
- Validation Macro F1
- Validation accuracy

---

## Planned Evaluation Outputs

The following outputs will be added after the remaining model stages:

- Test accuracy
- Test precision
- Test recall
- Macro F1
- Weighted F1
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix
- Model comparison
- Error analysis
- Explainability visualizations

No final test-set result is reported yet because the final model comparison has not been completed.

---

# Setup and Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Tusharparihast/Domestic-Animal-Classification.git
```

Navigate into the project:

```bash
cd Domestic-Animal-Classification
```

---

## 2. Create a Python Environment

Python 3.12 is currently used for the GPU-enabled environment.

On Windows:

```bash
py -3.12 -m venv classification-gpu
```

Activate it:

```powershell
.\classification-gpu\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

Install the project dependencies:

```bash
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the appropriate PyTorch and torchvision CUDA builds separately according to the official PyTorch installation instructions.

The current environment uses:

```text
torch 2.14.0+cu126
torchvision 0.29.0
```

---

# How to Run

## Test the Dataset Pipeline

From the project root:

```bash
python scripts\dataset.py
```

Expected output includes:

```text
Dataset loaded successfully.

Training images: 5971
Validation images: 1994
Test images: 2422
```

The class list should contain:

```text
['Buffalo',
 'Camel',
 'Cat',
 'Chicken',
 'Donkey',
 'Goat',
 'Horse',
 'Pig',
 'Sheep']
```

---

## Train ResNet50 Stage 1

Run:

```bash
python scripts\train.py
```

The script:

1. Loads the dataset.
2. Creates the weighted sampler.
3. Loads ImageNet-pretrained ResNet50.
4. Freezes the backbone.
5. Trains the classification head.
6. Evaluates on the validation set.
7. Tracks validation Macro F1.
8. Saves the best checkpoint.

---

# Configuration

Main training configuration is currently defined in:

```text
scripts/train.py
```

Dataset configuration is defined in:

```text
scripts/dataset.py
```

Model configuration is defined in:

```text
scripts/models.py
```

Important parameters include:

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3
NUM_CLASSES = 9
```

These values may be adjusted during later experiments.

---

# Modules Overview

## `scripts/dataset.py`

Responsible for:

- Dataset loading
- Image transformations
- Image normalization
- Weighted sampling
- DataLoader creation

---

## `scripts/models.py`

Responsible for:

- ResNet50 creation
- EfficientNet-B0 creation
- Loading ImageNet pretrained weights
- Replacing the original classifier

---

## `scripts/train.py`

Responsible for:

- Model training
- Validation
- Loss calculation
- Accuracy calculation
- Macro F1 calculation
- Checkpointing
- Early stopping

---

## `scripts/evaluate.py`

Reserved for model evaluation and final test-set analysis.

Planned responsibilities include:

- Test-set evaluation
- Classification metrics
- Confusion matrix
- Per-class performance
- Model comparison

---

## `scripts/utils.py`

Contains reusable helper functionality for the project.

---

# Troubleshooting

## CUDA is not available

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected:

```text
True
```

Then check:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected GPU:

```text
NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## Windows DataLoader Multiprocessing Error

During development, Windows multiprocessing caused issues when multiple DataLoader workers were used.

The working configuration for the training environment uses:

```python
NUM_WORKERS = 0
```

This avoids multiprocessing-related issues during local training.

---

## Pillow Transparency Warning

Some images produced a warning similar to:

```text
Palette images with Transparency expressed in bytes should be converted to RGBA images
```

This is a Pillow warning rather than a training failure.

The model training continued successfully.

Image-mode normalization can be made more explicit in a later preprocessing revision if required.

---

## Dataset Path Errors

The dataset directories must follow:

```text
data/
├── train/
├── validation/
└── test/
```

Each split should contain one folder per animal class.

For example:

```text
data/
└── train/
    ├── Buffalo/
    ├── Camel/
    ├── Cat/
    ├── Chicken/
    ├── Donkey/
    ├── Goat/
    ├── Horse/
    ├── Pig/
    └── Sheep/
```

---

# Future Work

## 1. Improve Training Monitoring

Add progress bars using `tqdm` for:

- Training batches
- Validation batches
- Current loss

---

## 2. ResNet50 Fine-Tuning

Load the best Stage 1 checkpoint and fine-tune selected later backbone layers using a lower learning rate.

---

## 3. EfficientNet-B0 Training

Run the same two-stage strategy:

```text
Stage 1 → Transfer Learning

Stage 2 → Fine-Tuning
```

---

## 4. Final Test Evaluation

Evaluate the final candidate models on the untouched test set.

Metrics will include:

- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1
- Per-class metrics

---

## 5. Confusion Matrix

Analyze which animal classes are most frequently confused.

---

## 6. Model Comparison

Compare ResNet50 and EfficientNet-B0 using the same evaluation protocol.

The comparison will be based on the measured experimental results after both models have been evaluated.

---

## 7. Error Analysis

Inspect incorrectly classified images to identify:

- Similar-looking animals
- Group images
- Difficult viewpoints
- Occlusion
- Background dependence
- Low-quality images
- Ambiguous examples

---

## 8. Explainability

Grad-CAM or a similar explainability method may be used to inspect which regions of the image influence model predictions.

---

# Documentation

Detailed dataset preprocessing and quality-assurance work is documented separately in:

**[`notebooks/README.md`](notebooks/README.md)**

This documentation contains the detailed work related to:

- Dataset inspection
- Dataset statistics
- Image quality
- Image dimensions
- Color modes
- File formats
- Duplicate detection
- Near-duplicate detection
- pHash analysis
- ORB verification
- Leakage investigation
- Duplicate grouping
- Dataset cleanup
- Dataset redistribution decisions

The root `README.md` provides the high-level project overview and current model-development status, while `notebooks/README.md` contains the detailed dataset investigation and preprocessing methodology.

---

# References

- [PyTorch Documentation](https://pytorch.org/docs/)

- [Torchvision Documentation](https://pytorch.org/vision/stable/)

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

- [ImageHash](https://github.com/JohannesBuchner/imagehash)

- [OpenCV ORB Documentation](https://docs.opencv.org/)

---

# Author

**Tushar Parihast**

AI Engineering | Machine Learning | Automation

GitHub: [@Tusharparihast](https://github.com/Tusharparihast)

Project Repository: [Domestic Animal Classification](https://github.com/Tusharparihast/Domestic-Animal-Classification)

---

# Project Development Workflow

The project follows an iterative workflow:

```text
Understand Dataset
        ↓
Check Data Quality
        ↓
Detect Exact Duplicates
        ↓
Detect Near Duplicates
        ↓
Investigate Cross-Split Leakage
        ↓
Verify Strong Candidates with ORB
        ↓
Clean Dataset
        ↓
Redistribute Dataset
        ↓
Build Data Pipeline
        ↓
Train Transfer-Learning Model
        ↓
Fine-Tune Model
        ↓
Evaluate on Test Set
        ↓
Compare Models
        ↓
Analyze Errors
        ↓
Document Results
```

The project follows the principle:

**Implement → Test → Analyze → Document → Continue**