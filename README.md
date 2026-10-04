# Domestic Animal Classification

A deep learning project for classifying domestic animals into **9 categories using a custom-prepared dataset**, developed through dataset curation, quality analysis, transfer learning, and fine-tuning of pretrained CNN architectures.

The project focuses not only on model performance, but also on **building and validating a reliable custom dataset**, including image quality checks, duplicate and near-duplicate removal, cross-split leakage prevention, class distribution analysis, class imbalance handling, reproducible training, and systematic model evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Models](#models)
- [Training Strategy](#training-strategy)
- [ResNet50 Training](#resnet50-training)
- [EfficientNet-B0 Training](#efficientnet-b0-training)
- [Final Results](#final-results)
- [Error Analysis](#error-analysis)
- [Visualizations](#visualizations)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Scripts](#scripts)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Author](#author)

---

## Overview

The project develops an image-classification system for identifying the following domestic animals:

- Buffalo
- Camel
- Cat
- Chicken
- Donkey
- Goat
- Horse
- Pig
- Sheep

Two ImageNet-pretrained CNN architectures are implemented and compared:

- **ResNet50**
- **EfficientNet-B0**

Both models use the same overall two-stage workflow:

1. **Transfer learning** — freeze the pretrained backbone and train the new classification head.
2. **Fine-tuning** — unfreeze selected later layers and continue training with a lower learning rate.

The final models are evaluated on a held-out test set using accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, per-class metrics, and confusion matrices.

---

## Project Structure

The repository keeps the tracked structure focused on preprocessing notebooks, documentation, source code, and final visualizations. The dataset and trained checkpoints are excluded through `.gitignore`.

```text
Domestic-Animal-Classification/
│
├── pre-processing/
│   ├── 01_dataset_overview.ipynb
│   ├── 02_check_duplicate_img.ipynb
│   ├── 03_check_near_duplicates.ipynb
│   ├── 04_review_near_duplicates.ipynb
│   ├── 05_Class_Distribution_Analysis.ipynb
│   ├── README.md
│   └── images/
│
├── results/
│   └── visualizations/
│
├── scripts/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── models.py
│   ├── train.py
│   └── visualization.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Features

- 9-class domestic animal image classification
- Dataset quality assurance
- Corrupted-image checking
- Image format, dimension, and color-mode analysis
- Exact duplicate detection
- Near-duplicate detection with pHash
- Cross-split leakage investigation
- ORB feature-based verification
- Empirical ORB threshold validation
- Duplicate grouping and cleanup
- Manual class redistribution
- Weighted random sampling
- Training-time augmentation
- ImageNet normalization
- Transfer learning
- CNN fine-tuning
- GPU acceleration
- Early stopping
- Validation Macro F1-based checkpoint selection
- Classification metrics
- Confusion matrix analysis
- Model comparison
- Training and evaluation visualizations

---

## Dataset and Preprocessing

The final cleaned dataset contains **10,387 images** across the nine animal classes.

| Split | Images |
|---|---:|
| Train | 5,971 |
| Validation | 1,994 |
| Test | 2,422 |
| **Total** | **10,387** |

All detailed dataset preparation and quality-assurance work is intentionally documented in the dedicated preprocessing README rather than repeated here.

### Dataset Documentation

See **[`pre-processing/README.md`](pre-processing/README.md)** for the complete dataset workflow, including:

- Original dataset distribution
- Image quality and corruption checks
- File formats, dimensions, and color modes
- Exact duplicate detection
- pHash near-duplicate detection
- Cross-split leakage investigation
- ORB verification
- Empirical ORB threshold validation
- Connected duplicate-group analysis
- Quarantine and cleanup decisions
- Manual redistribution of clean images
- Training-time class balancing

The preprocessing notebooks implement the analysis described in that document.

---

## Models

### ResNet50

ResNet50 is initialized with the torchvision ImageNet weights:

```python
ResNet50_Weights.IMAGENET1K_V2
```

The original ImageNet classifier is replaced with a **9-class classification layer**.

### EfficientNet-B0

EfficientNet-B0 is initialized with:

```python
EfficientNet_B0_Weights.IMAGENET1K_V1
```

Its original ImageNet classifier is also replaced with a **9-class classification layer**.

The V1/V2 difference refers to the pretrained weight versions provided by torchvision, not different model architectures.

Model definitions are implemented in:

```text
scripts/models.py
```

---

## Training Strategy

Both architectures use the same two-stage transfer-learning strategy.

### Stage 1 — Transfer Learning

- Load ImageNet-pretrained weights.
- Freeze the pretrained backbone.
- Replace the original classifier with a 9-class classifier.
- Train only the classification head.
- Use weighted random sampling for class imbalance.
- Monitor validation Macro F1.
- Save the best checkpoint.
- Apply early stopping.

### Stage 2 — Fine-Tuning

- Load the best Stage 1 checkpoint.
- Keep earlier backbone layers frozen.
- Fine-tune later feature layers and the classifier.
- Continue with a lower learning rate.
- Monitor validation Macro F1.
- Save the best checkpoint.
- Apply early stopping.

### Fine-Tuned Components

| Model | Fine-tuned components |
|---|---|
| ResNet50 | `layer4` + classifier |
| EfficientNet-B0 | `features[6:]` + classifier |

### Training Configuration

| Parameter | Stage 1 | Stage 2 |
|---|---:|---:|
| Learning rate | `1e-3` | `1e-4` |
| Maximum epochs | 10 | 10 |
| Early stopping patience | 3 | 3 |
| Batch size | 32 | 32 |
| Input size | 224 × 224 | 224 × 224 |

Validation **Macro F1** is used as the primary checkpoint-selection metric because the classes are not perfectly balanced.

---

## ResNet50 Training

### Stage 1

The best ResNet50 Stage 1 checkpoint was obtained at **epoch 8**.

| Metric | Result |
|---|---:|
| Training Accuracy | 96.73% |
| Validation Accuracy | 96.04% |
| Validation Macro F1 | 0.9299 |
| Validation Loss | 0.1332 |

### Stage 2

The best ResNet50 Stage 2 checkpoint was obtained at **epoch 2**.

| Metric | Result |
|---|---:|
| Training Accuracy | 98.43% |
| Validation Accuracy | 96.49% |
| Validation Macro F1 | 0.9425 |
| Validation Loss | 0.1088 |

The Stage 2 checkpoint was used for the final test evaluation.

---

## EfficientNet-B0 Training

### Stage 1

The best EfficientNet-B0 Stage 1 checkpoint was obtained at **epoch 5**.

| Metric | Result |
|---|---:|
| Training Accuracy | 89.37% |
| Validation Accuracy | 92.43% |
| Validation Macro F1 | 0.8828 |
| Validation Loss | 0.2520 |

### Stage 2

The best EfficientNet-B0 Stage 2 checkpoint was obtained at **epoch 3**.

| Metric | Result |
|---|---:|
| Training Accuracy | 95.70% |
| Validation Accuracy | 95.79% |
| Validation Macro F1 | 0.9325 |
| Validation Loss | 0.1264 |

The Stage 2 checkpoint was used for the final test evaluation.

---

## Final Results

The final Stage 2 models were evaluated on the held-out **2,422-image test set**.

### Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| **ResNet50** | **95.95%** | **95.04%** | **94.92%** | **94.96%** | **95.94%** |
| EfficientNet-B0 | 95.54% | 94.72% | 94.60% | 94.61% | 95.55% |

ResNet50 performed better across every reported aggregate test metric.

### ResNet50 Advantage

| Metric | Difference |
|---|---:|
| Accuracy | +0.41 percentage points |
| Macro Precision | +0.32 percentage points |
| Macro Recall | +0.32 percentage points |
| Macro F1 | +0.35 percentage points |
| Weighted F1 | +0.39 percentage points |

### Final Model Selection

Based on the final test-set evaluation, **ResNet50 is the selected model for this dataset**.

---

## Per-Class Test Performance

### ResNet50

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Buffalo | 0.9568 | 0.9815 | 0.9690 |
| Camel | 0.9418 | 0.9549 | 0.9483 |
| Cat | 1.0000 | 0.9967 | 0.9984 |
| Chicken | 0.9851 | 0.9851 | 0.9851 |
| Donkey | 0.9560 | 0.9062 | 0.9305 |
| Goat | 0.9091 | 0.8854 | 0.8971 |
| Horse | 0.9198 | 0.9316 | 0.9257 |
| Pig | 0.9524 | 0.9890 | 0.9704 |
| Sheep | 0.9326 | 0.9121 | 0.9222 |

### EfficientNet-B0

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Buffalo | 0.9362 | 0.9742 | 0.9548 |
| Camel | 0.9456 | 0.9653 | 0.9553 |
| Cat | 0.9983 | 0.9886 | 0.9934 |
| Chicken | 0.9776 | 0.9740 | 0.9758 |
| Donkey | 0.9719 | 0.9010 | 0.9351 |
| Goat | 0.8647 | 0.9323 | 0.8972 |
| Horse | 0.9261 | 0.9103 | 0.9181 |
| Pig | 0.9620 | 0.9725 | 0.9672 |
| Sheep | 0.9422 | 0.8956 | 0.9183 |

---

## Error Analysis

The final confusion matrices show that the remaining errors are concentrated among visually similar animal classes.

### ResNet50

| True Class | Predicted Class | Errors |
|---|---|---:|
| Sheep | Goat | 9 |
| Goat | Sheep | 9 |
| Donkey | Horse | 9 |
| Horse | Camel | 7 |
| Camel | Horse | 5 |

### EfficientNet-B0

| True Class | Predicted Class | Errors |
|---|---|---:|
| Sheep | Goat | 12 |
| Horse | Camel | 8 |
| Donkey | Horse | 7 |
| Goat | Sheep | 5 |
| Camel | Horse | 4 |

The most challenging distinctions are therefore **Goat vs Sheep**, **Donkey vs Horse**, and **Horse vs Camel**.

Cats were among the strongest-performing classes for both models.

---

## Visualizations

The repository keeps the final visualization outputs under:

```text
results/visualizations/
```

### ResNet50

- `resnet50_stage1_loss.png`
- `resnet50_stage1_accuracy.png`
- `resnet50_stage1_macro_f1.png`
- `resnet50_stage2_loss.png`
- `resnet50_stage2_accuracy.png`
- `resnet50_stage2_macro_f1.png`
- `resnet50_confusion_matrix.png`

### EfficientNet-B0

- `efficientnet_b0_stage1_loss.png`
- `efficientnet_b0_stage1_accuracy.png`
- `efficientnet_b0_stage1_macro_f1.png`
- `efficientnet_b0_stage2_loss.png`
- `efficientnet_b0_stage2_accuracy.png`
- `efficientnet_b0_stage2_macro_f1.png`
- `efficientnet_b0_confusion_matrix.png`

### Model Comparison

- `model_comparison.png`

The visualization pipeline is implemented in:

```text
scripts/visualization.py
```

Intermediate training-history CSV files, classification reports, confusion-matrix CSVs, and model-comparison CSVs are generated locally during experimentation and are excluded from the tracked repository outputs.

---

## Setup

### Clone the Repository

```bash
git clone https://github.com/Tusharparihast/Domestic-Animal-Classification.git
cd Domestic-Animal-Classification
```

### Create a Python Environment

The completed GPU experiments used Python 3.12.

On Windows:

```bash
py -3.12 -m venv classification-gpu
```

Activate it:

```powershell
classification-gpu\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The completed GPU environment used a CUDA-enabled PyTorch build with an NVIDIA GeForce RTX 4060 Laptop GPU.

---

## How to Run

The scripts expect the local dataset to be available in the following structure:

```text
data/
├── train/
├── validation/
└── test/
```

Each split should contain one folder for each of the nine animal classes.

### Check Dataset Loading

```bash
python scripts/dataset.py
```

The final working dataset contains:

```text
Training images: 5971
Validation images: 1994
Test images: 2422
```

### Train Both Models

The consolidated training script runs all four stages sequentially:

1. ResNet50 Stage 1
2. ResNet50 Stage 2
3. EfficientNet-B0 Stage 1
4. EfficientNet-B0 Stage 2

Run:

```bash
python scripts/train.py
```

### Evaluate Both Models

```bash
python scripts/evaluate.py
```

The evaluation script loads the best Stage 2 checkpoints, evaluates the test set, generates classification reports and confusion matrices, and produces the model-comparison results.

### Generate Visualizations

```bash
python scripts/visualization.py
```

This generates the training curves, confusion matrices, and model-comparison visualization under:

```text
results/visualizations/
```

---

## Configuration

### Dataset Configuration

Implemented in:

```text
scripts/dataset.py
```

Key settings used in the completed experiments:

```python
Image_size = 224
Batch_size = 32
Num_workers = 0
```

ImageNet normalization:

```python
Imagenet_mean = [0.485, 0.456, 0.406]
Imagenet_std = [0.229, 0.224, 0.225]
```

Training augmentation includes:

- Random horizontal flip
- Random rotation
- Random resized crop
- Color jitter

Validation and test images use deterministic preprocessing without random augmentation.

### Training Configuration

Implemented in:

```text
scripts/train.py
```

Key settings:

```python
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
STAGE1_LEARNING_RATE = 1e-3
STAGE2_LEARNING_RATE = 1e-4
```

---

## Scripts

### `scripts/dataset.py`

Responsible for:

- Loading Train, Validation, and Test datasets
- Applying image transformations
- ImageNet normalization
- Weighted random sampling
- Creating PyTorch DataLoaders

### `scripts/models.py`

Responsible for:

- Creating ResNet50
- Creating EfficientNet-B0
- Loading ImageNet pretrained weights
- Replacing the original classification heads

### `scripts/train.py`

The consolidated training pipeline for both models.

It handles:

- Stage 1 transfer learning
- Stage 2 fine-tuning
- Training and validation
- Accuracy calculation
- Macro F1 calculation
- Best-checkpoint selection
- Early stopping
- Training-history CSV generation

### `scripts/evaluate.py`

Responsible for:

- Loading the best Stage 2 checkpoints
- Test-set prediction
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Classification reports
- Confusion matrices
- Model comparison

### `scripts/visualization.py`

Responsible for:

- Stage 1 loss plots
- Stage 1 accuracy plots
- Stage 1 Macro F1 plots
- Stage 2 loss plots
- Stage 2 accuracy plots
- Stage 2 Macro F1 plots
- Confusion matrix visualizations
- Model-comparison visualization

---

## Troubleshooting

### CUDA Is Not Available

Check CUDA availability with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The completed experiments were run with an NVIDIA GeForce RTX 4060 Laptop GPU using a CUDA-enabled PyTorch build.

### Windows DataLoader Issues

The working Windows configuration uses:

```python
Num_workers = 0
```

This avoids multiprocessing-related DataLoader issues encountered during development.

### Pillow Transparency Warning

Some images may produce a Pillow warning related to palette-based transparency.

This warning did not indicate corrupted data and did not prevent the training pipeline from running successfully.

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Torchvision Documentation](https://pytorch.org/vision/stable/)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [ImageHash](https://github.com/JohannesBuchner/imagehash)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## Author

**Tushar Parihast**

GitHub: [@Tusharparihast](https://github.com/Tusharparihast)

Repository: [Domestic-Animal-Classification](https://github.com/Tusharparihast/Domestic-Animal-Classification)
