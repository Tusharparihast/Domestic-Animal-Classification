# Domestic Animal Classification

A computer vision project for classifying domestic animals into 9 categories using deep learning and transfer learning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Model Development](#model-development)
- [Training Strategy](#training-strategy)
- [ResNet50 Training](#resnet50-training)
- [ResNet50 Evaluation](#resnet50-evaluation)
- [ResNet50 Error Analysis](#resnet50-error-analysis)
- [Visualizations](#visualizations)
- [Results and Outputs](#results-and-outputs)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Modules Overview](#modules-overview)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Author](#author)

---

## Project Overview

The **Domestic Animal Classification** project uses deep learning and transfer learning to classify images of domestic animals into 9 classes.

The project includes:

- Dataset preparation and quality analysis
- Exact duplicate detection
- Near-duplicate detection
- Cross-split leakage investigation
- Dataset cleanup
- Image preprocessing and augmentation
- Class imbalance handling
- Transfer learning
- Fine-tuning
- Model evaluation
- Error analysis
- Training and evaluation visualization

The project currently includes a trained and evaluated **ResNet50** model.

---

## Project Structure

```text
Domestic-Animal-Classification/
│
│
├── pre-processing/
|   ├── 01_dataset_overview.ipynb
|   ├── 02_check_duplicate_img.ipynb
|   ├── 03_check_near_duplicates.ipynb
|   ├── 04_review_near_duplicates.ipynb
|   ├── 05_Class_Distribution_Analysis.ipynb
|   ├── README.md
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
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Features

- 9-class domestic animal image classification
- Dataset quality verification
- Exact duplicate detection
- Near-duplicate detection using perceptual hashing
- Cross-split leakage investigation
- ORB-based duplicate verification
- Dataset cleanup and quarantine
- Class imbalance handling
- Image augmentation
- ImageNet pretrained models
- Transfer learning
- Fine-tuning
- Early stopping
- Weighted random sampling
- Classification metrics
- Confusion matrix analysis
- Training history visualization

---

## Dataset and Preprocessing

The project uses a cleaned dataset of **10,387 images** across 9 domestic animal classes:

- Buffalo
- Camel
- Cat
- Chicken
- Donkey
- Goat
- Horse
- Pig
- Sheep

The dataset was prepared through dataset organization, split verification, image quality checks, duplicate and near-duplicate detection, cross-split leakage analysis, class distribution analysis, and dataset cleanup.

### Final Dataset

| Split | Images |
|---|---:|
| Train | 5,971 |
| Validation | 1,994 |
| Test | 2,422 |
| **Total** | **10,387** |

Detailed information about the dataset, preparation process, quality checks, duplicate detection, near-duplicate analysis, leakage prevention, cleanup, and class distribution is documented in:

**[`pre-processing/README.md`](pre-processing/README.md)**

The preprocessing notebooks contain the detailed dataset analysis and implementation.

### Data Loading and Preprocessing

The cleaned dataset is loaded using PyTorch `ImageFolder`.

Training preprocessing includes:

- Random resized crop
- Random horizontal flip
- Random rotation
- Color jitter
- ImageNet normalization
- Weighted random sampling for class imbalance

Validation and test data use deterministic preprocessing:

- Resize
- Center crop
- ImageNet normalization

The implementation is available in:

```text
scripts/dataset.py
```

---

## Model Development

The project uses pretrained convolutional neural networks with ImageNet weights.

The implemented architectures include:

- **ResNet50**
- **EfficientNet-B0**

The model definitions are available in:

```text
scripts/models.py
```

For this classification task, the original ImageNet classification layer is replaced with a classifier containing **9 output classes**.

---

## Training Strategy

The ResNet50 model was trained using a two-stage transfer learning approach.

### Stage 1 — Transfer Learning

- ImageNet pretrained ResNet50
- Backbone frozen
- Final classification layer trained
- Weighted random sampling used during training
- Validation Macro F1 used for checkpoint selection
- Early stopping applied

### Stage 2 — Fine-Tuning

- Best Stage 1 checkpoint loaded
- ResNet50 `layer4` unfrozen
- Final classification layer trained together with `layer4`
- Lower learning rate used
- Validation Macro F1 monitored
- Early stopping applied

---

## ResNet50 Training

### Stage 1

The best Stage 1 checkpoint was obtained at **epoch 10**.

| Metric | Value |
|---|---:|
| Training Accuracy | 96.78% |
| Validation Accuracy | 96.04% |
| Validation Macro F1 | 0.9305 |
| Validation Loss | 0.1314 |

### Stage 2

The best Stage 2 checkpoint was obtained at **epoch 4**.

| Metric | Value |
|---|---:|
| Training Accuracy | 99.23% |
| Validation Accuracy | 96.64% |
| Validation Macro F1 | 0.9450 |
| Validation Loss | 0.1104 |

Early stopping was applied after validation performance stopped improving.

The best Stage 2 model was saved as:

```text
models/resnet50_stage2_best.pth
```

---

## ResNet50 Evaluation

The final ResNet50 model was evaluated on the test set containing **2,422 images**.

### Overall Results

| Metric | Score |
|---|---:|
| Accuracy | 95.91% |
| Macro Precision | 95.10% |
| Macro Recall | 94.76% |
| Macro F1 | 94.89% |
| Weighted F1 | 95.91% |

### Per-Class Results

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Buffalo | 0.9564 | 0.9705 | 0.9634 |
| Camel | 0.9458 | 0.9688 | 0.9571 |
| Cat | 1.0000 | 0.9984 | 0.9992 |
| Chicken | 0.9925 | 0.9851 | 0.9888 |
| Donkey | 0.9714 | 0.8854 | 0.9264 |
| Goat | 0.8788 | 0.9062 | 0.8923 |
| Horse | 0.8943 | 0.9402 | 0.9167 |
| Pig | 0.9728 | 0.9835 | 0.9781 |
| Sheep | 0.9474 | 0.8901 | 0.9178 |

---

## ResNet50 Error Analysis

The confusion matrix was used to inspect classification errors.

Observed confusion patterns include:

- Sheep → Goat: 14 images
- Donkey → Horse: 13 images
- Donkey → Goat: 6 images
- Goat → Sheep: 5 images
- Horse → Buffalo: 5 images
- Horse → Camel: 5 images
- Camel → Horse: 5 images

The confusion matrix is stored in:

```text
results/ResNet_csv/resnet50_confusion_matrix.csv
```

---

## Visualizations

Training and evaluation visualizations are stored in:

```text
results/visualizations/
```

The generated visualizations include:

- ResNet50 Stage 1 loss
- ResNet50 Stage 1 accuracy
- ResNet50 Stage 1 Macro F1
- ResNet50 Stage 2 loss
- ResNet50 Stage 2 accuracy
- ResNet50 Stage 2 Macro F1
- ResNet50 confusion matrix

The visualization implementation is available in:

```text
scripts/visualization.py
```

---

## Results and Outputs

The project stores ResNet50 training and evaluation outputs under the `results/` directory.

```text
results/
├── ResNet_csv/
│   ├── resnet50_stage1_history.csv
│   ├── resnet50_stage2_history.csv
│   ├── resnet50_confusion_matrix.csv
│   └── resnet50_test_report.txt
│
└── visualizations/
    ├── resnet50_stage1_loss.png
    ├── resnet50_stage1_accuracy.png
    ├── resnet50_stage1_macro_f1.png
    ├── resnet50_stage2_loss.png
    ├── resnet50_stage2_accuracy.png
    ├── resnet50_stage2_macro_f1.png
    └── resnet50_confusion_matrix.png
```

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Tusharparihast/Domestic-Animal-Classification.git
cd Domestic-Animal-Classification
```

### 2. Create a Virtual Environment

```bash
python -m venv classification
```

Activate the environment on Windows:

```bash
classification\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU-enabled PyTorch, install the appropriate CUDA-compatible PyTorch and torchvision versions separately.

---

## How to Run

### Check the Dataset Pipeline

```bash
python scripts/dataset.py
```

### Train the Model

```bash
python scripts/train.py
```

### Evaluate the Model

```bash
python scripts/evaluate.py
```

### Generate Visualizations

```bash
python scripts/visualization.py
```

---

## Configuration

The main dataset and training configuration is defined in:

```text
scripts/dataset.py
```

Important parameters include:

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
```

ImageNet normalization is used:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

---

## Modules Overview

### `scripts/dataset.py`

Handles:

- Dataset loading
- Image transformations
- DataLoader creation
- Weighted random sampling
- Class imbalance handling

### `scripts/models.py`

Contains the pretrained model definitions:

- ResNet50
- EfficientNet-B0

### `scripts/train.py`

Handles:

- Training loop
- Validation
- Loss calculation
- Accuracy calculation
- Macro F1 calculation
- Checkpoint saving
- Early stopping

### `scripts/evaluate.py`

Handles model evaluation and generates classification metrics.

### `scripts/utils.py`

Contains reusable utility functions used throughout the project.

---

## Troubleshooting

### Windows DataLoader Error

If PyTorch DataLoader multiprocessing causes errors on Windows, use:

```python
NUM_WORKERS = 0
```

in:

```text
scripts/dataset.py
```

### Pillow Transparency Warning

A warning related to palette-based images with transparency may appear during image loading.

This is a warning and does not necessarily indicate a corrupted image or failed training process.

---

## References

- PyTorch Documentation
- Torchvision Documentation
- Scikit-learn Documentation
- ImageHash Documentation
- ResNet: Deep Residual Learning for Image Recognition
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

---

## Author

**Tushar Parihast**

GitHub: [@Tusharparihast](https://github.com/Tusharparihast)

Repository: [Domestic-Animal-Classification](https://github.com/Tusharparihast/Domestic-Animal-Classification)
