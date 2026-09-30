# Domestic Animal Classification

A deep learning project for classifying images of domestic animals into nine
different categories using transfer learning and pretrained convolutional
neural networks.

## Overview

This project explores the development of a computer vision system capable of
automatically identifying domestic animals from images. The classifier
recognizes nine classes:

- Buffalo
- Camel
- Cat
- Chicken
- Donkey
- Goat
- Horse
- Pig
- Sheep

The project uses pretrained deep learning models and transfer learning to
adapt models originally trained on ImageNet to the domestic-animal
classification task.

Rather than focusing only on achieving a high classification accuracy, the
project places particular emphasis on **dataset quality, prevention of data
leakage, class imbalance, reproducible training, and detailed model
evaluation**.

## Why This Project?

Image classification performance depends not only on the model architecture
but also on the quality and organization of the dataset used for training.

During the initial investigation of the dataset, issues such as class
imbalance, duplicate images, and near-duplicate images across dataset splits
were identified.

These issues can lead to misleading evaluation results if they are not
addressed before model training.

Therefore, the project treats dataset preparation and validation as an
important part of the machine learning pipeline rather than simply training
a model on the original dataset.

## Main Focus

The project focuses on four main areas:

### 1. Dataset Quality

The dataset is systematically inspected for corrupted files, inconsistent
formats, unusual image dimensions, duplicate images, and near-duplicate
images.

### 2. Data Leakage Prevention

Near-duplicate images are investigated across the training, validation, and
test sets to reduce the possibility of the same underlying image appearing
in multiple splits.

### 3. Transfer Learning

Two pretrained convolutional neural network architectures are evaluated:

- ResNet50
- EfficientNet-B0

Both models use ImageNet pretrained weights and are adapted to the nine-class
classification problem.

### 4. Reliable Model Evaluation

The models are evaluated using more than overall accuracy. The evaluation
will include precision, recall, macro F1-score, weighted F1-score,
per-class performance, and confusion matrices.

This allows the project to examine how well the models perform across
different animal classes, including classes with different numbers of
training examples.

## Dataset

The original dataset contains **10,778 images across 9 classes**.

After dataset-quality investigation and removal of confirmed redundant
cross-split copies, the current clean dataset contains **10,387 images**.

| Split | Images |
|---|---:|
| Train | 5,971 |
| Validation | 1,994 |
| Test | 2,422 |
| **Total** | **10,387** |

The current training distribution is not perfectly balanced. Therefore,
training-time weighted sampling will be used to increase the sampling
frequency of classes with fewer training examples without creating duplicate
image files.