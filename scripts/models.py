import torch.nn as nn
from torchvision.models import (
    resnet50, 
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

Num_classes = 9

def create_resnet50(num_classes = Num_classes):
    """
    Create a ResNet50 model with ImageNet pretrained weights
    and replace the original classifier with a 9-class classifier.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights = weights)

    num_features = model.fc.in_features

    # Replace the ImageNet classifier with our 9-class classifier.
    model.fc = nn.Linear(num_features, num_classes)

    return model

def create_efficientnet_b0(num_classes = Num_classes):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights = weights)

    # Get the number of features in the classifier layer of the EfficientNet model
    num_features = model.classifier[1].in_features

    #Replace the ImageNet classifier with our 9-class classifier.
    model.classifier[1] = nn.Linear(
        num_features, num_classes
    )

    return model