"""
This module defines the ImageClassifier class for image classification tasks.

TODO: Implement a custom model by defining layers, activation functions, and architectures.

Helpful Resources:
- PyTorch Modules Docs: https://pytorch.org/docs/stable/nn.html
- PyTorch Functional API Docs: https://pytorch.org/docs/stable/nn.functional.html

Available Layers:
- nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding): 2D convolution layer.
- nn.BatchNorm2d(num_features): Batch normalization for convolutional layers.
- nn.Linear(in_features, out_features): Fully connected (dense) layer.
- nn.Dropout(p): Dropout layer for regularization.
- nn.MaxPool2d(kernel_size, stride): Standard max pooling layer.
- nn.AvgPool2d(kernel_size, stride): Average pooling layer.
- nn.AdaptiveAvgPool2d(output_size): Adaptive average pooling.
- nn.AdaptiveMaxPool2d(output_size): Adaptive max pooling.

Available Activation Functions:
- nn.ReLU(), nn.GELU(), nn.LeakyReLU(): Common functions.
- nn.Sigmoid(): For binary classification.
- nn.Softmax(dim): For conversion of logits to probabilities.

Example Usage:
class CustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

model = CustomModel(num_classes=4)
y = model(x)
"""


# Import necessary libraries
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    """
    Base image classification model.
    Modify or extend this class to create a custom architecture.
    """
    def __init__(self, num_classes=3):
        """
        Initialize the model layers.
        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        """ TODO: Define convolutional and batch normalization layers """
        ...
        
        """ TODO: Define pooling and dropout layers (optional) """
        ...
        
        """ TODO: Define the fully connected layer(s) """
        ...

    def forward(self, x):
        """
        Define the forward pass step-by-step.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        
        """ TODO: Implement the forward pass """
        ...
        
        return x