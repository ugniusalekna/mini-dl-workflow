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
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
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
    def __init__(self, input_channels=3, num_classes=10):
        """
        Initialize the model layers.
        Args:
            input_channels (int): Number of channels input image has.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        """ TODO: Store input_channels and num_classes """
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        """ TODO: Define convolutional and batch normalization layers """
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        """ TODO: Define the fully connected layer(s) """
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Define the forward pass step-by-step.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        
        """ TODO: Implement the forward pass """
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.bn2(self.conv2(x)))
        
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.bn4(self.conv4(x)))

        x = F.gelu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.bn6(self.conv6(x)))

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        x = self.fc(x)
        return x