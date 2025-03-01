"""
This module defines the Augmenter class for applying data augmentations.
Modify the transformations based on the specific task requirements.
NOTE: Some tasks may not benefit from certain augmentations rather than others.

TODO: Implement the augmentation pipeline by selecting appropriate transformations.

Helpful Resources:
- Torchvision Transforms Docs: https://pytorch.org/vision/stable/transforms.html
- Torchvision v2 Transforms Docs: https://pytorch.org/vision/stable/transforms_v2.html

Some Transformations:
- T.Resize(size): Resizes the image to the given size.
- T.RandomResizedCrop(size, scale): Randomly crops and resizes the image.,
- T.RandomHorizontalFlip(p): Flips the image horizontally with probability p.
- T.RandomRotation(degrees): Rotates the image by a random degree.
- T.ColorJitter(brightness, contrast, saturation, hue): Randomly changes image color properties.
- T.GaussianBlur(kernel_size, sigma): Applies a Gaussian blur.
- T.RandomApply([T2.GaussianNoise(sigma)], p): Randomly applies Gaussian noise.

Example Usage:
example_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
])

"""

# Import necessary libraries for transformations
import torchvision.transforms.v2 as T


class Augmenter:
    """
    Augmenter class applies different transformations to images.
    Modify the augmentations as needed for specific tasks.
    """
    def __init__(self, train=True, image_size=224):
        """
        Initialize transformations.
        Args:
            train (bool): Whether to apply training augmentations.
            image_size (int): Target image size.
        """
        if train:
            """ TODO: Define the training augmentations """
            self.transforms = T.Compose([
                # Replace with appropriate transformations
                T.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                T.RandomRotation(degrees=10),
                T.RandomHorizontalFlip(p=0.5),                      # <------ COMMENT OUT for EMNIST
                T.RandomPerspective(distortion_scale=0.2, p=0.5),
                T.GaussianNoise(sigma=0.01),
            ])
        else:
            """ TODO: Define the validation augmentations """
            self.transforms = T.Compose([
                # Replace with appropriate transformations
                T.Resize(image_size)
            ])

    def __call__(self, img):
        """Apply the defined transformations to an image."""
        return self.transforms(img)