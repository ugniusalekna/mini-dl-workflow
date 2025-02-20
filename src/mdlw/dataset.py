"""
This module defines the ImageDataset class for handling image datasets.
It loads images, assigns labels based on a class map, and applies transformations.

TODO: Implement dataset handling by defining how images are read and labeled.

Helpful Resources:
- PyTorch Dataset Docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

Overview:
- Uses image paths to load images.
- Retrieves class labels from file paths.
- Applies transformations if provided.

Example Usage:
# Create a dataset instance
image_dataset = ImageDataset(
    image_paths=["path/to/img1.jpg", "path/to/img2.jpg"],
    class_map={"cat": 0, "dog": 1},
    transform=some_transform_pipeline # should be callable, i.e. y = some_transform_pipeline(x)
)
# Access an image and label
image, label = image_dataset[0]
"""

# Import necessary libraries
from torch.utils.data import Dataset

# Import utility functions for image handling
from .utils.data import get_cls_from_path, read_image


class ImageDataset(Dataset):
    """
    Custom dataset class for loading images and labels.
    """
    def __init__(self, image_paths, class_map, transform=None):
        """
        Initialize the dataset.
        Args:
            image_paths (list): List of image file paths.
            class_map (dict): Mapping of class names to integer labels.
            transform (callable, optional): Transformations to apply to images.
        """
        """ TODO: Store image paths, class mapping, and transformations """
        self.image_paths = ...
        self.class_map = ...
        self.transform = ...
        
    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        """ TODO: Return dataset length """
        return ...

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding label.
        Args:
            idx (int): Index of the image.
        Returns:
            tuple: (image, label)
        """
        """ TODO: Retrieve image path and class label """
        img_path = ...
        cls_name = ...
        label = ...
        
        """ TODO: Read the image from disk """
        img = ...
        
        """ Apply transformations if provided """
        if self.transform:
            img = self.transform(img)

        return img, label
