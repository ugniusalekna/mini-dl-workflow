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
import psutil
from torch.utils.data import Dataset

# Import utility functions for image handling
from .utils.data import get_cls_from_path, read_image, to_tensor


class ImageDataset(Dataset):
    """
    Custom dataset class for loading images and labels.
    """
    def __init__(self, image_paths, class_map, transform=None, cache_images=False):
        """
        Initialize the dataset.
        Args:
            image_paths (list): List of image file paths.
            class_map (dict): Mapping of class names to integer labels.
            transform (callable, optional): Transformations to apply to images.
            cache_images (bool): Whether to preload all images into memory.
        """
        """ TODO: Store image paths, class mapping, and transformations """
        self.image_paths = image_paths
        self.class_map = class_map
        self.transform = transform
        self.cache_images = cache_images
        self.cache = None
        
        if self.cache_images:
            self.cache = self._attempt_cache_images()

    def _attempt_cache_images(self):
        """ Load all images into RAM if there is enough memory. """
        estimated_size = self._estimate_mem()
        available_mem = psutil.virtual_memory().available
        
        if estimated_size > available_mem * 0.8:
            print(f"Not enough memory to cache images ({estimated_size / 1e9:.2f} GB needed). Using disk loading.")
            return None
        else:
            print(f"Sufficient memory available ({estimated_size / 1e9:.2f} GB). Caching images...")
            return [read_image(path) for path in self.image_paths]

    def _estimate_mem(self):
        sample_img = read_image(self.image_paths[0])
        return len(self.image_paths) * sample_img.nbytes

    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        """ TODO: Return dataset length """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding label.
        Args:
            idx (int): Index of the image.
        Returns:
            tuple: (image, label)
        """
        """ TODO: Retrieve image path and class label """
        img_path = self.image_paths[idx]
        cls_name = get_cls_from_path(img_path)
        label = self.class_map[cls_name]
        
        """ TODO: Read the image from disk (or cache) """
        img = read_image(img_path) if self.cache is None else self.cache[idx]
        img = to_tensor(img)

        """ Apply transformations if provided """
        if self.transform:
            img = self.transform(img)
        
        return img, label