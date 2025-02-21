# Import necessary libraries from existing frameworks
# Import our own implementations: augmentations, dataset, model, and training utilities


# 0. Setup and configuration
"""
TODO: Define hyperparameters and settings
"""


# 1. Reading, loading and splitting data
"""
TODO: Load image paths 
"""
img_paths, class_map = ..., ...

""" 
TODO: Split data into training and validation sets
"""
train_img_paths, val_img_paths = ..., ...


# 2. Defining datasets and data loaders
""" 
TODO: Initialize instances of dataset class for training and validation 
"""
train_dataset = ...
val_dataset = ...

""" 
TODO: Initialize instances of DataLoader from torch.utils.data for training and validation datasets 
"""
train_loader = ...
val_loader = ...


# 3. Model initialization; other necessary components for training
from mdlw.utils.misc import get_device
device = get_device()  # selects best available device

""" 
TODO: Instantiate the model
"""
model = ...

""" 
TODO: Define optimizer from torch.optim 
"""
optimizer = ...

""" 
TODO: Define loss function from torch.nn """
loss_fn = ...


# 4. Set up logging and training utilities
# from mdlw.utils.writer import Writer
# writer = Writer(log_dir='')

""" TODO: Create instances of Trainer, Validator classes """
trainer = ...
validator = ...


# 5. Training loop
"""
TODO: Iterate through epochs, training and validating the model
"""


# 6. Exporting trained model to ONNX format
""" TODO: Setup ONNX export """
# from mdlw.engine import Exporter
# exporter = Exporter(model, imgsz=None, device=device)
# exporter.export_onnx('')