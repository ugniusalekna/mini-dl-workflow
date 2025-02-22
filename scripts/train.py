# Import necessary libraries from existing frameworks
# Import our own implementations: augmentations, dataset, model, and training utilities


# 0. Setup and configuration
""" TODO: Define hyperparameters and settings """

from mdlw.utils.misc import load_cfg, initialize_run_dir, save_cfg
cfg = load_cfg(path='./config/config.yaml')
run_dir = initialize_run_dir(log_dir=cfg.log_dir)
save_cfg(cfg, path=f'{run_dir}/args.yaml')


# 1. Reading, loading and splitting data
""" TODO: Load image paths """

from mdlw.utils.data import get_image_paths, make_class_map
img_paths, class_map = get_image_paths(cfg.data_dir), make_class_map(cfg.data_dir)

""" TODO: Split data into training and validation sets """

from mdlw.utils.data import train_val_split
train_img_paths, val_img_paths = train_val_split(img_paths, val_ratio=cfg.val_ratio, seed=cfg.seed)


# 2. Defining datasets and data loaders
""" TODO: Initialize instances of dataset class for training and validation """

from mdlw.augment import Augmenter
train_transform = Augmenter(train=True, image_size=cfg.image_size)
val_transform = Augmenter(train=False, image_size=cfg.image_size)

from mdlw.dataset import ImageDataset
train_dataset = ImageDataset(
    image_paths=train_img_paths, 
    class_map=class_map,
    transform=train_transform,
)
val_dataset = ImageDataset(
    image_paths=val_img_paths, 
    class_map=class_map,
    transform=val_transform,
)

""" TODO: Initialize instances of DataLoader from torch.utils.data for training and validation datasets """

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)


# 3. Model initialization; other necessary components for training
from mdlw.utils.misc import get_device
device = get_device()  # selects best available device

""" TODO: Instantiate the model """

from mdlw.model import ImageClassifier
model = ImageClassifier(input_channels=cfg.input_channels, num_classes=cfg.num_classes).to(device)

""" TODO: Define optimizer and scheduler (optional) from torch.optim """

import torch
optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.learning_rate, steps_per_epoch=len(train_loader), epochs=cfg.num_epochs)

""" TODO: Define loss function from torch.nn """

loss_fn = torch.nn.CrossEntropyLoss()


# 4. Set up logging and training utilities
from mdlw.utils.writer import Writer
writer = Writer(log_dir=f"{run_dir}/logs")

""" TODO: Create instances of Trainer, Validator classes """

from mdlw.engine import Trainer, Validator
trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, criterion=loss_fn, device=device, writer=writer)
validator = Validator(model=model, criterion=loss_fn, device=device, writer=writer)


# 5. Training loop
""" TODO: Iterate through epochs, training and validating the model """

best_val_acc = 0.0

from tqdm import tqdm
pbar = tqdm(range(1, cfg.num_epochs + 1), leave=False)
for epoch in pbar:
    train_loss, train_acc = trainer.train_epoch(train_loader, epoch=epoch)
    val_loss, val_acc = validator.validate(val_loader, epoch=epoch)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model, f"{run_dir}/best_model.pt")
        
    pbar.set_postfix_str(f"Loss: {train_loss:.2f}, Acc: {train_acc:.2f}; Val Loss: {val_loss:.2f}, Val Acc: {val_acc:.2f}")

writer.close()
print(f"Training completed; Best validation accuracy: {best_val_acc:.2%}")
print(f"Model saved at: {run_dir}/best_model.pt")


# 6. Exporting trained model to ONNX format
""" TODO: Setup ONNX export """

try:
    import onnx
except ImportError:
    onnx = None
    print("ONNX is not installed. Skipping export.")

if onnx:
    from mdlw.engine import Exporter
    exporter = Exporter(model, imgsz=cfg.image_size, device=device)
    exporter.export_onnx(f"{run_dir}/best_model.onnx")