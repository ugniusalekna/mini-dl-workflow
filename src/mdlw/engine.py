"""
This module provides classes for training, validation, and exporting a PyTorch model.

TODO: Implement training and validation logic for handling datasets and model optimization.
Exporter is provided for model conversion to ONNX format.

Helpful Resources:
- PyTorch Training Loop Docs: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- PyTorch ONNX Export Docs: https://pytorch.org/docs/stable/onnx.html

Overview:
- Trainer: Handles model training.
- Validator: Evaluates the model on validation data.
- Exporter: Converts the trained model to ONNX format.

Example Usage:
# Initialize trainer
trainer = Trainer(model, optimizer, criterion, scheduler, device, writer)
loss, accuracy = trainer.train_epoch(train_loader, epoch)

# Validate model
validator = Validator(model, criterion, device, writer)
val_loss, val_accuracy = validator.validate(val_loader, epoch)

# Export model
exporter = Exporter(model, imgsz=224, device='cpu')
exporter.export_onnx("model.onnx")
"""

# Import necessary libraries
from tqdm import tqdm
import torch


class Trainer:
    """
    Handles training of the model using a dataset.
    """
    def __init__(self, model, optimizer, criterion, scheduler=None, device='cpu', writer=None):
        """
        Initialize Trainer.
        Args:
            model (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            criterion (torch.nn.Module): Loss function.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
            device (str): Device for computation ('cpu' or 'cuda').
            writer (torch.utils.tensorboard.SummaryWriter, optional): Logger for TensorBoard.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.writer = writer

    def train_epoch(self, dataloader, epoch):
        """
        Perform one epoch of training.
        Args:
            dataloader (torch.utils.data.DataLoader): Training dataset loader.
            epoch (int): Current epoch number.
        Returns:
            tuple: Average loss and accuracy for the epoch.
        """
        self.model.train()
        running_loss, running_corrects, total_samples = 0.0, 0, 0

        """ TODO: Implement one epoch of the training loop """
        for img_batch, label_batch in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}", leave=False):
            """ TODO: Move data to the correct device """
            img_batch = img_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            
            """ TODO: Zero gradients """
            self.optimizer.zero_grad()
            
            """ TODO: Forward pass through the model """
            logit_batch = self.model(img_batch)
            
            """ TODO: Compute the loss """
            loss = self.criterion(logit_batch, label_batch)
            
            """ TODO: Backpropagate the loss """
            loss.backward()
            
            """ TODO: Update model weights """
            self.optimizer.step()
            
            """ TODO: Update the learning rate scheduler if one is used """
            if self.scheduler:
                self.scheduler.step()

            """ TODO: Track loss and accuracy """
            running_loss += loss.item()
            pred_batch = torch.argmax(logit_batch, dim=1)
            running_corrects += (label_batch == pred_batch).sum().item()
            total_samples += img_batch.shape[0]

        """ TODO: Compute average loss and accuracy """
        avg_loss = running_loss / total_samples
        avg_acc = running_corrects / total_samples
        
        """ Log metrics if a writer is available """
        if self.writer:
            if epoch == 1:
                self.writer.add_graph(self.model, img_batch)
            self.writer.add_scalar("Train/Loss", avg_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", avg_acc, epoch)
            self.writer.add_img_grid("Train/Images", img_batch, epoch)
            self.writer.add_param_hist(self.model, epoch)
            if not epoch % 5:
                self.writer.add_embeddings(self.model, logit_batch, img_batch, epoch)
            
        return avg_loss, avg_acc


class Validator:
    """
    Handles model validation using a dataset.
    """
    def __init__(self, model, criterion, device, writer=None):
        """
        Initialize Validator.
        Args:
            model (torch.nn.Module): The neural network model.
            criterion (torch.nn.Module): Loss function.
            device (str): Device for computation ('cpu' or 'cuda').
            writer (torch.utils.tensorboard.SummaryWriter, optional): Logger for TensorBoard.
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.writer = writer

    def validate(self, dataloader, epoch):
        """
        Perform model validation.
        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataset loader.
            epoch (int): Current epoch number.
        Returns:
            tuple: Average validation loss and accuracy.
        """
        self.model.eval()
        running_loss, running_corrects, total_samples = 0.0, 0, 0

        """ TODO: Implement one epoch of the validation loop """
        for img_batch, label_batch in tqdm(dataloader, total=len(dataloader), desc="Validation", leave=False):
            """ TODO: Move data to the correct device """
            img_batch = img_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            
            """ TODO: Forward pass through the model without computing gradients! """
            with torch.no_grad():
                logit_batch = self.model(img_batch)
            
            """ TODO: Compute the loss """
            loss = self.criterion(logit_batch, label_batch)
            
            """ TODO: Track loss and accuracy """
            running_loss += loss.item()
            pred_batch = torch.argmax(logit_batch, dim=1)
            running_corrects += (pred_batch == label_batch).sum().item()
            total_samples += img_batch.shape[0]

        """ TODO: Compute average loss and accuracy """
        avg_loss = running_loss / total_samples
        avg_acc = running_corrects / total_samples
        
        """ Log metrics if a writer is available """
        if self.writer:
            self.writer.add_scalar("Val/Loss", avg_loss, epoch)
            self.writer.add_scalar("Val/Accuracy", avg_acc, epoch)

        return avg_loss, avg_acc


class Exporter:
    """
    Handles exporting the model to ONNX format.
    """
    def __init__(self, model, imgsz, device):
        self.model = model
        self.imgsz = imgsz
        self.device = device
        
    def export_onnx(self, onnx_path):
        """ Export the model to ONNX format. """
        print("Exporting model...")
        self.model.eval()
        in_channels = list(self.model.children())[0].in_channels
        
        dummy_input = torch.randn(1, in_channels, self.imgsz, self.imgsz, device=self.device)
        torch.onnx.export(
            model=self.model, 
            args=dummy_input, 
            f=onnx_path,
            input_names=["input"], 
            output_names=["output"], 
            opset_version=17
        )
        print(f"Model exported to {onnx_path}")