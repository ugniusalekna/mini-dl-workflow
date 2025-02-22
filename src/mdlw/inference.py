"""
This module provides the InferenceModel class for performing inference using an ONNX model.
It loads a pre-trained model, processes input frames, and returns predictions.

Helpful Resources:
- ONNX Runtime Docs: https://onnxruntime.ai/docs/
- PyTorch Export Docs: https://pytorch.org/docs/stable/export.html

Overview:
- Loads an ONNX model for inference.
- Preprocesses input images to match model requirements.
- Runs inference and returns predicted class labels.

Example Usage:
inference_model = InferenceModel("model.onnx", class_map)
prediction = inference_model(frame)
"""
from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv
import torch
try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .utils.data import reverse_class_map


class BaseInferenceModel(ABC):
    """
    Base class for inference model.
    """
    def __init__(self, model_path, class_map, image_size=64):
        self.model_path = model_path
        self.class_map = class_map
        self.reversed_map = reverse_class_map(class_map)
        self.image_size = image_size
        self.is_grayscale = None

        self.load_model()

    @abstractmethod
    def load_model(self):
        """Subclasses must implement how to load the model."""
        pass

    @abstractmethod
    def run_inference(self, input_data):
        """Subclasses must implement how to run inference."""
        pass

    def preprocess(self, frame):
        """
        Preprocess an image before inference.
        Automatically converts to grayscale if required.
        """
        frame = cv.resize(frame, (self.image_size, self.image_size))
        if self.is_grayscale:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, axis=-1)
        else:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        return frame

    def __call__(self, frame, return_prob=False):
        """
        Runs inference on an input image.
        """
        frame = self.preprocess(frame)
        logits = self.run_inference(frame)

        class_idx = np.argmax(logits)

        if return_prob:
            """ Compute softmax probabilities """
            probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
            return self.reversed_map[class_idx], probs[class_idx]

        return self.reversed_map[class_idx]
    

class TorchInferenceModel(BaseInferenceModel):
    """
    PyTorch-based inference model.
    """
    def __init__(self, model_path, class_map, image_size=(64, 64), device='cpu'):
        self.device = device
        super().__init__(model_path, class_map, image_size)
        
    def load_model(self):
        """Loads the PyTorch model."""
        self.net = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.net.eval()

        first_layer = list(self.net.children())[0]
        if isinstance(first_layer, torch.nn.Conv2d):
            self.is_grayscale = first_layer.in_channels == 1
        else:
            raise ValueError("Cannot determine input format from model.")

    def run_inference(self, input_data):
        """Runs inference on a PyTorch model."""
        input_tensor = torch.tensor(input_data, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.net(input_tensor).cpu().numpy()[0]
            
        return logits


class ONNXInferenceModel(BaseInferenceModel):
    """
    ONNX-based inference model.
    """
    def load_model(self):
        """Loads the ONNX model."""
        if not ort:
            raise ImportError("ONNXRuntime is not installed. Cannot load ONNX model.")

        self.session = ort.InferenceSession(self.model_path)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape

        self.is_grayscale = input_shape[1] == 1

    def run_inference(self, input_data):
        """Runs inference on an ONNX model."""
        input_data = np.expand_dims(input_data, axis=0)
        
        (logits,) = self.session.run([self.output_name], {self.input_name: input_data})[0]
        
        return logits