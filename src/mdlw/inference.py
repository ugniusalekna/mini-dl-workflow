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

import cv2 as cv
import numpy as np
import onnxruntime as ort
from .utils.data import reverse_class_map


class InferenceModel:
    """
    Handles inference using an ONNX model.
    """
    def __init__(self, model_path, class_map):
        """
        Initialize the inference model.
        Args:
            model_path (str): Path to the ONNX model file.
            class_map (dict): Mapping of class names to indices.
        """
        self.session = ort.InferenceSession(model_path)
        self.reversed_map = reverse_class_map(class_map)
        self.fetch_input_data()
        
    def fetch_input_data(self):
        """
        Retrieve input and output metadata from the ONNX model.
        """
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.image_size = self.input_shape[-2:]
        
    def preprocess(self, frame):
        """
        Preprocess an image before feeding it into the model.
        Args:
            frame (numpy.ndarray): Input image in BGR format.
        Returns:
            numpy.ndarray: Preprocessed image ready for inference.
        """
        frame = cv.resize(frame, self.image_size)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def __call__(self, frame, return_prob=False):
        """
        Perform inference on an input image.
        Args:
            frame (numpy.ndarray): Input image.
            return_prob (bool): Whether to return confidence score.
        Returns:
            str: Predicted class label.
            float (optional): Confidence score of the prediction.
        """
        frame = self.preprocess(frame)
        (logits,) = self.session.run([self.output_name], {self.input_name: frame})[0]
        class_idx = np.argmax(logits)
        
        if return_prob:
            """ Compute softmax probabilities """
            probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
            return self.reversed_map[class_idx], probs[class_idx]
        
        return self.reversed_map[class_idx]
