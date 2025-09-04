# src/explainability/explainability.py
import torch
import torch.nn as nn
import torch.nn.functional as F  # ← Добавлен импорт
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize GradCAM
        Args:
            model: PyTorch model
            target_layer: Name of target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                
    def generate_heatmap(self, 
                        input_tensor: torch.Tensor, 
                        class_idx: int) -> np.ndarray:
        """
        Generate GradCAM heatmap
        Args:
            input_tensor: Input image tensor
            class_idx: Target class index
        Returns:
            Heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Calculate weighted combination
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam / torch.max(cam)
        
        # Convert to numpy and resize
        heatmap = cam.squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        
        return heatmap
    
    def visualize(self, 
                  image: np.ndarray, 
                  heatmap: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
        """
        Overlay heatmap on original image
        Args:
            image: Original image
            heatmap: GradCAM heatmap
            alpha: Overlay transparency
        Returns:
            Overlaid image
        """
        # Normalize heatmap to 0-255
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlaid = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        
        return overlaid