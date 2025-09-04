# tests/test_model.py
import unittest
import torch
import numpy as np
from src.model import MedicalImageClassifier, EnsembleModel
from src.dataset import MedicalImageDataset

class TestModel(unittest.TestCase):
    """Unit tests for model components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = MedicalImageClassifier(num_classes=14)
        self.batch_size = 4
        self.img_size = 224
        
    def test_model_forward(self):
        """Test forward pass"""
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        output = self.model(x)
        
        self.assertEqual(output.shape, (self.batch_size, 14))
        
    def test_model_gradients(self):
        """Test if gradients flow correctly"""
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        output = self.model(x)
        loss = output.mean()
        loss.backward()
        
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                
    def test_ensemble_model(self):
        """Test ensemble model"""
        ensemble = EnsembleModel(num_classes=14)
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        output = ensemble(x)
        
        self.assertEqual(output.shape, (self.batch_size, 14))

if __name__ == '__main__':
    unittest.main()