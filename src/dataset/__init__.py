# src/dataset/__init__.py  
from .medical_dataset import MedicalImageDataset, get_augmentation_pipeline

__all__ = ['MedicalImageDataset', 'get_augmentation_pipeline']