# src/dataset/medical_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import os
from typing import Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalImageDataset(Dataset):
    """Dataset for NIH Chest X-ray images"""
    
    def __init__(self, 
                 csv_file: str, 
                 img_dir: str = None,
                 transform=None, 
                 mode='train'):
        """
        Initialize dataset
        Args:
            csv_file: Path to CSV with image names and labels
            img_dir: Directory with images (optional if paths in CSV are absolute)
            transform: Albumentations transform pipeline
            mode: 'train', 'val', or 'test'
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        
        # Check if we have image_path column (from NIH dataset)
        if 'image_path' in self.data.columns:
            self.use_full_path = True
        else:
            self.use_full_path = False
        
        # Pathology columns
        self.pathology_columns = [
            'Cardiomegaly', 'Emphysema', 'Effusion', 
            'Hernia', 'Infiltration', 'Mass', 'Nodule',
            'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
            'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
        ]
        
        print(f"Dataset loaded: {len(self.data)} images for {mode}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index"""
        
        # Get image path
        if self.use_full_path:
            img_path = self.data.iloc[idx]['image_path']
        else:
            img_name = self.data.iloc[idx]['image_name']
            img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            # Create dummy image
            image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        else:
            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations or default preprocessing
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default preprocessing
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            
            # Normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Get labels
        labels = self.data.iloc[idx][self.pathology_columns].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels

def get_augmentation_pipeline(mode='train'):
    """Get augmentation pipeline for NIH dataset"""
    
    if mode == 'train':
        return A.Compose([
            # Resize
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            
            # Augmentations
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # Brightness/Contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Noise
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            ToTensorV2()
        ])
    else:  # validation and test
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])