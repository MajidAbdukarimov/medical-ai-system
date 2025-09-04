# main.py
import argparse
import torch
import random
import numpy as np
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Now import with correct paths
from config.config import load_config
from src.dataset.medical_dataset import MedicalImageDataset, get_augmentation_pipeline
from src.models.model import MedicalImageClassifier, EnsembleModel
from src.training.train import Trainer
from torch.utils.data import DataLoader

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    """Main training pipeline"""
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config.seed)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Create datasets
    train_dataset = MedicalImageDataset(
        csv_file=config.data.train_csv,
        img_dir=config.data.img_dir,
        transform=get_augmentation_pipeline('train'),
        mode='train'
    )
    
    val_dataset = MedicalImageDataset(
        csv_file=config.data.val_csv,
        img_dir=config.data.img_dir,
        transform=get_augmentation_pipeline('val'),
        mode='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # Create model
    if args.ensemble:
        model = EnsembleModel(num_classes=config.model.num_classes)
        logger.info("Using ensemble model")
    else:
        model = MedicalImageClassifier(
            num_classes=config.model.num_classes,
            backbone=config.model.backbone,
            pretrained=config.model.pretrained
        )
        logger.info(f"Using single model with backbone: {config.model.backbone}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=vars(config.training)
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train(config.training.num_epochs)
    
    logger.info("Training completed!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical AI Training")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble model')
    args = parser.parse_args()
    
    main(args)