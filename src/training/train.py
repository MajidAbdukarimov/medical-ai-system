# src/training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from tqdm import tqdm
from typing import Dict
import os

class Trainer:
    """Advanced trainer with mixed precision, gradient accumulation, and monitoring"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict):
        """
        Initialize trainer
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Loss function for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Get learning rate from config (handle both 'lr' and 'learning_rate')
        lr = config.get('lr', config.get('learning_rate', 1e-4))
        
        # Optimizer with different learning rates for backbone and head
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        if backbone_params and head_params:
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': lr * 0.1},
                {'params': head_params, 'lr': lr}
            ], weight_decay=config.get('weight_decay', 1e-5))
        else:
            # If no backbone/head distinction, use single lr
            self.optimizer = optim.AdamW(
                model.parameters(), 
                lr=lr,
                weight_decay=config.get('weight_decay', 1e-5)
            )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2
        )
        
        # Mixed precision training (only if CUDA available)
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Create directory for saving models
        os.makedirs('saved_models', exist_ok=True)
        
    def save_model_safe(self, model, optimizer, epoch, best_auc, config, filepath):
        """Save model in a format compatible with newer PyTorch versions"""
        # Convert numpy types to Python types
        save_dict = {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auc': float(best_auc),
            'config': {}
        }
        
        # Convert config values to Python types
        for k, v in config.items():
            if isinstance(v, np.integer):
                save_dict['config'][k] = int(v)
            elif isinstance(v, np.floating):
                save_dict['config'][k] = float(v)
            elif isinstance(v, np.ndarray):
                save_dict['config'][k] = v.tolist()
            else:
                save_dict['config'][k] = v
        
        torch.save(save_dict, filepath)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with or without mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Gradient accumulation
            grad_acc_steps = self.config.get('gradient_accumulation_steps', 1)
            loss = loss / grad_acc_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % grad_acc_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('max_grad_norm', 1.0)
                    )
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('max_grad_norm', 1.0)
                    )
                    # Optimizer step
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * grad_acc_steps
            
            # Store predictions for metrics
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Handle edge cases for metrics
        try:
            auc = roc_auc_score(all_labels, all_preds, average='macro')
        except:
            auc = 0.0
            
        try:
            f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='macro')
        except:
            f1 = 0.0
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_auc': auc,
            'train_f1': f1
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Handle edge cases for metrics
        try:
            auc = roc_auc_score(all_labels, all_preds, average='macro')
        except:
            auc = 0.0
            
        try:
            f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='macro')
        except:
            f1 = 0.0
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_auc': auc,
            'val_f1': f1
        }
        
        return metrics
    
    def train(self, num_epochs: int):
        """Full training loop"""
        best_val_auc = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train AUC: {train_metrics['train_auc']:.4f}, "
                  f"Train F1: {train_metrics['train_f1']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val AUC: {val_metrics['val_auc']:.4f}, "
                  f"Val F1: {val_metrics['val_f1']:.4f}")
            
            # Save best model using safe save method
            if val_metrics['val_auc'] > best_val_auc:
                best_val_auc = val_metrics['val_auc']
                self.save_model_safe(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_auc=best_val_auc,
                    config=self.config,
                    filepath='saved_models/best_model.pth'
                )
                print(f"✓ Saved best model with AUC: {best_val_auc:.4f}")
            
            # Also save latest model
            self.save_model_safe(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_auc=val_metrics['val_auc'],
                config=self.config,
                filepath='saved_models/latest_model.pth'
            )
            
            print("-" * 50)