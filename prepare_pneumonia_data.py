# prepare_pneumonia_data.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def prepare_pneumonia_dataset():
    """
    Prepare Pneumonia dataset for training
    Этот датасет намного меньше - всего ~5000 изображений!
    """
    print("="*60)
    print("Chest X-Ray Pneumonia Dataset Preparation")
    print("="*60)
    
    base_path = "data/chest_xray"
    
    if not os.path.exists(base_path):
        print("❌ Error: Dataset not found!")
        print(f"Please extract the dataset to: {base_path}")
        return
    
    # Collect all images
    data_list = []
    
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(base_path, split)
        
        if not os.path.exists(split_path):
            print(f"Warning: {split} folder not found")
            continue
            
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(split_path, class_name)
            
            if not os.path.exists(class_path):
                continue
                
            images = list(Path(class_path).glob('*.jpeg')) + \
                    list(Path(class_path).glob('*.jpg')) + \
                    list(Path(class_path).glob('*.png'))
            
            print(f"Found {len(images)} images in {split}/{class_name}")
            
            for img_path in images:
                data_list.append({
                    'image_name': img_path.name,
                    'image_path': str(img_path),
                    'split': split,
                    'label': 1 if class_name == 'PNEUMONIA' else 0,
                    'class_name': class_name
                })
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    print(f"\nTotal images found: {len(df)}")
    
    # Create train/val/test splits
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Since validation set is too small, take some from train
    if len(val_df) < 100:
        print("\n✓ Rebalancing validation set...")
        # Take 10% from train for validation
        from sklearn.model_selection import train_test_split
        train_df, extra_val = train_test_split(
            train_df, test_size=0.1, stratify=train_df['label'], random_state=42
        )
        val_df = pd.concat([val_df, extra_val])
    
    print(f"\nFinal splits:")
    print(f"Train: {len(train_df)} images")
    print(f"Val: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")
    
    # Convert to multi-label format (for compatibility)
    for df_split in [train_df, val_df, test_df]:
        # Add columns for compatibility with multi-label code
        df_split['Pneumonia'] = df_split['label']
        # Add dummy columns for other pathologies (all zeros)
        for pathology in ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia',
                         'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                         'Pneumothorax', 'Pleural_Thickening', 
                         'Fibrosis', 'Edema', 'Consolidation']:
            df_split[pathology] = 0
    
    # Save CSVs
    train_df.to_csv('data/pneumonia_train.csv', index=False)
    val_df.to_csv('data/pneumonia_val.csv', index=False)
    test_df.to_csv('data/pneumonia_test.csv', index=False)
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    
    for name, df_split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        normal = (df_split['label'] == 0).sum()
        pneumonia = (df_split['label'] == 1).sum()
        print(f"\n{name} Set:")
        print(f"  Normal: {normal} ({normal/len(df_split)*100:.1f}%)")
        print(f"  Pneumonia: {pneumonia} ({pneumonia/len(df_split)*100:.1f}%)")
    
    print("\n✓ Dataset preparation completed!")
    print("\n⚡ This dataset is MUCH smaller - training will be FAST!")
    print("   Expected time: ~2-3 minutes per epoch")

if __name__ == "__main__":
    prepare_pneumonia_dataset()