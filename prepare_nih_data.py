# prepare_nih_data.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

def prepare_nih_dataset():
    """Prepare NIH Chest X-ray dataset for training"""
    
    print("="*60)
    print("NIH Chest X-ray Dataset Preparation")
    print("="*60)
    
    # Paths
    base_path = "data/nih_chest_xrays"
    csv_path = os.path.join(base_path, "Data_Entry_2017.csv")
    test_list_path = os.path.join(base_path, "test_list.txt")
    
    # Check if data exists
    if not os.path.exists(csv_path):
        print("❌ Error: Data_Entry_2017.csv not found!")
        print(f"Please make sure you have extracted the dataset to: {base_path}")
        return
    
    print("✓ Loading metadata...")
    
    # Load main CSV
    df = pd.read_csv(csv_path)
    print(f"  Total images: {len(df)}")
    
    # Process labels
    print("\n✓ Processing labels...")
    
    # The 'Finding Labels' column contains pipe-separated disease names
    all_labels = []
    for labels in df['Finding Labels']:
        if labels != 'No Finding':
            all_labels.extend(labels.split('|'))
    
    # Get unique pathologies
    unique_pathologies = list(set(all_labels))
    unique_pathologies.sort()
    print(f"  Found {len(unique_pathologies)} unique pathologies:")
    for i, pathology in enumerate(unique_pathologies, 1):
        print(f"    {i}. {pathology}")
    
    # Our target pathologies (14 main ones)
    target_pathologies = [
        'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia',
        'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
        'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
        'Fibrosis', 'Edema', 'Consolidation'
    ]
    
    print(f"\n✓ Creating binary labels for {len(target_pathologies)} pathologies...")
    
    # Create binary columns for each pathology
    for pathology in target_pathologies:
        df[pathology] = df['Finding Labels'].apply(
            lambda x: 1 if pathology in x else 0
        )
    
    # Add image path
    df['image_path'] = df['Image Index'].apply(
        lambda x: find_image_path(base_path, x)
    )
    
    # Remove images that don't exist
    print("\n✓ Checking image files...")
    existing_images = df['image_path'].apply(os.path.exists)
    df = df[existing_images]
    print(f"  Found {len(df)} images with valid paths")
    
    # Load test list if available
    test_images = set()
    if os.path.exists(test_list_path):
        with open(test_list_path, 'r') as f:
            test_images = set(line.strip() for line in f)
        print(f"  Official test set: {len(test_images)} images")
    
    # Split data
    print("\n✓ Splitting dataset...")
    
    if test_images:
        # Use official test split
        test_df = df[df['Image Index'].isin(test_images)]
        trainval_df = df[~df['Image Index'].isin(test_images)]
        train_df, val_df = train_test_split(
            trainval_df, test_size=0.15, random_state=42
        )
    else:
        # Create our own splits
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    # Select subset for faster training (optional)
    if len(train_df) > 10000:
        print("\n✓ Creating subset for faster training...")
        train_subset = train_df.sample(n=10000, random_state=42)
        val_subset = val_df.sample(n=min(2000, len(val_df)), random_state=42)
        test_subset = test_df.sample(n=min(2000, len(test_df)), random_state=42)
        
        # Save subset CSVs
        save_dataset(train_subset, val_subset, test_subset, 
                    target_pathologies, prefix='subset_')
        print("  Subset saved with prefix 'subset_'")
    
    # Save full CSVs
    save_dataset(train_df, val_df, test_df, target_pathologies)
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{split_name} Set:")
        total_positive = 0
        for pathology in target_pathologies:
            count = split_df[pathology].sum()
            total_positive += count
            percentage = (count / len(split_df)) * 100
            print(f"  {pathology:20s}: {count:6d} ({percentage:5.1f}%)")
        
        no_finding = len(split_df) - len(split_df[split_df[target_pathologies].any(axis=1)])
        print(f"  {'No Finding':20s}: {no_finding:6d} ({no_finding/len(split_df)*100:5.1f}%)")
    
    print("\n✓ Dataset preparation completed!")
    print("\nNext steps:")
    print("1. Update config/config.yaml with new data paths")
    print("2. Run: python main.py --config config/config.yaml")
    
def find_image_path(base_path, image_name):
    """Find the actual path of an image"""
    # Images might be in different folders (images_001, images_002, etc.)
    possible_paths = [
        os.path.join(base_path, 'images', image_name),
        os.path.join(base_path, image_name),
    ]
    
    # Check numbered folders
    for i in range(1, 13):
        folder_name = f'images_{i:03d}'
        possible_paths.append(
            os.path.join(base_path, folder_name, 'images', image_name)
        )
        possible_paths.append(
            os.path.join(base_path, folder_name, image_name)
        )
    
    # Return first existing path
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Default path
    return os.path.join(base_path, 'images', image_name)

def save_dataset(train_df, val_df, test_df, target_pathologies, prefix=''):
    """Save dataset CSVs"""
    
    # Columns to save
    columns_to_save = ['Image Index', 'image_path'] + target_pathologies
    
    # Save CSVs
    train_df[columns_to_save].to_csv(f'data/{prefix}train_nih.csv', index=False)
    val_df[columns_to_save].to_csv(f'data/{prefix}val_nih.csv', index=False)
    test_df[columns_to_save].to_csv(f'data/{prefix}test_nih.csv', index=False)
    
    # Also save with simplified column name for compatibility
    train_df_compat = train_df[columns_to_save].copy()
    val_df_compat = val_df[columns_to_save].copy()
    test_df_compat = test_df[columns_to_save].copy()
    
    train_df_compat.rename(columns={'Image Index': 'image_name'}, inplace=True)
    val_df_compat.rename(columns={'Image Index': 'image_name'}, inplace=True)
    test_df_compat.rename(columns={'Image Index': 'image_name'}, inplace=True)
    
    train_df_compat.to_csv(f'data/{prefix}train.csv', index=False)
    val_df_compat.to_csv(f'data/{prefix}val.csv', index=False)
    test_df_compat.to_csv(f'data/{prefix}test.csv', index=False)

if __name__ == "__main__":
    prepare_nih_dataset()