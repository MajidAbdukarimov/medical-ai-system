# setup_project.py
import os
import pandas as pd
import numpy as np
from PIL import Image

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/images',
        'config',
        'src/dataset',
        'src/models',
        'src/training',
        'src/explainability',
        'src/utils',
        'saved_models',
        'notebooks',
        'tests',
        'deployment/templates',
        'deployment/static'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        
        # Create __init__.py files
        if dir_path.startswith('src/') or dir_path == 'src':
            init_file = os.path.join(dir_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Auto-generated init file\n')
    
    print("✓ Directories created")

def create_dummy_data():
    """Create dummy dataset for testing"""
    # Create directories
    os.makedirs('data/images', exist_ok=True)
    
    # Pathology columns
    pathology_columns = [
        'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 
        'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
        'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
        'Fibrosis', 'Edema', 'Consolidation'
    ]
    
    # Generate dummy data
    n_samples = 100
    data = {
        'image_name': [f'img_{i:04d}.jpg' for i in range(n_samples)]
    }
    
    # Add random labels for each pathology
    for col in pathology_columns:
        data[col] = np.random.randint(0, 2, n_samples)
    
    # Create train/val/test splits
    df = pd.DataFrame(data)
    train_df = df[:70]
    val_df = df[70:85]
    test_df = df[85:]
    
    # Save CSV files
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    # Create dummy images
    for i in range(n_samples):
        # Create random grayscale image
        img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        # Convert to RGB
        img = Image.fromarray(np.stack([img_array]*3, axis=-1))
        img.save(f'data/images/img_{i:04d}.jpg')
    
    print("✓ Dummy data created")
    print(f"  - Train samples: {len(train_df)}")
    print(f"  - Val samples: {len(val_df)}")
    print(f"  - Test samples: {len(test_df)}")

def create_config():
    """Create config.yaml file"""
    os.makedirs('config', exist_ok=True)
    
    config_content = """experiment_name: "medical_ai_baseline"
seed: 42

data:
  train_csv: "data/train.csv"
  val_csv: "data/val.csv"
  test_csv: "data/test.csv"
  img_dir: "data/images"
  batch_size: 8
  num_workers: 0

model:
  backbone: "resnet50"
  num_classes: 14
  pretrained: false
  use_attention: false
  dropout_rate: 0.5

training:
  num_epochs: 5
  learning_rate: 0.0001
  weight_decay: 0.00001
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  early_stopping_patience: 10
"""
    
    with open('config/config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✓ Config file created")

def main():
    print("Setting up Medical AI System project...")
    print("-" * 40)
    
    create_directories()
    create_dummy_data()
    create_config()
    
    print("-" * 40)
    print("✓ Setup completed successfully!")
    print("\nYou can now run:")
    print("  python main.py --config config/config.yaml")

if __name__ == "__main__":
    main()