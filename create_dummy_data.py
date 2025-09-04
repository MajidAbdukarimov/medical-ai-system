# create_dummy_data.py
import pandas as pd
import numpy as np
import os
from PIL import Image

# Create directories
os.makedirs('data/images', exist_ok=True)

# Create dummy CSV files
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
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    img.save(f'data/images/img_{i:04d}.jpg')

print("Dummy data created successfully!")
print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")