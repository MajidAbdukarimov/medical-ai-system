# download_real_data.py
import os
import requests
from tqdm import tqdm

def download_nih_chest_xray_sample():
    """
    Download sample of NIH Chest X-ray dataset
    Note: For full dataset, visit: https://www.kaggle.com/nih-chest-xrays/data
    """
    print("For real medical data, please:")
    print("1. Visit Kaggle: https://www.kaggle.com/nih-chest-xrays/data")
    print("2. Download the dataset")
    print("3. Extract to data/nih_chest_xrays/")
    print("\nAlternatively, you can use these smaller datasets:")
    print("- COVID-19 Radiography: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database")
    print("- Pneumonia Detection: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
    
    # Create info file
    info_content = """
# Recommended Datasets for Medical AI Project

## 1. NIH Chest X-ray Dataset
- Link: https://www.kaggle.com/nih-chest-xrays/data
- Size: ~42GB
- Images: 112,120 X-ray images
- Labels: 14 disease labels

## 2. COVID-19 Radiography Database
- Link: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
- Size: ~2GB
- Good for COVID/Normal/Pneumonia classification

## 3. Chest X-Ray Images (Pneumonia)
- Link: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- Size: ~1.2GB
- Binary classification: Pneumonia vs Normal

## How to use:
1. Download dataset from Kaggle
2. Extract to data/ folder
3. Update config.yaml with correct paths
4. Run preprocessing script
"""
    
    with open('data/DATASETS_INFO.md', 'w') as f:
        f.write(info_content)
    
    print("\n✓ Dataset information saved to data/DATASETS_INFO.md")

if __name__ == "__main__":
    download_nih_chest_xray_sample()