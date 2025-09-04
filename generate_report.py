# generate_report.py
import os
from datetime import datetime
from textwrap import dedent

def generate_project_report(out_dir: str = "reports") -> str:
    """Generate final project report as Markdown and save to disk"""
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(out_dir, f"final_project_report_{date_str}.md")

    report = dedent(f"""
    # Medical AI System - Final Project Report
    **Date**: {date_str}  
    **Team**: [Your Names Here]

    ---

    ## 📋 Executive Summary
    Successfully developed and deployed an end-to-end deep learning system for chest X-ray analysis, capable of detecting 14 different pathological conditions using multi-label classification.

    ## 🎯 Project Objectives
    1. ✅ Build a deep learning model for medical image classification  
    2. ✅ Implement data preprocessing and augmentation pipeline  
    3. ✅ Train model on real medical dataset  
    4. ✅ Deploy model as web application  
    5. ✅ Create user-friendly interface for real-time predictions

    ## 📊 Dataset
    - **Source**: NIH Chest X-ray Dataset (subset)
    - **Size**: 10,000 images for training
    - **Classes**: 14 pathological conditions
    - **Split**: 70% train, 15% validation, 15% test

    ## 🏗️ Architecture

    ### Model Architecture
    - **Backbone**: ResNet50 (pretrained on ImageNet)
    - **Transfer Learning**: Fine-tuned on medical images
    - **Output**: Multi-label classification (14 classes)
    - **Loss Function**: Binary Cross-Entropy with Logits

    ### Technical Stack
    - Python 3.10+, PyTorch, torchvision, Albumentations
    - FastAPI/Streamlit for serving UI + inference endpoint
    - ONNX export (optional) for portability; Docker for deployment
    - Logging: Python logging + TensorBoard/W&B (optional)

    ## ⚙️ Data Preprocessing & Augmentation
    - DICOM/PNG loading, resizing, normalization (ImageNet stats)
    - Augmentations: random rotate/flip, CLAHE (optional), brightness/contrast
    - Class imbalance handling: weighted loss / sampling

    ## 🧪 Training Setup
    - Optimizer: AdamW (lr=1e-4), Scheduler: OneCycleLR
    - Batch size: 32, Epochs: 20–30
    - Early stopping on val AUROC / F1

    ## ✅ Results (Example)
    - Mean AUROC (14 labels): 0.86
    - Macro F1: 0.61
    - Inference latency (GPU T4): ~12 ms / image

    ## 🚀 Deployment
    - Containerized inference service (FastAPI) behind Nginx
    - Model weights: `models/resnet50_chestxray_best.pth` (or `model.onnx`)
    - Healthcheck endpoint `/health`, prediction endpoint `/predict`

    ## ♻️ Reproducibility
        # Train
        python src/train.py --config configs/resnet50.yaml
        # Evaluate
        python src/eval.py --ckpt models/resnet50_chestxray_best.pth
        # Run API
        uvicorn app.main:app --host 0.0.0.0 --port 8000

    ## ⚠️ Limitations & Future Work
    - Label noise in NIH dataset; limited external validation
    - Next: calibration, test-time augmentation, ensembling, OOD detection

    ## 📎 References
    - Wang et al., "ChestX-ray8/14"
    - He et al., "Deep Residual Learning for Image Recognition"
    """).strip() + "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    return out_path


if __name__ == "__main__":
    path = generate_project_report()
    print(f"[OK] Report saved to: {path}")
