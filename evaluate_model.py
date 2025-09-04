# evaluate_model.py
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.model import MedicalImageClassifier
from src.dataset.medical_dataset import MedicalImageDataset
from torch.utils.data import DataLoader

def evaluate_model():
    """Evaluate trained model and create visualizations"""
    
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalImageClassifier(num_classes=14, pretrained=False)
    
    try:
        checkpoint = torch.load('saved_models/best_model.pth', 
                              map_location=device, 
                              weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded (Best AUC: {checkpoint.get('best_auc', 'N/A')})")
    except:
        print("⚠️ Using latest model instead")
        checkpoint = torch.load('saved_models/latest_model.pth', 
                              map_location=device, 
                              weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = MedicalImageDataset(
        csv_file='data/subset_test.csv',  # или test.csv
        img_dir='data/images',
        mode='test'
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Collect predictions
    all_preds = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics for main pathologies
    pathologies = test_dataset.pathology_columns
    
    print("\n" + "="*60)
    print("Performance by Pathology:")
    print("="*60)
    
    results = {}
    for i, pathology in enumerate(pathologies[:5]):  # Top 5 pathologies
        y_true = all_labels[:, i]
        y_pred = (all_preds[:, i] > 0.5).astype(int)
        
        # Skip if no positive samples
        if y_true.sum() == 0:
            print(f"{pathology:20s}: No positive samples in test set")
            continue
            
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[pathology] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"{pathology:20s}: Acc={accuracy:.2f}, P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    
    # Create visualization
    create_evaluation_plots(results)
    
    return results

def create_evaluation_plots(results):
    """Create visualization plots"""
    
    if not results:
        print("No results to visualize")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Metrics comparison
    pathologies = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    x = np.arange(len(pathologies))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[p][metric] for p in pathologies]
        axes[0].bar(x + i*width, values, width, label=metric.capitalize())
    
    axes[0].set_xlabel('Pathology')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance by Pathology')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(pathologies, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average metrics
    avg_metrics = {}
    for metric in metrics:
        avg_metrics[metric] = np.mean([results[p][metric] for p in pathologies])
    
    axes[1].bar(metrics, avg_metrics.values(), color=['blue', 'green', 'orange', 'red'])
    axes[1].set_ylabel('Score')
    axes[1].set_title('Average Model Performance')
    axes[1].set_ylim([0, 1])
    
    for i, (metric, value) in enumerate(avg_metrics.items()):
        axes[1].text(i, value + 0.02, f'{value:.2f}', ha='center')
    
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Evaluation plots saved to model_evaluation.png")
    
if __name__ == "__main__":
    results = evaluate_model()