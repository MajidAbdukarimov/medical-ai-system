# deployment/app.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import io
import base64
import numpy as np
from src.models.model import MedicalImageClassifier

app = Flask(__name__)

# Global variables
MODEL = None
DEVICE = torch.device('cpu')
PATHOLOGY_CLASSES = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 
    'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
    'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
    'Fibrosis', 'Edema', 'Consolidation'
]

def load_model():
    """Load trained model"""
    global MODEL
    MODEL = MedicalImageClassifier(num_classes=14, pretrained=False)
    
    # Load saved weights
    model_path = '../saved_models/best_model.pth'
    if os.path.exists(model_path):
        try:
            # Try loading with weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            MODEL.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded from {model_path}")
            print(f"  Best AUC: {checkpoint.get('best_auc', 'N/A')}")
        except Exception as e:
            print(f"Warning: Could not load saved model: {e}")
            print("Using random weights for demonstration")
    else:
        print("Warning: No saved model found, using random weights")
    
    MODEL.to(DEVICE)
    MODEL.eval()
    print("✓ Model ready for inference")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get image from request
        file = request.files['image']
        img_bytes = file.read()
        
        # Preprocess image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = MODEL(img_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Prepare results
        results = {
            'predictions': [
                {
                    'pathology': PATHOLOGY_CLASSES[i],
                    'probability': float(probs[i]),
                    'detected': bool(probs[i] > 0.5)
                }
                for i in range(len(PATHOLOGY_CLASSES))
            ]
        }
        
        # Sort by probability
        results['predictions'] = sorted(
            results['predictions'], 
            key=lambda x: x['probability'], 
            reverse=True
        )
        
        # Add summary
        detected_count = sum(1 for p in results['predictions'] if p['detected'])
        results['summary'] = {
            'total_pathologies': len(PATHOLOGY_CLASSES),
            'detected_count': detected_count,
            'top_finding': results['predictions'][0]['pathology'] if results['predictions'] else None,
            'top_probability': results['predictions'][0]['probability'] if results['predictions'] else 0
        }
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': MODEL is not None})

if __name__ == '__main__':
    print("="*50)
    print("Medical AI System - Web Application")
    print("="*50)
    
    # Load model
    load_model()
    
    # Start Flask app
    print("\n✓ Starting Flask server...")
    print("✓ Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*50)
    
    app.run(debug=False, host='0.0.0.0', port=5000)