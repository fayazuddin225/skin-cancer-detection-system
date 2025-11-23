import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


app = Flask(__name__)

# Global variables for models
svm_model = None
mlp_model = None
outlier_model = None
hybrid_model = None
inception_model = None

def load_models():
    global outlier_model, hybrid_model, inception_model
    
    # Get absolute path of the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {base_dir}")
    
    try:
        outlier_path = os.path.join(base_dir, 'outlier_model.pkl')
        hybrid_path = os.path.join(base_dir, 'hybrid_model.pkl')

        if os.path.exists(outlier_path):
            print(f"Loading outlier_model from {outlier_path}...")
            outlier_model = joblib.load(outlier_path)
        else:
            print(f"outlier_model.pkl NOT FOUND at {outlier_path}")

        if os.path.exists(hybrid_path):
            print(f"Loading hybrid_model from {hybrid_path}...")
            hybrid_model = joblib.load(hybrid_path)
        else:
            print(f"hybrid_model.pkl NOT FOUND at {hybrid_path}")
        
        # Load InceptionV3
        print("Loading InceptionV3 model...")
        inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        print("InceptionV3 loaded.")
        
        if hybrid_model and inception_model:
            print("Models loaded successfully.")
        else:
            print("Models not found. Please train them first.")
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()

# Load models on startup
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Use global variables
    global outlier_model, hybrid_model, inception_model

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        print("Received prediction request")
        # Preprocess image for Outlier Detection (64x64)
        img = Image.open(file.stream)
        img_outlier = img.resize((64, 64))
        img_outlier = img_outlier.convert('RGB')
        img_array_outlier = np.array(img_outlier).flatten().reshape(1, -1)
        img_array_outlier = img_array_outlier / 255.0 # Normalize
        print("Image preprocessed for outlier detection")
        
        # Outlier Detection
        if outlier_model is not None:
            # Check if image is an outlier
            is_inlier = outlier_model.predict(img_array_outlier)[0]
            if is_inlier == -1:
                print("Outlier detected (Non-skin image)")
                return jsonify({'error': 'Invalid image detected. Please upload a skin lesion image.'}), 400
            print("Image is valid") 
        else:
            print("Warning: Outlier model not loaded, skipping check.")

        results = {}
        
        # Hybrid Model Prediction (InceptionV3 + SVM)
        if hybrid_model is not None and inception_model is not None:
            print("Running Hybrid Model...")
            # Preprocess for InceptionV3 (299x299)
            img_inception = img.resize((299, 299))
            img_inception = img_inception.convert('RGB')
            x = img_to_array(img_inception)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Extract features
            features = inception_model.predict(x, verbose=0)
            features = features.flatten().reshape(1, -1)
            
            pred = hybrid_model.predict(features)[0]
            prob = hybrid_model.predict_proba(features)[0]
            print(f"Prediction: {pred}, Probability: {prob}")
            
            results['hybrid'] = {
                'prediction': 'Malignant' if pred == 1 else 'Benign',
                'confidence': float(max(prob))
            }
            
        if not results:
            print("No models loaded!")
            return jsonify({'error': 'Models not loaded'}), 500
            
        print("Returning results:", results)
        return jsonify(results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Reload trigger
