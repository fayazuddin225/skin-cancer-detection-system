import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Configuration
DATASET_PATH = "." 
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
IMG_HEIGHT, IMG_WIDTH = 299, 299 # InceptionV3 standard size
MAX_IMAGES = 500 # Limit for speed

# Load InceptionV3 model (pre-trained on ImageNet, no top layer)
print("Loading InceptionV3 model...")
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
print("InceptionV3 loaded.")

def extract_inception_features(img):
    # Resize image to InceptionV3 standard
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img = img.convert('RGB')
    
    # Convert to array and preprocess
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Extract features
    features = inception_model.predict(x, verbose=0)
    return features.flatten()

def load_data_hybrid(directory, limit=None):
    print(f"Loading data for Hybrid Model from {directory}...")
    features = []
    labels = []
    classes = ['benign', 'malignant']
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            continue
        
        count = 0
        files = os.listdir(class_dir)
        files.sort()
        
        for img_name in files:
            if limit and count >= limit:
                break
                
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    feats = extract_inception_features(img)
                    features.append(feats)
                    labels.append(label)
                    count += 1
                    if count % 50 == 0:
                        print(f"Processed {count} images for {class_name}")
            except Exception:
                pass
    
    return np.array(features), np.array(labels)

def train_hybrid():
    print("Loading Training Data...")
    X_train, y_train = load_data_hybrid(TRAIN_DIR, limit=MAX_IMAGES)
    print(f"Loaded {len(X_train)} training samples.")
    
    print("Loading Validation Data...")
    X_val, y_val = load_data_hybrid(VAL_DIR, limit=100)
    
    if len(X_train) == 0:
        print("No training data found!")
        return

    # --- Train SVM (Hybrid Classifier) ---
    print("\nStarting Hybrid SVM Training...")
    svm = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
    svm.fit(X_train, y_train)
    
    if len(X_val) > 0:
        val_predictions = svm.predict(X_val)
        acc = accuracy_score(y_val, val_predictions)
        print(f"Hybrid Validation Accuracy: {acc:.4f}")
    
    joblib.dump(svm, 'hybrid_model.pkl')
    print("Hybrid Model saved as hybrid_model.pkl")

if __name__ == "__main__":
    try:
        train_hybrid()
    except Exception as e:
        print(f"Training failed: {e}")
