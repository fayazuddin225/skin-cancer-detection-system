import os
import numpy as np
import joblib
from PIL import Image
from sklearn.ensemble import IsolationForest

# Configuration
DATASET_PATH = "." 
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
IMG_HEIGHT, IMG_WIDTH = 64, 64 # Match app.py preprocessing
MAX_IMAGES = 1000 # Sufficient for outlier detection

def load_data_for_outlier(directory, limit=None):
    print(f"Loading data for Outlier Detection from {directory}...")
    images = []
    classes = ['benign', 'malignant']
    
    for class_name in classes:
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
                    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
                    img = img.convert('RGB')
                    img_array = np.array(img).flatten()
                    img_array = img_array / 255.0
                    images.append(img_array)
                    count += 1
            except Exception:
                pass
    
    return np.array(images)

def train_outlier():
    print("Loading Training Data for Outlier Detection...")
    X_train = load_data_for_outlier(TRAIN_DIR, limit=MAX_IMAGES)
    print(f"Loaded {len(X_train)} images.")
    
    if len(X_train) == 0:
        print("No training data found!")
        return

    # --- Train Isolation Forest ---
    print("Training Isolation Forest...")
    # contamination=0.01: Very low contamination assumes most training data is valid.
    # This makes the model less strict, so it won't reject valid skin images easily.
    # It will still detect significantly different images (like horses/cats) as outliers.
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_train)
    
    joblib.dump(iso_forest, 'outlier_model.pkl')
    print("Outlier Detector saved as outlier_model.pkl")

if __name__ == "__main__":
    train_outlier()
