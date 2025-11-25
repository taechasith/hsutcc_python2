import numpy as np
import kagglehub
import os
import random
import traceback

try:
    print("Attempting to download dataset...")
    path = kagglehub.dataset_download("orvile/mhsma-sperm-morphology-analysis-dataset")
    print(f"Dataset downloaded to: {path}")
    
    label_files = [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.startswith("y_") and f.endswith(".npy")]
    if label_files:
        selected_file = random.choice(label_files)
        print(f"Selected file: {selected_file}")
        
        raw_data = np.load(selected_file)
        print(f"Raw data type: {type(raw_data)}")
        print(f"Raw data: {raw_data}")
        
        if isinstance(raw_data, np.ndarray):
            print(f"Raw data shape: {raw_data.shape}")
            print(f"Raw data ndim: {raw_data.ndim}")
        
        labels = np.atleast_1d(raw_data)
        print(f"Processed labels type: {type(labels)}")
        print(f"Processed labels shape: {labels.shape}")
        print(f"Processed labels: {labels}")
        
        print(f"Len labels: {len(labels)}")
        
    else:
        print("No label files found.")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
