#contain helper funtions 
import numpy as np
from PIL import Image

def preprocess_image(path, size):
    img = Image.open(path).convert("RGB").resize(size)
    return np.expand_dims(np.array(img) /255.0,axis=0)
print("✅ utilis.py loaded and preprocess_image() ready for use!")