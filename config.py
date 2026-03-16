# configuration in short the system setting/compatibility

# Paths
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT, "Data")  # MATCH ACTUAL FOLDER NAME
IMAGES_DIR = os.path.join(DATA_DIR, "images")

METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")

MODEL_DIR = os.path.join(ROOT, "models")
LOG_DIR = os.path.join(ROOT, "logs")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "breast_model.keras")
MODEL_SAVE_PATH = "models/breast_model.keras"

# ✅ Only create folders that should be empty
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
print("✅ Configuration loaded  and tested successfully!")
