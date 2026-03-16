# config.py
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

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "breast_model.h5")

# ✅ Only create folders that should be empty
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
print("✅ Configuration loaded  and tested successfully!")
# dataset.py
# dataset.py
# Multi-output Data Loader for Clinical Breast AI Model

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from config import METADATA_CSV, IMG_SIZE, BATCH_SIZE, SEED


# ---------------------------------------------------
# 1️⃣ Custom Multi-Output Generator (Stable Version)
# ---------------------------------------------------
class MultiOutputGenerator(Sequence):
    def __init__(self, image_generator, dataframe):
        self.image_generator = image_generator
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.image_generator)

    def __getitem__(self, idx):
        images = self.image_generator[idx]

        batch_start = idx * self.image_generator.batch_size
        batch_end = batch_start + len(images)

        batch_indices = self.image_generator.index_array[batch_start:batch_end]
        batch_df = self.df.iloc[batch_indices]

        labels = {
            "malignancy": batch_df["malignancy"].values.astype("float32"),

            "molecular_subtype": tf.keras.utils.to_categorical(
                batch_df["molecular_subtype"], 4
            ),

            "aggressiveness": batch_df["aggressiveness"].values.astype("float32"),

            "lymph_node_risk": batch_df["lymph_node"].values.astype("float32"),

            "stage_prediction": tf.keras.utils.to_categorical(
                batch_df["stage"], 2
            ),
        }

        return images, labels


# ---------------------------------------------------
# 2️⃣ Load & Prepare Dataset
# ---------------------------------------------------
def load_data():
    df = pd.read_csv(METADATA_CSV)

    # Binary label
    df["malignancy"] = df["label"].astype(float)

    # Safe numeric conversions
    df["aggressiveness"] = pd.to_numeric(
        df["Aggressiveness"], errors="coerce"
    ).fillna(0)

    df["lymph_node"] = pd.to_numeric(
        df["Lymph_Node_Involvement"], errors="coerce"
    ).fillna(0)

    df["molecular_subtype"] = pd.to_numeric(
        df["Molecular_Subtype"], errors="coerce"
    ).fillna(0).astype(int)

    df["stage"] = pd.to_numeric(
        df["Breast_Cancer_Stage"], errors="coerce"
    ).fillna(0).astype(int)

    # Train / Validation Split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["malignancy"]
    )

    return train_df, val_df


# ---------------------------------------------------
# 3️⃣ Build Generators
# ---------------------------------------------------
def get_generators(train_df, val_df):

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    image_dir = os.path.join(
        os.path.dirname(METADATA_CSV),
        "images"
    )

    train_img_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="image_path",
        y_col=None,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=True,
        seed=SEED
    )

    val_img_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=image_dir,
        x_col="image_path",
        y_col=None,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False
    )

    # Wrap with multi-output generator
    train_multi_gen = MultiOutputGenerator(train_img_gen, train_df)
    val_multi_gen = MultiOutputGenerator(val_img_gen, val_df)

    return train_multi_gen, val_multi_gen


# ---------------------------------------------------
# Debug Info
# ---------------------------------------------------
print("Metadata path:", METADATA_CSV)
print("File exists:", os.path.exists(METADATA_CSV))
print("✅ dataset.py loaded successfully")

# model_cnn.py
import tensorflow as tf
from keras import layers, Model
from config import IMG_SIZE

def build_image_model(base_trainable=False):
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet', pooling='avg'
    )
    base.trainable = base_trainable
    x = layers.Input(shape=(*IMG_SIZE, 3))
    y = base(x, training=False)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(256, activation='relu')(y)
    
    feature_extractor = Model(inputs=x, outputs=y, name='image_feature_extractor')
    return feature_extractor
# model_tabular.py
from keras import layers, models

def build_tabular_model(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
print("✅ model_tabular.py loaded and build_tabular_model() ready for training!") 
# multimodal_model.py
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from keras.optimizers import Adam
from config import IMG_SIZE


def build_clinical_model():

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    # 🔹 Multi-task outputs

    # 1️⃣ Malignancy (binary)
    malignancy = Dense(
        1, activation='sigmoid',
        name="malignancy"
    )(x)

    # 2️⃣ Molecular Subtype (4-class)
    molecular_subtype = Dense(
        4, activation='softmax',
        name="molecular_subtype"
    )(x)

    # 3️⃣ Aggressiveness (regression score)
    aggressiveness = Dense(
        1, activation='linear',
        name="aggressiveness"
    )(x)

    # 4️⃣ Lymph Node Risk (binary)
    lymph_node_risk = Dense(
        1, activation='sigmoid',
        name="lymph_node_risk"
    )(x)

    # 5️⃣ Stage Prediction (2-class)
    stage_prediction = Dense(
        2, activation='softmax',
        name="stage_prediction"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=[
            malignancy,
            molecular_subtype,
            aggressiveness,
            lymph_node_risk,
            stage_prediction
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss={
            "malignancy": "binary_crossentropy",
            "molecular_subtype": "categorical_crossentropy",
            "aggressiveness": "mse",
            "lymph_node_risk": "binary_crossentropy",
            "stage_prediction": "categorical_crossentropy"
        },
        metrics={
            "malignancy": ["accuracy"],
            "molecular_subtype": ["accuracy"],
            "stage_prediction": ["accuracy"]
        }
    )

    model.summary()

    return model
print("✅ multimodal_model.py loaded and build_clinical_model() ready for training!")
# train.py
# train.py
# Trains the clinical multimodal breast cancer model

import tensorflow as tf
from multimodal_model import build_clinical_model
from dataset import load_data, get_generators
from config import MODEL_SAVE_PATH

print("✅ train.py running — Clinical Multimodal Version")


def train_model():

    print("🔹 Loading dataset...")
    train_df, val_df = load_data()

    print("🔹 Creating generators...")
    train_gen, val_gen = get_generators(train_df, val_df)

    print("🔹 Building clinical model...")
    model = build_clinical_model()

    print("🔹 Model Summary:")
    model.summary()

    # Callbacks (important for uniqueness + performance)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    print("🔹 Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Training complete!")
    print(f"📁 Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
    print("✅ train.py complete  and ready for evaluation or deployment!")
# evaluate.py
# evaluate.py
# Multi-output Evaluation – Focus on Malignancy AUC

import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from dataset import load_data, get_generators
from config import MODEL_SAVE_PATH


def evaluate_model():

    print("🔹 Loading validation data...")
    train_df, val_df = load_data()

    print("🔹 Preparing validation generator...")
    _, val_gen = get_generators(train_df, val_df)

    print(f"🔹 Loading model from: {MODEL_SAVE_PATH}")
    model = load_model(MODEL_SAVE_PATH)

    print("🔹 Generating predictions...")
    
    preds = model.predict(val_gen, verbose=1)

    # 🔹 Extract malignancy predictions
    malignancy_preds = preds[0].ravel()

    # 🔹 Get true labels from dataframe
    y_true = val_df["malignancy"].values

    print("🔹 Computing AUC for malignancy...")
    auc = roc_auc_score(y_true, malignancy_preds)

    print(f"✅ Malignancy AUC: {auc:.4f}")


if __name__ == "__main__":
    evaluate_model()

print("✅ evaluate.py started")
# utils.py
#contain helper funtions 
import numpy as np
from PIL import Image

def preprocess_image(path, size):
    img = Image.open(path).convert("RGB").resize(size)
    return np.expand_dims(np.array(img) /255.0,axis=0)
print("✅ utilis.py loaded and preprocess_image() ready for use!")
# app.py
from flask import Flask, request, jsonify, render_template_string
import os
from keras.models import load_model
from config import MODEL_DIR, IMG_SIZE
from utils import load_image_for_inference
import pandas as pd
import numpy as np

app = Flask(__name__)
model = load_model(os.path.join(MODEL_DIR, "multimodal_best.h5"), compile=False)

# Simple HTML UI
HTML = """
<!doctype html>
<title>Breast AI Demo</title>
<h2>Upload image and CSV row (tabular) for inference</h2>
<form method=post enctype=multipart/form-data action="/predict">
  Image: <input type=file name=image><br>
  Age: <input type=text name=age><br>
  biomarker1: <input type=text name=biomarker1><br>
  biomarker2: <input type=text name=biomarker2><br>
  <input type=submit value=Predict>
</form>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    # basic single-row inference
    image = request.files.get('image')
    if image is None:
        return jsonify({"error":"No image uploaded"}), 400
    img_path = os.path.join("tmp_upload.jpg")
    image.save(img_path)
    img_arr = load_image_for_inference(img_path, IMG_SIZE)

    # gather tabular features in same order used for training
    # for demo: assume features are ['age','biomarker1','biomarker2']
    features = ['age','biomarker1','biomarker2']
    vals = []
    for f in features:
        v = request.form.get(f, 0)
        try:
            vals.append(float(v))
        except:
            vals.append(0.0)
    tab = np.array(vals).reshape(1,-1)
    pred = model.predict([img_arr, tab]).ravel()[0]
    os.remove(img_path)
    return jsonify({"probability": float(pred), "label": int(pred>=0.5)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
# shap_explain.py
import shap
import numpy as np
from keras.models import load_model
from dataset import prepare_datasets
from config import MODEL_DIR

def explain():
    ( (X_train, y_train)), ( (X_val, y_val)), (test_img_ds, (X_test, y_test)), feature_cols = prepare_datasets()
    model = load_model(f"{MODEL_DIR}/multimodal_best.h5", compile=False)
    # For SHAP it's easiest to explain the tabular branch:
    # Build a wrapper to predict from tabular only using fixed dummy image (average)
    mean_img = None
    imgs = []
    for img_batch, _ in test_img_ds:
        imgs.append(img_batch.numpy())
    imgs = np.vstack(imgs)
    mean_img = np.mean(imgs, axis=0, keepdims=True)
    def tabular_predict(X_tab):
        imgs_batch = np.repeat(mean_img, X_tab.shape[0], axis=0)
        preds = model.predict([imgs_batch, X_tab])
        return preds

    explainer = shap.Explainer(tabular_predict, X_train[:100])
    shap_values = explainer(X_test[:50])
    shap.summary_plot(shap_values, X_test[:50], feature_names=feature_cols)

if __name__ == "_main_":
    explain()
