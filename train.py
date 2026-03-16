# =====================================================
# train.py
# Trains the clinical multimodal breast cancer model
# =====================================================

import tensorflow as tf
import os
import numpy as np
from sklearn.utils import class_weight

from multimodal_model import build_clinical_model
from dataset import load_data, get_generators
from config import MODEL_SAVE_PATH
from utilis.database import save_training_patient

print("✅ train.py running — Clinical Multimodal Version")


def train_model():

    # ==========================================
    # LOAD DATASET
    # ==========================================
    print("🔹 Loading dataset...")
    train_df, val_df = load_data()

    # ==========================================
    # SAVE TRAINING DATASET TO DATABASE
    # ==========================================
    print("🔹 Saving training dataset to database...")

    for _, row in train_df.iterrows():

        patient_name = os.path.basename(row["image_path"])

        save_training_patient(
            patient_name,
            row["image_path"],
            row["malignancy"]
        )

    print("✅ Training patients stored in database")

    # ==========================================
    # CREATE DATA GENERATORS
    # ==========================================
    print("🔹 Creating generators...")

    train_gen, val_gen = get_generators(train_df, val_df)

    # ==========================================
    # COMPUTE CLASS WEIGHTS
    # ==========================================
    print("🔹 Computing class weights...")

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["malignancy"]),
        y=train_df["malignancy"]
    )

    class_weights = dict(enumerate(class_weights))

    print("✅ Class weights:", class_weights)

    # ==========================================
    # BUILD MODEL
    # ==========================================
    print("🔹 Building clinical model...")

    model = build_clinical_model()

    print("🔹 Model Summary:")
    model.summary()

    # ==========================================
    # TRAINING CALLBACKS
    # ==========================================

    callbacks = [

        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
            restore_best_weights=True
        ),

        # Reduce LR if stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.3,
            patience=3,
            verbose=1
        )
    ]

    # ==========================================
    # TRAIN MODEL
    # ==========================================

    print("🔹 Starting training...")

    history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

    print("✅ Training complete!")
    print(f"📁 Best model saved to: {MODEL_SAVE_PATH}")

    return history


# ==========================================
# RUN TRAINING
# ==========================================

if __name__ == "__main__":

    history = train_model()

    print("✅ train.py complete and ready for evaluation or deployment!")