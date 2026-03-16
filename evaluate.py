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
    model = load_model(MODEL_SAVE_PATH, compile=False)

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
print("🔹 Evaluating model performance on validation set...")
