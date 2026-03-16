# dataset.py
# Multi-output Data Loader for Clinical Breast AI Model

import os
import pandas as pd
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from config import METADATA_CSV, IMG_SIZE, BATCH_SIZE, SEED

# ---------------------------------------------------
# 1️⃣ Custom Multi-Output Generator (Stable Version)
# ---------------------------------------------------
ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)
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
