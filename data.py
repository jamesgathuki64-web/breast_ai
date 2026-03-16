# dataset.py
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from config import METADATA_CSV, IMAGES_DIR, IMG_SIZE, BATCH_SIZE, SEED
from sklearn.model_selection import train_test_split

data = pd.read_csv("Data/metadata.csv")
cols=['age', 'bmi', 'family_history', 'HER2', 'BRCA1', 'CA15_3',
       'smoking', 'alcohol', 'exercise']

def read_metadata():
    df = pd.read_csv(METADATA_CSV)
    # Ensure required columns: id,label,image_path (or image filename)
    # Example CSV columns: id,label,age,biomarker1,biomarker2,image_path
    return df

def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    print(img)
    return img

def make_image_dataset(df, subset='train'):
    AUTOTUNE = tf.data.AUTOTUNE
    # df with 'image_path' and 'label'
    paths = df['image_path'].apply(os.path.abspath).tolist()
    paths = [
    os.path.join(IMAGES_DIR, os.path.basename(p))
    for p in df['image_path']
]
    labels = df['label'].astype(int).tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    for p in paths[:5]:
     print("CHECK:", p, os.path.exists(p))
    def _load(path, label):
        return preprocess_image(path), label
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if subset == 'train':
        ds = ds.shuffle(2048, seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def get_tabular(df, cols):
    X = df[cols].astype(float).values
    y = df['label'].astype(int).values
    return X, y

def prepare_datasets(test_size=0.15, val_size=0.15, feature_cols=None, random_state=SEED):
    df = read_metadata()
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
    
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), stratify=train_df['label'], random_state=random_state)
    # image datasets
    train_img_ds = make_image_dataset(train_df, subset='train')
    val_img_ds = make_image_dataset(val_df, subset='val')
    test_img_ds = make_image_dataset(test_df, subset='test')
    # tabular
    if feature_cols is None:
        exclude = {'id','label','image_path'}
        feature_cols = [c for c in df.columns if c not in exclude]
    X_train, y_train = get_tabular(train_df, cols)
    X_val, y_val = get_tabular(val_df, cols)
    X_test, y_test = get_tabular(test_df, cols)
    return (train_img_ds, (X_train, y_train)), (val_img_ds, (X_val, y_val)), (test_img_ds, (X_test, y_test)), feature_cols
prepare_datasets()