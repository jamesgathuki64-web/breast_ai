#Just but a CNN model in short image based image analysis
#this is the model brain
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
