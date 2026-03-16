# Handles non Images data like csv, 
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