# multimodal_model.py
# Clinical Decision Intelligence CNN – Multi-Task Model

from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from keras.optimizers import Adam
from config import IMG_SIZE
from keras.metrics import AUC
metrics=[AUC(name="auc")]

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
