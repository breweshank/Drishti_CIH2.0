import numpy as np
import pickle
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# === Load preprocessed data ===
with open(r"E:\CIH\processed_data.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Type of data: {type(data)}")

# Unpack the data (processed_data_list, participant_ids)
processed_data_list, participant_ids_list = data

# Extract features and PHQ-8 scores
features = np.array([item[0] for item in processed_data_list])
phq_scores = np.array([item[1] for item in processed_data_list])

# === Classify PHQ-8 scores ===
def classify_phq8(score):
    if score <= 4:
        return 0  # No Depression
    elif score <= 9:
        return 1  # Mild Depression
    else:
        return 2  # Moderate/Severe Depression

labels = np.array([classify_phq8(s) for s in phq_scores])
labels_cat = to_categorical(labels, num_classes=3)

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_cat, test_size=0.2, random_state=42
)

# === Build model ===
input_dim = features.shape[1]

input_layer = Input(shape=(input_dim,), name="features_input")
x = Dense(128, activation="relu")(input_layer)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(3, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# === Train ===
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

# === Save the model ===
os.makedirs("models", exist_ok=True)
model.save("")

print("✅ Model saved successfully!")
