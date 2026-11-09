
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import shap

try:
    from .features import extract_features
except ImportError:  # pragma: no cover - script execution fallback
    from features import extract_features

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = BASE_DIR / "model"

# Load trained model and scaler
model = tf.keras.models.load_model(MODEL_DIR / "phishing_model.keras")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# Feature names
feature_names = [
    "url_length", "domain_length", "dot_count",
    "hyphen_count", "at_count", "question_count",
    "equal_count", "slash_count", "digit_count",
    "letter_count", "uses_https", "has_ip_address",
    "has_at_symbol", "has_path"
]

# Encode URL exactly as in training
def encode_url(url):
    chars = []

    for char in url[:200]:
        code = ord(char)
        chars.append(code if 0 < code < 128 else 1)

    chars += [0] * (200 - len(chars))
    return np.array([chars], dtype="int32")


# Prepare background data for SHAP
data = pd.read_csv(
    DATASET_PATH,
    usecols=["URL"],
    nrows=5000
).dropna()

background_features = np.array(
    [extract_features(url) for url in data["URL"].astype(str)],
    dtype="float32"
)

background_features = scaler.transform(background_features)

background = pd.DataFrame(
    background_features,
    columns=feature_names
).sample(n=30, random_state=42)


# Ask for URL
url = input("Enter a URL to explain: ")

target_features = np.array(
    [extract_features(url)],
    dtype="float32"
)

target_features = scaler.transform(target_features)


# Keep the URL character input fixed while SHAP varies numerical features
sequence = encode_url(url)

def predict_from_features(features):
    repeated_sequences = np.repeat(
        sequence, len(features), axis=0
    )

    return model.predict(
        [repeated_sequences, features],
        verbose=0
    )


# Generate SHAP explanation
print("\nGenerating SHAP explanation...")

explainer = shap.KernelExplainer(
    predict_from_features,
    background
)

shap_values = explainer.shap_values(
    target_features,
    nsamples=100
)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values).squeeze()

# Display prediction and feature contributions
score = float(
    predict_from_features(target_features)[0][0]
)

print("\nSHAP EXPLANATION")
print("-------------------------")
print("URL:", url)
print("Phishing score:", round(score * 100, 2), "%")

print("\nTop contributing features:")

for index in np.argsort(np.abs(shap_values))[::-1][:5]:
    direction = (
        "increases phishing score"
        if shap_values[index] > 0
        else "decreases phishing score"
    )

    print(
        feature_names[index],
        ":", round(float(shap_values[index]), 5),
        "-", direction
    )

print("\nSHAP explanation completed!")
