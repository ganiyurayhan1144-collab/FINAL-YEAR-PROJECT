
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .features import extract_features
    from .model import build_model
except ImportError:  # pragma: no cover - script execution fallback
    from features import extract_features
    from model import build_model

# Project locations
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Load real dataset
data = pd.read_csv(DATASET_PATH)
data = data.dropna(subset=["URL", "label"])

# Prepare URL data
urls = data["URL"].astype(str).tolist()

# Extract 14 numerical features
X_features = np.array(
    [extract_features(url) for url in urls],
    dtype="float32"
)

# Encode URL characters
def encode_url(url):
    chars = []
    for char in url[:200]:
        code = ord(char)
        chars.append(code if 0 < code < 128 else 1)

    chars += [0] * (200 - len(chars))
    return chars

X_sequences = np.array(
    [encode_url(url) for url in urls],
    dtype="int32"
)

# Dataset labels: 1 = legitimate, 0 = phishing
# Convert to: 1 = phishing, 0 = legitimate
y = 1 - data["label"].to_numpy(dtype="float32")

# Split data
X_seq_train, X_seq_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
    X_sequences,
    X_features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_feat_train = scaler.fit_transform(X_feat_train)
X_feat_test = scaler.transform(X_feat_test)

# Build hybrid ANN + BiLSTM
model = build_model()

# Train
model.fit(
    [X_seq_train, X_feat_train],
    y_train,
    epochs=3,
    batch_size=256,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(
    [X_seq_test, X_feat_test],
    y_test
)

# Save model and scaler
model.save(MODEL_DIR / "phishing_model.keras")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

print("\nTraining completed!")
print("Test accuracy:", round(float(accuracy) * 100, 2), "%")
print("Model saved successfully!")
