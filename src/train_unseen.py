
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

try:
    from .features import extract_features
    from .model import build_model
except ImportError:  # pragma: no cover - script execution fallback
    from features import extract_features
    from model import build_model


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
data = pd.read_csv(DATASET_PATH)
data = data.dropna(subset=["URL", "label"]).copy()

urls = data["URL"].astype(str).to_numpy()
y = 1 - data["label"].to_numpy(dtype="float32")

# Group URLs by hostname
from urllib.parse import urlparse

def get_domain(url):
    return urlparse(url).hostname or ""

groups = np.array([get_domain(url).lower() for url in urls])

valid = groups != ""
urls, y, groups = urls[valid], y[valid], groups[valid]

# Split by domain
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(urls, y, groups)
)

train_urls = urls[train_idx]
test_urls = urls[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print("Training URLs:", len(train_urls))
print("Testing URLs:", len(test_urls))
print("Preparing features...")

# Extract numerical features
X_train = np.array(
    [extract_features(url) for url in train_urls],
    dtype="float32"
)

X_test = np.array(
    [extract_features(url) for url in test_urls],
    dtype="float32"
)

# Fit scaler ONLY on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode URL characters
def encode_url(url):
    chars = []

    for char in url[:200]:
        code = ord(char)
        chars.append(code if 0 < code < 128 else 1)

    chars += [0] * (200 - len(chars))
    return chars

X_train_seq = np.array(
    [encode_url(url) for url in train_urls],
    dtype="int32"
)

X_test_seq = np.array(
    [encode_url(url) for url in test_urls],
    dtype="int32"
)

# Build and train a fresh model
model = build_model()

print("\nTraining new model on training domains...")

model.fit(
    [X_train_seq, X_train],
    y_train,
    epochs=3,
    batch_size=256,
    validation_split=0.2
)

# Evaluate on completely held-out domains
probabilities = model.predict(
    [X_test_seq, X_test],
    verbose=1
).ravel()

predictions = (probabilities >= 0.5).astype("int32")

print("\nUNSEEN-DOMAIN PERFORMANCE")
print("=========================")

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, predictions, zero_division=0) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, predictions, zero_division=0) * 100, 2), "%")
print("F1-score:", round(f1_score(y_test, predictions, zero_division=0) * 100, 2), "%")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nDomain overlap:",
      len(set(groups[train_idx]) & set(groups[test_idx])))

# Save separately; preserve the original model
model.save(MODEL_DIR / "unseen_phishing_model.keras")
joblib.dump(scaler, MODEL_DIR / "unseen_scaler.pkl")

print("\nUnseen-domain experiment completed!")
print("New model and scaler saved separately.")
