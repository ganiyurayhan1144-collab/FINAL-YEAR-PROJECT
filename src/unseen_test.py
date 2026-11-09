
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    from .features import extract_features
except ImportError:  # pragma: no cover - script execution fallback
    from features import extract_features


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = BASE_DIR / "model"

# Load dataset
data = pd.read_csv(DATASET_PATH)
data = data.dropna(subset=["URL", "label"]).copy()

urls = data["URL"].astype(str).tolist()
y = 1 - data["label"].to_numpy(dtype="int32")


# Extract hostname for grouping
def get_domain(url):
    hostname = urlparse(url).hostname

    if hostname:
        return hostname.lower()

    return ""


groups = np.array([get_domain(url) for url in urls])

# Remove rows with missing domains
valid = groups != ""
urls = np.array(urls)[valid]
y = y[valid]
groups = groups[valid]


# Split by domain, not by individual URL
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(urls, y, groups)
)

train_urls = urls[train_indices]
test_urls = urls[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]


# Prepare numerical features
X_train = np.array(
    [extract_features(url) for url in train_urls],
    dtype="float32"
)

X_test = np.array(
    [extract_features(url) for url in test_urls],
    dtype="float32"
)


# Fit scaler using training domains only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Encode URLs
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


# Load the existing trained model
model = tf.keras.models.load_model(
    MODEL_DIR / "phishing_model.keras"
)

# Evaluate on unseen domains
probabilities = model.predict(
    [X_test_seq, X_test],
    verbose=1
).ravel()

predictions = (probabilities >= 0.5).astype("int32")


print("\nUNSEEN-DOMAIN TEST RESULTS")
print("==========================")

print("Training URLs:", len(train_urls))
print("Testing URLs:", len(test_urls))
print("Training domains:", len(set(groups[train_indices])))
print("Testing domains:", len(set(groups[test_indices])))

print(
    "Domain overlap:",
    len(set(groups[train_indices]) & set(groups[test_indices]))
)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, predictions, zero_division=0) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, predictions, zero_division=0) * 100, 2), "%")
print("F1-score:", round(f1_score(y_test, predictions, zero_division=0) * 100, 2), "%")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Phishing"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nUnseen-domain testing completed!")
