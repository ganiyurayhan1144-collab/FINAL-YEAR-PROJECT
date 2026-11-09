
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
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


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = BASE_DIR / "model"

# Load trained model and scaler
model = tf.keras.models.load_model(MODEL_DIR / "phishing_model.keras")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# Load dataset
data = pd.read_csv(DATASET_PATH)
data = data.dropna(subset=["URL", "label"])

urls = data["URL"].astype(str).tolist()
y = 1 - data["label"].to_numpy(dtype="int32")

# Recreate the same test split used during training
indices = np.arange(len(y))

_, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Prepare test features
test_urls = [urls[i] for i in test_indices]

X_test_features = np.array(
    [extract_features(url) for url in test_urls],
    dtype="float32"
)

X_test_features = scaler.transform(X_test_features)


# Encode URLs exactly as during training
def encode_url(url):
    chars = []

    for char in url[:200]:
        code = ord(char)
        chars.append(code if 0 < code < 128 else 1)

    chars += [0] * (200 - len(chars))
    return chars


X_test_sequences = np.array(
    [encode_url(url) for url in test_urls],
    dtype="int32"
)

y_test = y[test_indices]

# Make predictions
probabilities = model.predict(
    [X_test_sequences, X_test_features],
    verbose=1
).ravel()

y_pred = (probabilities >= 0.5).astype("int32")


# Calculate evaluation metrics
print("\nMODEL PERFORMANCE")
print("========================")

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0) * 100, 2), "%")
print("F1-score:", round(f1_score(y_test, y_pred, zero_division=0) * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Legitimate", "Phishing"],
    zero_division=0
))

print("\nCONFUSION MATRIX")
print("Rows = Actual, Columns = Predicted")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# Save evaluation results
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

results = pd.DataFrame({
    "URL": test_urls,
    "Actual": y_test,
    "Predicted": y_pred,
    "Phishing_Score": probabilities
})

results.to_csv(results_dir / "evaluation_results.csv", index=False)

print("\nEvaluation completed!")
print("Results saved in results/evaluation_results.csv")
