
import pandas as pd

try:
    from .features import extract_features
except ImportError:  # pragma: no cover - script execution fallback
    from features import extract_features

# Load the dataset
data = pd.read_csv("../dataset/phishing_dataset.csv")

# Extract URL features
X = data["url"].apply(extract_features).tolist()

# Get the labels
y = data["label"]

print("Dataset loaded successfully!")
print("Total URLs:", len(X))
print("Total labels:", len(y))
print("Feature extraction completed!")
