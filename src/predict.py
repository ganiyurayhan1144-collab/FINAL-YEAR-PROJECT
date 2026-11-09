from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import tensorflow as tf

try:
    from .features import extract_features
except ImportError:  # pragma: no cover - script execution fallback
    from features import extract_features


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
SEQUENCE_LENGTH = 200
FEATURE_NAMES = (
    "url_length",
    "domain_length",
    "dot_count",
    "hyphen_count",
    "at_count",
    "question_count",
    "equal_count",
    "slash_count",
    "digit_count",
    "letter_count",
    "uses_https",
    "has_ip_address",
    "has_at_symbol",
    "has_path",
)


class PredictionInputError(ValueError):
    """Raised when a submitted URL is not supported by the model."""


@dataclass(frozen=True)
class FeatureImpact:
    name: str
    value: int
    score_change: float


@dataclass(frozen=True)
class PredictionResult:
    url: str
    phishing_score: float
    is_phishing: bool
    feature_impacts: tuple[FeatureImpact, ...]


@lru_cache(maxsize=1)
def _load_artifacts():
    model_path = MODEL_DIR / "phishing_model.keras"
    scaler_path = MODEL_DIR / "scaler.pkl"
    missing = [path for path in (model_path, scaler_path) if not path.is_file()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required model artifact(s) not found: {paths}")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    if scaler.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError(
            "The saved feature scaler does not match the URL feature schema."
        )
    return model, scaler


def normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        raise PredictionInputError("Enter a URL to analyze.")
    if len(value) > 2048:
        raise PredictionInputError("URLs must be 2,048 characters or fewer.")
    if any(character.isspace() for character in value):
        raise PredictionInputError("The URL cannot contain whitespace.")
    if "://" not in value:
        value = f"https://{value}"

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise PredictionInputError("Enter a valid HTTP or HTTPS URL.") from error

    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise PredictionInputError(
            "Enter a URL with a hostname using HTTP or HTTPS."
        )
    return value


def encode_url(url: str) -> np.ndarray:
    chars = [
        ord(char) if 0 < ord(char) < 128 else 1
        for char in url[:SEQUENCE_LENGTH]
    ]
    chars.extend([0] * (SEQUENCE_LENGTH - len(chars)))
    return np.asarray([chars], dtype="int32")


def predict_url(url: str) -> PredictionResult:
    normalized_url = normalize_url(url)
    model, scaler = _load_artifacts()

    raw_features = np.asarray(
        [extract_features(normalized_url)], dtype="float32"
    )
    scaled_features = scaler.transform(raw_features)
    sequence = encode_url(normalized_url)

    # Reset one scaled feature at a time to its training mean (zero), leaving
    # the URL-character input and all other numerical features unchanged.
    perturbed_features = np.repeat(scaled_features, len(FEATURE_NAMES), axis=0)
    np.fill_diagonal(perturbed_features, 0.0)
    feature_batch = np.concatenate((scaled_features, perturbed_features), axis=0)
    sequence_batch = np.repeat(sequence, len(FEATURE_NAMES) + 1, axis=0)
    scores = model.predict(
        [sequence_batch, feature_batch], verbose=0
    ).reshape(-1)

    phishing_score = float(scores[0])
    impacts = tuple(
        sorted(
            (
                FeatureImpact(
                    name=name,
                    value=int(value),
                    score_change=phishing_score - float(scores[index + 1]),
                )
                for index, (name, value) in enumerate(
                    zip(FEATURE_NAMES, raw_features[0])
                )
            ),
            key=lambda impact: abs(impact.score_change),
            reverse=True,
        )
    )
    return PredictionResult(
        url=normalized_url,
        phishing_score=phishing_score,
        is_phishing=phishing_score >= 0.5,
        feature_impacts=impacts,
    )


if __name__ == "__main__":
    submitted_url = input("Enter a URL to check: ")
    result = predict_url(submitted_url)
    verdict = "may be phishing" if result.is_phishing else "not flagged as phishing"
    print(f"This URL {verdict}.")
    print(f"Phishing score: {result.phishing_score:.2%}")
