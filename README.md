# Zero-Day Phishing URL Detection

Academic demonstration of a hybrid ANN + BiLSTM model for classifying URLs as
potentially phishing or legitimate, with a local feature-sensitivity view.

## Run the web app (Windows PowerShell)

From the project directory:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run .\src\app.py
```

The app expects the trained model and feature scaler at:

- `model/phishing_model.keras`
- `model/scaler.pkl`

It accepts HTTP or HTTPS URLs. If the scheme is omitted, HTTPS is added. URLs
are analyzed locally; the app does not open or fetch the submitted site.

## Project layout

- `src/app.py` — Streamlit interface
- `src/predict.py` — input validation, model inference, and feature sensitivity
- `src/features.py` — 14 URL-level numeric features
- `src/model.py` — ANN + bidirectional LSTM architecture
- `src/train_model.py` — train and save the standard model
- `src/train_unseen.py` — train a separate model using domain-grouped splits
- `src/evaluate.py` — evaluate the standard model and save per-URL results
- `src/explain.py` — SHAP-based command-line explanation
- `dataset/` — input datasets
- `model/` — trained models and scalers
- `results/` — evaluation output

The training scripts map dataset label `1` to legitimate and label `0` to
phishing, so the model's positive output is the phishing score.

## Interpretation and limitations

The app's score is a model output, not a calibrated probability or proof of
safety. Its feature table measures how the score changes when each numerical
feature is independently reset to its training mean while the character-based
URL input is held fixed. This is a local sensitivity check, not a causal
explanation. Treat predictions as research output and verify URLs independently.
