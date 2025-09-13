# Breast Cancer Screening – ML + FastAPI + Docker

End-to-end binary classification project packaged as a small web service. 
You can reproduce data splits and training from notebooks, serve the best model behind a FastAPI endpoint, and ship it with Docker.

---

## Repository layout

```
app/
  main.py                 # FastAPI app (GET / health, POST /predict)
  model/
    model.py              # Loads model + metadata and runs predict()
data/
  data.csv                # raw dataset (id, diagnosis, 30 numeric features, Unnamed: 32)
  processed/              # created by 01 notebook
    train.csv
    test.csv
models/
  logisticregression.joblib   # fitted pipeline saved by 04 notebook
  logreg_meta.json            # metadata: features list, threshold, version, example record
notebooks/
  01_data_import_eda.ipynb
  02_perceptron_and_gd.ipynb
  03_mlp_and_kfold.ipynb
  04_logistic_regression_grid.ipynb
testing-api.ipynb
Dockerfile
requirements.txt
```

> **Note**: In the uploaded files, `main.py` and `model.py` are at the project root. 
> The FastAPI import path expects `app/main.py` and `app/model/model.py`. 
> Create that `app/` package (or adjust imports) before running Docker.

---

## Data & preprocessing

- Raw file: `data/data.csv` with columns `id`, `diagnosis` (B/M), 30 numeric features, and an unused `Unnamed: 32` column.  
- Notebook **01** converts the target to integers (`B → 0`, `M → 1`), drops `id` and `Unnamed: 32`, performs a **stratified** train/test split, and saves:
  - `data/processed/train.csv`
  - `data/processed/test.csv`
  - An optional SMOTE-resampled training set for analysis (`train_resampled.csv`).

---

## Models

You’ll find three modeling notebooks:

- **02_perceptron_and_gd.ipynb** – Baseline perceptron and gradient-descent explorations (with notes on class imbalance and SMOTE).
- **03_mlp_and_kfold.ipynb** – MLP with StandardScaler, **Stratified K-Fold**, and test-set evaluation; saves `models/model_mlp.joblib` and `models/meta_mlp.json` (for comparison).
- **04_logistic_regression_grid.ipynb** – Logistic Regression with `GridSearchCV` across solvers/penalties/C. 
  - Optimizes **recall** (and tracks **F2** = recall‑weighted score), then evaluates on the test set.
  - Saves:
    - `models/logisticregression.joblib` (the pipeline)
    - `models/logreg_meta.json` containing:
      - `features`: exact training feature order
      - `threshold`: default decision threshold (0.50)
      - `model_version`: ISO date
      - `test_metrics`: summary metrics
      - `example_record`: one valid feature dict for quick API testing

**Why metadata?** The API loads this file at startup to enforce the exact input schema and threshold and to expose a health/version check.

---

## API (FastAPI)

- **GET /** → `{"health_check": "OK", "model_version": "<YYYY-MM-DD>"}`
- **POST /predict** → accepts a JSON object with **exactly** the training features (same names/order as in `logreg_meta.json`) and returns:
  - `proba` (positive‑class probability),
  - `prediction` (`1` if `proba ≥ threshold`, else `0`),
  - `threshold` (from metadata).

### Run locally (without Docker)

1) Create the expected layout:

```
mkdir -p app/model models data/processed
mv main.py app/main.py
mv model.py app/model/model.py
# put your trained files here:
# models/logisticregression.joblib
# models/logreg_meta.json
```

2) Install deps (Python 3.12 recommended to match Docker):

```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Example `requirements.txt` (runtime only):
```
fastapi
uvicorn[standard]
scikit-learn
pandas
numpy
pydantic>=2,<3
joblib
```

3) Launch the API:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Example requests

Health check:
```
curl -s http://localhost:8000/ | jq
```

Prediction (schema uses **exact** feature names from training):
```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d @- <<'JSON'
{
  "radius_mean": 14.127,
  "texture_mean": 19.29,
  "perimeter_mean": 91.969,
  "area_mean": 654.889,
  "smoothness_mean": 0.096,
  "compactness_mean": 0.104,
  "concavity_mean": 0.089,
  "concave points_mean": 0.049,
  "symmetry_mean": 0.181,
  "fractal_dimension_mean": 0.063,
  "radius_se": 0.405,
  "texture_se": 1.217,
  "perimeter_se": 2.866,
  "area_se": 40.337,
  "smoothness_se": 0.007,
  "compactness_se": 0.025,
  "concavity_se": 0.032,
  "concave points_se": 0.012,
  "symmetry_se": 0.021,
  "fractal_dimension_se": 0.004,
  "radius_worst": 16.269,
  "texture_worst": 25.677,
  "perimeter_worst": 107.261,
  "area_worst": 880.583,
  "smoothness_worst": 0.132,
  "compactness_worst": 0.254,
  "concavity_worst": 0.272,
  "concave points_worst": 0.115,
  "symmetry_worst": 0.29,
  "fractal_dimension_worst": 0.084
}
JSON
```

---

## Training workflow (reproducible)

1) Place the raw data at `data/data.csv`.
2) Run **01_data_import_eda.ipynb** to create `data/processed/train.csv` and `test.csv`.
3) Run **04_logistic_regression_grid.ipynb** to train LR and write:
   - `models/logisticregression.joblib`
   - `models/logreg_meta.json` (features, threshold, model_version, example_record, test metrics)
4) (Optional) Run **03_mlp_and_kfold.ipynb** to train and compare an MLP.

The FastAPI service reads the files in `models/` at startup; override with `MODEL_DIR=/path/to/models` if you keep them elsewhere.

---

## Docker

Build the image from the project root (the `Dockerfile` assumes `app/` and `models/` exist):

```
docker build -t bc-screening:latest .
```

Run the container on port 8000:

```
docker run --rm -p 8000:8000 bc-screening:latest
```

If you’re mounting a different models directory at runtime:

```
docker run --rm -p 8000:8000 -e MODEL_DIR=/models -v $PWD/models:/models bc-screening:latest
```

---

## Configuration

- **MODEL_DIR**: environment variable to override where the API looks for model files (`logisticregression.joblib` and `logreg_meta.json`). Defaults to `./models` relative to repository root.

---

## Dev dependencies (optional for notebooks)

Add these to your `requirements.txt` (or a separate `requirements-dev.txt`) if you’ll run the notebooks:

```
jupyter
matplotlib
seaborn
imbalanced-learn
```

---

## Testing

Use **`testing-api.ipynb`** for quick local/Docker sanity checks against `/` and `/predict`. It auto-loads the `example_record` from `logreg_meta.json` and measures simple latency, then posts to your chosen endpoint.

---

## Notes & Gotchas

- The API **forbids unknown keys** and requires **all** features present (no extras, no omissions). 
- Train/test splits are **stratified**; SMOTE is applied **only to training data** for analysis.
- The decision threshold is loaded from metadata and applied uniformly for inference; tune it during validation if your use‑case prioritizes recall, precision, or F2.

---
