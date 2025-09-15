# Breast Cancer Screening – ML + FastAPI + Docker

End-to-end binary classification project packaged as a small web service. 
You can reproduce data splits and training from notebooks, serve the best model behind a FastAPI endpoint, and ship it with Docker.

---
<img width="2048" height="1447" alt="image" src="https://github.com/user-attachments/assets/bed58c10-66dd-40e0-9b4d-636d47424afd" />

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
"example_record": {
    "radius_mean": 11.41,
    "texture_mean": 10.82,
    "perimeter_mean": 73.34,
    "area_mean": 403.3,
    "smoothness_mean": 0.09373,
    "compactness_mean": 0.06685,
    "concavity_mean": 0.03512,
    "concave points_mean": 0.02623,
    "symmetry_mean": 0.1667,
    "fractal_dimension_mean": 0.06113,
    "radius_se": 0.1408,
    "texture_se": 0.4607,
    "perimeter_se": 1.103,
    "area_se": 10.5,
    "smoothness_se": 0.00604,
    "compactness_se": 0.01529,
    "concavity_se": 0.01514,
    "concave points_se": 0.00646,
    "symmetry_se": 0.01344,
    "fractal_dimension_se": 0.002206,
    "radius_worst": 12.82,
    "texture_worst": 15.97,
    "perimeter_worst": 83.74,
    "area_worst": 510.5,
    "smoothness_worst": 0.1548,
    "compactness_worst": 0.239,
    "concavity_worst": 0.2102,
    "concave points_worst": 0.08958,
    "symmetry_worst": 0.3016,
    "fractal_dimension_worst": 0.08523
  }
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
docker build -t cancer-api:latest .
```

Run the container on port 8000:

```
 docker run -d --name cancer-container-api -p 8000:8000 cancer-api
```

---

## Configuration

- **MODEL_DIR**: environment variable to override where the API looks for model files (`logisticregression.joblib` and `logreg_meta.json`). Defaults to `./models` relative to repository root.

---

## Dev dependencies (optional for notebooks)

Add these to your `requirements.txt` (or a separate `requirements-dev.txt`) if you’ll run the notebooks:

```
fastapi==0.116.1
uvicorn==0.35.0
pydantic==2.11.7
scikit-learn==1.7.1
pandas==2.3.2
numpy==2.3.2
joblib==1.5.2
```

---

## Testing

Use **`testing-api.ipynb`** for quick local/Docker sanity checks against `/` and `/predict`. It auto-loads the `example_record` from `logreg_meta.json` and measures simple latency, then posts to your chosen endpoint.

---

## Notes & Gotchas

- The API **forbids unknown keys** and requires **all** features present (no extras, no omissions). 
- Train/test splits are **stratified**; SMOTE is applied **only to training data** for analysis.
- The decision threshold is loaded from metadata and applied uniformly for inference. Tune it during validation if your use‑case prioritizes recall, precision, or F2.

---

## License
Project is distributed under [MIT License](https://github.com/AngelX62/Breast-Cancer-Diagnosis/blob/main/LICENSE)
