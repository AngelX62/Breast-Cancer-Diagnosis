
import json, joblib, numpy as np, pandas as pd
import gradio as gr
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
MODEL_DIR = None
for p in [BASE, *BASE.parents]:
    cand = p / "models"
    if cand.exists():
        MODEL_DIR = cand
        break
assert MODEL_DIR, f"Couldn't find a 'models' folder starting from {BASE}"

# exact filenames / keys
model = joblib.load(MODEL_DIR / "logisticregression.joblib")
with open(MODEL_DIR / "logreg_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

feature = meta["features"]
thresh  = meta.get("threshold", 0.5)

# pull scaler stats if present (for nicer randoms) 
scaler = None
if isinstance(model, Pipeline):
    for _, step in model.steps:
        if isinstance(step, StandardScaler):
            scaler = step
            break

EPS = 1e-6  # avoid zeros

def randomize():
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        x = np.random.normal(scaler.mean_, scaler.scale_)   # around training means
        x = np.clip(x, EPS, None)                           # keep positive-ish
    else:
        x = np.random.uniform(0.0, 1.0, size=len(feature))  # simple fallback
        x = np.clip(x, EPS, None)
    vals = [float(v) for v in x]
    row_json = json.dumps({feature[i]: vals[i] for i in range(len(feature))}, indent=2)
    return [*vals, row_json]

def clear_inputs():
    return [None]*len(feature) + ["{}"]

def predict(*vals):
    *feat_vals, threshold = vals
    row = {feature[i]: (EPS if feat_vals[i] is None else float(feat_vals[i]))
           for i in range(len(feature))}
    df = pd.DataFrame([row], columns=feature)
    proba_m = float(model.predict_proba(df)[:, 1][0])         # P(malignant)
    proba_b = 1 - proba_m # Probability for benign
    label = "malignant" if proba_m >= float(threshold) else "benign"
    pred = {"label": label, 
            "prob_malignant": round(proba_m, 4), 
            "prob_benign": round(proba_b,  4),
            "threshold_used": round(float(threshold), 3)
        }
    return pred, json.dumps(row, indent=2)

with gr.Blocks(title="Breast Cancer Classifier (demo)") as demo:
    gr.HTML("<h1>Breast Cancer detection demo")

    inputs = [gr.Number(label=f) for f in feature]
    threshold = gr.Slider(0, 1, value=thresh, step=0.01, label="Decision threshold")

    with gr.Row():
        btn_rand  = gr.Button("Randomize")
        btn_clear = gr.Button("Clear")
        btn_pred  = gr.Button("Predict")

    out_pred = gr.JSON(label="Prediction")
    out_row  = gr.Code(label="Current features (JSON)", language="json")

    btn_rand.click(fn=randomize, outputs=inputs + [out_row])
    btn_clear.click(fn=clear_inputs, outputs=inputs + [out_row])
    btn_pred.click(fn=predict, inputs=inputs + [threshold], outputs=[out_pred, out_row])

if __name__ == "__main__":
    
    demo.launch()  