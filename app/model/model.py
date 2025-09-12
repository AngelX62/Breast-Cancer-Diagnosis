from typing import Any, Dict, List
import joblib, json, os
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))

META_PATH = MODEL_DIR / "logreg_meta.json"
MODEL_PATH = MODEL_DIR / "logisticregression.joblib"
META: Dict[str, Any] = json.loads(META_PATH.read_text())
PIPE = joblib.load(MODEL_PATH)

FEATURES: List[str] = META["features"]
THRESH: float = float(META.get("threshold", 0.50)) # Set threshold
__version__ = META.get("model_version", "")

def predict_pipeline(record: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([record], columns=FEATURES)
    proba = float(PIPE.predict_proba(df)[:, 1][0])
    return {"proba": proba, "prediction": int(proba >= THRESH), "threshold": THRESH}