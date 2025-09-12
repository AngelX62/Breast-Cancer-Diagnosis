from typing import Any, Dict, List
import joblib, json
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")
META = json.loads(MODEL_DIR.joinpath("logreg_meta.json").read_text())
PIPE = joblib.load(MODEL_DIR / "logisticregression.joblib")

FEATURES: List[str] = META["features"]
THRESH: float = float(META.get("threshold", 0.50))
__version__ = META.get("model_version", "")

def predict_pipeline(record: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([record], columns=FEATURES)
    proba = float(PIPE.predict_proba(df)[:, 1][0])
    return {"proba": proba, "pred": int(proba >= THRESH), "threshold": THRESH}