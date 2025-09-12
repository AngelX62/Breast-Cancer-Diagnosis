from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model # Validate incoming request data
from pydantic.config import ConfigDict
from app.model.model import predict_pipeline, __version__ as model_version, FEATURES


# Make our API only accept exact training features
class FeatureBase(BaseModel):
    model_config = ConfigDict(extra="forbid") # Forbids unknown keys

FeatureModel = create_model(
    "FeatureModel",
    __base__ = FeatureBase,
    **{feat: (float, Field(...)) for feat in FEATURES}
)


class PredictionOut(BaseModel):
    proba: float
    prediction: int
    threshold: float

app = FastAPI(title="Breast Cancer Screening API")

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: FeatureModel) -> PredictionOut: # type: ignore 
    return predict_pipeline(payload.model_dump())