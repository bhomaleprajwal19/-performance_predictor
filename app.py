from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os   # ✅ FIXED

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
models = {
    "lr":  joblib.load(os.path.join(BASE_DIR, "model_lr.pkl")),
    "rf":  joblib.load(os.path.join(BASE_DIR, "model_rf.pkl")),
    "xgb": joblib.load(os.path.join(BASE_DIR, "model_xgb.pkl")),
}

scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

MODEL_META = {
    "lr":  {"name": "Linear Regression",     "r2": 0.9889, "mae": 1.98, "rmse": 2.51},
    "rf":  {"name": "Random Forest (Tuned)", "r2": 0.9901, "mae": 1.77, "rmse": 2.31},
    "xgb": {"name": "XGBoost (Tuned)",       "r2": 0.9910, "mae": 1.65, "rmse": 2.18},
}

@app.get("/")
def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))  # ✅ safer

@app.get("/models")
def get_models():
    return MODEL_META

class InputData(BaseModel):
    Hours_Studied: float
    Previous_Scores: float
    Extracurricular_Activities: int
    Sleep_Hours: float
    Sample_Question_Papers_Practiced: float
    model: str = "xgb"

@app.post("/predict")
def predict(data: InputData):
    if data.model not in models:   
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{data.model}'. Choose from: lr, rf, xgb"
        )

    input_array = np.array([[ 
        data.Hours_Studied,
        data.Previous_Scores,
        data.Extracurricular_Activities,
        data.Sleep_Hours,
        data.Sample_Question_Papers_Practiced,
    ]])

    scaled = scaler.transform(input_array)
    prediction = models[data.model].predict(scaled)[0]  

    return {
        "Predicted Performance Index": round(float(prediction), 2),
        "model_used": MODEL_META[data.model]["name"],
    }