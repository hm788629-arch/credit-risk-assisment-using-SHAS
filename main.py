from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "credit_risk_model_local.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold_local.pkl"

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure Render starts with the exact model files bundled in the repository.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH.name}")

    ml_model["model"] = joblib.load(MODEL_PATH)
    ml_model["threshold"] = float(joblib.load(THRESHOLD_PATH))
    yield
    ml_model.clear()


app = FastAPI(title="CrediScope Credit Risk API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "model_loaded": "model" in ml_model}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/predict")
def predict(data: LoanApplication):
    input_data = pd.DataFrame([data.model_dump()])
    probability = float(ml_model["model"].predict_proba(input_data)[0, 1])
    threshold = float(ml_model["threshold"])
    prediction = int(probability >= threshold)
    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "best_threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }
