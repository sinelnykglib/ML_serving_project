# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import io
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

from training.features import preprocess_features
from app.schemas import AirbnbInput
from app.utils import preprocess_input

app = FastAPI(title="Airbnb Price Prediction API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

# ---------------- Frontend ----------------
@app.get("/")
def serve_frontend():
    return FileResponse(WEB_DIR / "screen_1.html")

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# ---------------- Load model once ----------------
# Модель збережена разом з pipeline через joblib
model_path = BASE_DIR / "mlruns/1/models/m-726779d47628485d8aac9447b995b88a/artifacts/model.pkl"
model = joblib.load(model_path)

# ==================================================
#                 SINGLE PREDICTION
# ==================================================
@app.post("/predict/")
def predict_price(item: AirbnbInput):
    # конвертуємо Pydantic у DataFrame
    df = pd.DataFrame([item.dict()])
    prediction = model.predict(df)
    return {"predicted_price": float(prediction[0])}

# ==================================================
#                 BATCH PREDICTION
# ==================================================
@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), low_memory=False)

    y_true = None

    if "price" in df.columns:
        y_true = (
            df["price"]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .str.strip()
        )
        y_true = pd.to_numeric(y_true, errors="coerce")
        # Видаляємо NaN
        valid_mask = y_true.notna()
        df = df[valid_mask]
        y_true = y_true[valid_mask]

    # --- Приводимо CSV до фіч, які чекає модель
    df_processed = preprocess_input(df)

    # --- Передаємо у pipeline + модель
    predictions = model.predict(df_processed)

    mse, rmse, actual_prices = None, None, None

    if y_true is not None and len(y_true) > 0:
        mse = float(mean_squared_error(y_true, predictions))
        rmse = float(np.sqrt(mse))
        actual_prices = y_true.tolist()

    return {
        "predictions": predictions.tolist(),
        "actual_prices": actual_prices,
        "mse": mse,
        "rmse": rmse
    }
