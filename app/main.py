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
from app.utils import preprocess_input_with_indices

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
# Model saves with pipeline using by joblib
model_path = BASE_DIR / "mlruns/1/models/m-726779d47628485d8aac9447b995b88a/artifacts/model.pkl"
model = joblib.load(model_path)

# ==================================================
#                 SINGLE PREDICTION
# ==================================================
@app.post("/predict/")
def predict_price(item: AirbnbInput):
    try:
        # convert Pydantic into DataFrame
        df = pd.DataFrame([item.dict()])
        # Using the same preprocessing as during training
        df_processed, _ = preprocess_input_with_indices(df)
        prediction = model.predict(df_processed)
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

# ==================================================
#                 BATCH PREDICTION
# ==================================================
@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
        
        print(f"CSV has a {len(df)} rows")
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Saves original prices and indices for metrics calculation
        original_prices = None
        original_indices = None
        if "price" in df.columns:
            # Clear the data
            original_prices = df["price"].copy()
            # Saves original indices before any filtering
            original_indices = df.index.copy()
        
        # --- Clear data for modes ---
        df_processed, valid_indices = preprocess_input_with_indices(df)
        
        # --- Cheking data ---
        if len(df_processed) == 0:
            return {"error": "No valid data after preprocessing"}
        
        print(f"After cleaning: {len(df_processed)} rows")
        
        # --- Send pipeline + model ---
        predictions = model.predict(df_processed)
        
        # --- Caunt metrics ---
        mse, rmse, actual_prices = None, None, None
        
        if original_prices is not None and valid_indices is not None:
            # Use cleaned prices 
            y_true_filtered = original_prices.loc[valid_indices]
            
            # Clear true values like we did during training
            y_true_clean = y_true_filtered.astype(str).str.replace(r'[$,]', '', regex=True)
            y_true_clean = pd.to_numeric(y_true_clean, errors='coerce')
            
            # Create a grid (>0)
            valid_price_mask = (y_true_clean > 0) & (y_true_clean.notna())
            
            if valid_price_mask.any():
                # Use 
                y_true_final = y_true_clean[valid_price_mask]
                predictions_final = predictions[valid_price_mask]
                
                if len(y_true_final) > 0 and len(predictions_final) > 0:
                    mse = float(mean_squared_error(y_true_final, predictions_final))
                    rmse = float(np.sqrt(mse))
                    actual_prices = y_true_final.tolist()
                    predictions = predictions_final.tolist()  # Update predictions
                    
                    print(f"Count metrics {len(y_true_final)} exapmples")
                    print(f"MSE: {mse}, RMSE: {rmse}")
        
        return {
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "actual_prices": actual_prices,
            "mse": mse,
            "rmse": rmse,
            "num_samples": len(predictions) if predictions is not None else 0,
            "num_valid": len(actual_prices) if actual_prices else 0
        }
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}