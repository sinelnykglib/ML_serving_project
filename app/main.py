# FastAPI app
from fastapi import FastAPI
from app.model import AirbnbPriceModel
from app.schemas import AirbnbInput
from app.utils import preprocess_input

app = FastAPI(title="Airbnb Price Prediction API")

# Initialize model
model_path = "mlruns/1/models/m-726779d47628485d8aac9447b995b88a/artifacts"  # way to the trained model
model = AirbnbPriceModel(model_path)

@app.post("/predict/")
def predict_price(item: AirbnbInput):
    # Preprocess input data
    df = preprocess_input(item.dict())
    prediction = model.predict(df)
    return {"predicted_price": float(prediction[0])}
