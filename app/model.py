import mlflow.sklearn
import pandas as pd

class AirbnbPriceModel:
    def __init__(self, model_path: str):
        # Loading the modes via MLflow
        self.model = mlflow.sklearn.load_model(model_path)

    def predict(self, X: pd.DataFrame):
        # Processing if need to use (OneHot / Imputer)
        return self.model.predict(X)
