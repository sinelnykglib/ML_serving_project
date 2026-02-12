import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from features import preprocess_features
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# === Load data ===
df = pd.read_csv("data/Airbnb_Open_Data.csv")  

# === Preprocess features ===
df = preprocess_features(df)

# === Select target and features ===
target = 'price'
X = df.drop(columns=[target])
y = df[target]

#check
print(df.columns)
print('Delimiter---------------------------------')
print(X.columns.tolist())

# === Pipline for encoding categorical features ===

categorical_cols = [
    "neighbourhood group"
    ]

binary_cols = [
    "instant_bookable",
    "host_identity_verified"
    ]

numeric_cols = [
    "service fee",
    "minimum nights",
    "number of reviews",
    "reviews per month",
    "review rate number",
    "calculated host listings count",
    "availability 365",
    "lat",
    "long",
    "Construction year"
]  

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), binary_cols),
        ("num", "passthrough", numeric_cols),
    ]
)      

print('CHECING NULL VALUES IN NUMERIC COLUMNS---------')
print(X[numeric_cols].isna().sum())

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ]
)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MLflow experiment ===
mlflow.set_experiment("airbnb_price_prediction")

with mlflow.start_run():

    # Train full pipeline
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Log params
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 5)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

    # Log FULL pipeline
    mlflow.sklearn.log_model(model, "model")
