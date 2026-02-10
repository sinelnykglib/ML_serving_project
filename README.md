# ML_serving_project
## Problem
Serve a trained ML model as a production-ready API.

# Data
Dataset Airbnb Open Data, Link: [Airbnb Open Data](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata/data)

## Solution
Implemented training pipeline with MLflow and inference service using FastAPI.

## How to run
docker build -t ml-serving .
docker run -p 8000:8000 ml-serving

## API
POST /predict
