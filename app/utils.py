# app/utils.py

import pandas as pd

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

categorical_cols = ["neighbourhood group"]
binary_cols = ["instant_bookable", "host_identity_verified"]

def preprocess_input(data) -> pd.DataFrame:
    """
    Transform input data into DataFrame suitable for model prediction.
    Works for both:
    - dict (single prediction)
    - DataFrame (batch prediction)
    """
    # ---- 1. Normalize input type ----
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame([data])

    # ---- 2. Rename columns to match training ----
    df.rename(columns={
        "service_fee": "service fee",
        "minimum_nights": "minimum nights",
        "number_of_reviews": "number of reviews",
        "reviews_per_month": "reviews per month",
        "review_rate_number": "review rate number",
        "calculated_host_listings_count": "calculated host listings count",
        "availability_365": "availability 365",
        "construction_year": "Construction year",
        "room_type": "room type",
        "neighbourhood_group": "neighbourhood group"
    }, inplace=True)

    # ---- 3. Numeric columns ----
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ---- 4. Binary columns ----
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().replace({
                "true": "t", "false": "f", "1": "t", "0": "f"
            })

    # ---- 5. Categorical missing ----
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("missing")

    # ---- 6. Drop price column if exists ----
    df = df.drop(columns=["price"], errors="ignore")

    return df
