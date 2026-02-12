import pandas as pd

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Trasnform input data into DataFrame suitable for model prediction.
    """
    df = pd.DataFrame([data])
    
    # Ranema colmns to match training data
    df.rename(columns={
        "room_type": "room type",
        "neighbourhood_group": "neighbourhood group",
        "construction_year": "Construction year",
        # інші якщо треба
    }, inplace=True)

    # Covert columns to appropriate types
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df
