import pandas as pd
import numpy as np


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simpl function for usinf features of Airbnb price prediction.
    Cleaning data, filling missing values, encoding categorical features.
    """
    df = df.copy()
    
    df = df.drop(columns=["id"])
    df = df.drop(columns=["license"])

    df['price'] = (
    df['price'].astype(str)
              .str.replace(r'[$,]', '', regex=True)
              .astype(float)
                         )
    df = df[df['price'] > 0]

    df["service fee"] = (
    df["service fee"]
    .astype(str)
    .str.replace(r"[$,]", "", regex=True)
        )
    df["service fee"] = pd.to_numeric(df["service fee"], errors="coerce")
    df = df.dropna(subset=["service fee"])

    median_year = df["Construction year"].median()
    df["Construction year"] = df["Construction year"].fillna(median_year)
    df["Construction year"] = df["Construction year"].astype(int)
    

    df["last review"] = pd.to_datetime(
    df["last review"],
    format="%m/%d/%Y",
    errors="coerce"   # invalid or NaN -> NaT
    )

    df = df.drop(columns=["last review"])
    df["reviews per month"] = df["reviews per month"].fillna(0)
    df["review rate number"] = df["review rate number"].fillna(
    df["review rate number"].mean()
    )

    for col in ["minimum nights",
            "number of reviews",
            "calculated host listings count",
            "availability 365"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)


    for col in ["minimum nights",
            "number of reviews",
            "calculated host listings count",
            "availability 365"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)




    # Clear data from useles columns
    df = df.drop(columns=["country"])
    df = df.drop(columns=["country code"])
    df = df.drop(columns=["NAME"])
    df = df.drop(columns=["host name"])
    df = df.dropna(subset=["instant_bookable"])
    df = df.dropna(subset=["host_identity_verified"])
    df = df.dropna(subset=["neighbourhood group"])
    df = df.dropna(subset=["neighbourhood"])
    df = df.dropna(subset=["lat"])
    df = df.dropna(subset=["long"])
    df = df.drop(columns=["house_rules"])   
    
    
    return df
