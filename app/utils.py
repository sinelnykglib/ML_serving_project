import pandas as pd
import numpy as np
import re

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

categorical_cols = ["neighbourhood group", "room type"]
binary_cols = ["instant_bookable", "host_identity_verified"]

def preprocess_input_with_indices(data) -> tuple[pd.DataFrame, pd.Index]:
    """
    Transform input data into DataFrame suitable for model prediction.
    Повертає оброблений DataFrame та індекси валідних рядків.
    """
    # ---- 1. Normalize input type ----
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        original_indices = df.index.copy()
    else:
        df = pd.DataFrame([data])
        original_indices = df.index.copy()
    
    print("Початкові колонки:", df.columns.tolist())
    print("Початкова форма:", df.shape)
    
    # ---- 2. Clear the columns ----
    cols_to_drop = ['id', 'license', 'country', 'country code', 'NAME', 
                    'host name', 'house_rules', 'neighbourhood', 'host id', 'cancellation_policy']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
    
    # ---- 3. Rename columns to match training ----
    column_mapping = {
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
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Save indexes
    current_indices = df.index.copy()
    
    # ---- 4. Очищення price (якщо є) ----
    if 'price' in df.columns:
        # Delet $ and ,
        df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        # Drop prise <= 0
        price_mask = df['price'] > 0
        df = df[price_mask]
        current_indices = current_indices[price_mask]
        print(f"Після очищення price: {df.shape}")
    
    # ---- 5. Clear service fee ----
    if 'service fee' in df.columns:
        df['service fee'] = df['service fee'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['service fee'] = pd.to_numeric(df['service fee'], errors='coerce')
        service_fee_mask = df['service fee'].notna()
        df = df[service_fee_mask]
        current_indices = current_indices[service_fee_mask]
        print(f"Після очищення service fee: {df.shape}")
    else:
        df['service fee'] = 0
    
    # ---- 6. Construction year ----
    if 'Construction year' in df.columns:
        df['Construction year'] = pd.to_numeric(df['Construction year'], errors='coerce')
        median_year = df['Construction year'].median()
        if pd.isna(median_year):
            median_year = 2000
        df['Construction year'] = df['Construction year'].fillna(median_year)
        df['Construction year'] = df['Construction year'].astype(int)
    else:
        df['Construction year'] = 2000
    
    # ---- 7. Drop last review ----
    if 'last review' in df.columns:
        df = df.drop(columns=['last review'])
    
    # ---- 8. Clear reviews per month ----
    if 'reviews per month' in df.columns:
        df['reviews per month'] = pd.to_numeric(df['reviews per month'], errors='coerce')
        df['reviews per month'] = df['reviews per month'].fillna(0)
    else:
        df['reviews per month'] = 0
    
    # ---- 9. Clear review rate number ----
    if 'review rate number' in df.columns:
        df['review rate number'] = pd.to_numeric(df['review rate number'], errors='coerce')
        mean_rate = df['review rate number'].mean()
        if pd.isna(mean_rate):
            mean_rate = 0
        df['review rate number'] = df['review rate number'].fillna(mean_rate)
    else:
        df['review rate number'] = 0
    
    # ---- 10. Clear other numeric columns ----
    other_numeric = ["minimum nights", "number of reviews", 
                     "calculated host listings count", "availability 365"]
    
    for col in other_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = 0
    
    # ---- 11. Clear coordinates ----
    for col in ['lat', 'long']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Delete rows with invalid coordinates
    coords_mask = df['lat'].notna() & df['long'].notna()
    df = df[coords_mask]
    current_indices = current_indices[coords_mask]
    print(f"Після обробки координат: {df.shape}")
    
    # ---- 12. Binary columns ----
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            binary_mask = df[col].notna()
            df = df[binary_mask]
            current_indices = current_indices[binary_mask]
            df[col] = df[col].str.lower().str.strip()
            df[col] = df[col].map({'true': 't', 'false': 'f', '1': 't', '0': 'f', 't': 't', 'f': 'f'})
            df[col] = df[col].fillna('f')
        else:
            df[col] = 'f'
    
    print(f"Після обробки binary колонок: {df.shape}")
    
    # ---- 13. Categorical columns ----
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            cat_mask = df[col].notna()
            df = df[cat_mask]
            current_indices = current_indices[cat_mask]
            df[col] = df[col].str.strip()
            df[col] = df[col].fillna('missing')
        else:
            df[col] = 'missing'
    
    print(f" categorical columns: {df.shape}")
    
    # ---- 14. Drop price column if exists ----
    df = df.drop(columns=['price'], errors='ignore')
    
    # ---- 15. Ensure all required columns are in correct order ----
    required_cols = categorical_cols + binary_cols + numeric_cols
    
    for col in required_cols:
        if col not in df.columns:
            if col in numeric_cols:
                df[col] = 0
            else:
                df[col] = 'missing'
    
    print("Final columns:", required_cols)
    print("Final shape:", df.shape)
    print("Number of NaN in final data:", df[required_cols].isna().sum().sum())
    
    return df[required_cols], current_indices