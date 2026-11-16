"""AutoValue AI - Production Inference Module
===========================================
Reusable Python inference script to estimate fair market value of used vehicles.
Loads the serialized Preprocessing Pipeline, Ridge Regression model, and Manufacturer frequency map.
"""

import os
import joblib
import numpy as np
import pandas as pd

def predict_price(vehicle_specs, model_dir=None):
    """
    Predicts the dollar price of a vehicle given raw specifications.

    Parameters:
    -----------
    vehicle_specs : dict
        Dictionary of raw vehicle features.
        Keys can include: 'year', 'odometer', 'manufacturer', 'condition',
        'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'state', 'lat', 'long'
    model_dir : str, optional
        Directory path where the serialized joblib artifacts reside.
        Defaults to checking 'models/' then '.'.

    Returns:
    --------
    float: Estimated fair market price in USD.
    """
    if model_dir is None:
        model_dir = 'models' if os.path.exists(os.path.join('models', 'preprocessor.joblib')) else '.'

    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    model_path = os.path.join(model_dir, 'ridge_model.joblib')
    mfr_counts_path = os.path.join(model_dir, 'mfr_counts.joblib')

    # Verify assets exist
    for path in [preprocessor_path, model_path, mfr_counts_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}. Run the training notebook first.")

    # 1. Load exported pipeline and model assets
    preprocessor = joblib.load(preprocessor_path)
    ridge_model = joblib.load(model_path)
    mfr_counts = joblib.load(mfr_counts_path)

    # 2. Setup dataframe and fill missing defaults defensively
    input_df = pd.DataFrame([vehicle_specs])
    default_vals = {
        'year': 2015,
        'odometer': 100000,
        'lat': 40.0,
        'long': -95.0,
        'manufacturer': 'ford',
        'condition': 'excellent',
        'cylinders': '6 cylinders',
        'fuel': 'gas',
        'title_status': 'clean',
        'transmission': 'automatic',
        'drive': '4wd',
        'type': 'sedan',
        'state': 'tx'
    }
    for col, val in default_vals.items():
        if col not in input_df.columns:
            input_df[col] = val

    # 3. Apply identical domain feature engineering
    input_df['vehicle_age'] = 2023 - input_df['year']
    input_df['mileage_per_year'] = np.where(
        input_df['vehicle_age'] > 0,
        input_df['odometer'] / input_df['vehicle_age'],
        input_df['odometer']
    )
    input_df['age_group'] = pd.cut(
        input_df['vehicle_age'],
        bins=[-1, 3, 10, 25, 100],
        labels=['new', 'modern', 'older', 'classic']
    )
    input_df['manufacturer_popularity'] = input_df['manufacturer'].map(mfr_counts).fillna(100)

    # 4. Filter features to match trained schema
    numerical_cols = ['year', 'odometer', 'lat', 'long', 'vehicle_age', 'mileage_per_year', 'manufacturer_popularity']
    categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'state', 'age_group']
    input_features = input_df[numerical_cols + categorical_cols]

    # 5. Transform & Predict
    processed = preprocessor.transform(input_features)
    predicted_log = ridge_model.predict(processed)[0]
    return round(float(np.expm1(predicted_log)), 2)

if __name__ == '__main__':
    # CLI Validation Example
    sample_vehicle = {
        'year': 2018,
        'odometer': 50000,
        'manufacturer': 'ford',
        'condition': 'excellent',
        'fuel': 'gas',
        'transmission': 'automatic',
        'type': 'SUV'
    }
    print("AutoValue AI Inference Module Ready.")
    print("Sample vehicle payload:", sample_vehicle)
    try:
        est = predict_price(sample_vehicle)
        print(f"Fair Market Valuation: ${est:,.2f}")
    except FileNotFoundError as e:
        print(f"Notice: {e}")
