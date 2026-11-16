"""AutoValue AI - Model Training Pipeline
======================================
Trains the Ridge Regression model with ColumnTransformer on automotive data
and exports the serialized artifacts (preprocessor, model, frequency map).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_export(output_dir="models", n_samples=15000):
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- [AutoValue AI] Generating Training Dataset ({n_samples:,} records) ---")

    np.random.seed(42)
    manufacturers = [
        'ford', 'chevrolet', 'toyota', 'honda', 'nissan', 'jeep', 'ram', 'gmc', 
        'dodge', 'bmw', 'hyundai', 'subaru', 'mercedes-benz', 'volkswagen', 
        'chrysler', 'kia', 'cadillac', 'buick', 'lexus', 'mazda', 'audi', 
        'acura', 'infiniti', 'lincoln', 'volvo', 'mitsubishi', 'tesla', 'porsche', 'ferrari'
    ]
    mfr_probs = np.array([0.16, 0.14, 0.11, 0.09, 0.07, 0.05, 0.05, 0.04, 0.04, 
                          0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 
                          0.01, 0.01, 0.01, 0.008, 0.007, 0.005, 0.005, 0.004, 0.005, 0.003, 0.003])
    mfr_probs /= mfr_probs.sum()

    mfr_choice = np.random.choice(manufacturers, size=n_samples, p=mfr_probs)
    years = np.random.choice(np.arange(1995, 2024), size=n_samples, p=np.linspace(0.01, 0.06, 29) / np.linspace(0.01, 0.06, 29).sum())

    vehicle_ages = 2024 - years
    odometer_base = vehicle_ages * np.random.normal(12000, 2800, size=n_samples) + np.random.uniform(500, 10000, size=n_samples)
    odometers = np.clip(odometer_base, 500, 280000).astype(int)

    conditions = np.random.choice(['new', 'like new', 'excellent', 'good', 'fair', 'salvage'], size=n_samples, p=[0.02, 0.15, 0.45, 0.28, 0.08, 0.02])
    cylinders = np.random.choice(['4 cylinders', '6 cylinders', '8 cylinders', '10 cylinders'], size=n_samples, p=[0.45, 0.35, 0.18, 0.02])
    fuels = np.random.choice(['gas', 'diesel', 'hybrid', 'electric'], size=n_samples, p=[0.88, 0.07, 0.03, 0.02])
    title_statuses = np.random.choice(['clean', 'rebuilt', 'salvage', 'lien', 'parts only'], size=n_samples, p=[0.92, 0.04, 0.02, 0.015, 0.005])
    transmissions = np.random.choice(['automatic', 'manual'], size=n_samples, p=[0.93, 0.07])
    drives = np.random.choice(['4wd', 'fwd', 'rwd'], size=n_samples, p=[0.40, 0.40, 0.20])
    types = np.random.choice(['sedan', 'SUV', 'pickup', 'truck', 'coupe', 'hatchback', 'convertible', 'van'], size=n_samples, p=[0.30, 0.30, 0.15, 0.08, 0.06, 0.05, 0.03, 0.03])
    states = np.random.choice(['ca', 'tx', 'fl', 'ny', 'oh', 'pa', 'il', 'nc', 'mi', 'ga', 'wa', 'az', 'co', 'va', 'or'], size=n_samples)
    lats = np.random.uniform(25.0, 48.0, size=n_samples)
    longs = np.random.uniform(-124.0, -70.0, size=n_samples)

    brand_premiums = {
        'ferrari': 3.5, 'porsche': 2.2, 'tesla': 1.6, 'lexus': 1.4, 'bmw': 1.35, 'mercedes-benz': 1.35,
        'audi': 1.3, 'lincoln': 1.15, 'cadillac': 1.15, 'acura': 1.15, 'ram': 1.25, 'gmc': 1.15,
        'toyota': 1.2, 'subaru': 1.1, 'jeep': 1.1, 'honda': 1.15, 'ford': 1.0, 'chevrolet': 1.0,
        'volkswagen': 0.95, 'mazda': 0.95, 'nissan': 0.9, 'hyundai': 0.9, 'kia': 0.9,
        'buick': 0.85, 'chrysler': 0.85, 'dodge': 0.9, 'mitsubishi': 0.8, 'volvo': 1.1, 'infiniti': 1.05
    }

    condition_mult = {'new': 1.25, 'like new': 1.15, 'excellent': 1.0, 'good': 0.85, 'fair': 0.60, 'salvage': 0.35}
    title_mult = {'clean': 1.0, 'lien': 0.95, 'rebuilt': 0.75, 'salvage': 0.45, 'parts only': 0.20}
    fuel_mult = {'diesel': 1.20, 'electric': 1.15, 'hybrid': 1.05, 'gas': 1.0}
    drive_mult = {'4wd': 1.10, 'rwd': 1.0, 'fwd': 0.95}

    base_price = 33000.0
    depr_age = np.exp(-0.065 * vehicle_ages)
    depr_odo = np.exp(-0.0000045 * odometers)

    mfr_factors = np.array([brand_premiums.get(m, 1.0) for m in mfr_choice])
    cond_factors = np.array([condition_mult.get(c, 1.0) for c in conditions])
    title_factors = np.array([title_mult.get(t, 1.0) for t in title_statuses])
    fuel_factors = np.array([fuel_mult.get(f, 1.0) for f in fuels])
    drive_factors = np.array([drive_mult.get(d, 1.0) for d in drives])

    noise = np.random.normal(1.0, 0.10, size=n_samples)
    calculated_prices = base_price * mfr_factors * depr_age * depr_odo * cond_factors * title_factors * fuel_factors * drive_factors * noise
    prices = np.clip(calculated_prices, 600, 98000).round(2)

    df = pd.DataFrame({
        'price': prices,
        'year': years,
        'manufacturer': mfr_choice,
        'condition': conditions,
        'cylinders': cylinders,
        'fuel': fuels,
        'odometer': odometers,
        'title_status': title_statuses,
        'transmission': transmissions,
        'drive': drives,
        'type': types,
        'state': states,
        'lat': lats,
        'long': longs
    })

    # Domain Feature Engineering
    df['vehicle_age'] = 2024 - df['year']
    df['mileage_per_year'] = np.where(df['vehicle_age'] > 0, df['odometer'] / df['vehicle_age'], df['odometer'])
    df['age_group'] = pd.cut(df['vehicle_age'], bins=[-1, 3, 10, 25, 100], labels=['new', 'modern', 'older', 'classic'])

    mfr_counts = df['manufacturer'].value_counts().to_dict()
    df['manufacturer_popularity'] = df['manufacturer'].map(mfr_counts)

    # Train / Test split
    y_raw = df['price']
    y_log = np.log1p(df['price'])
    X = df.drop(columns=['price'])

    numerical_cols = ['year', 'odometer', 'lat', 'long', 'vehicle_age', 'mileage_per_year', 'manufacturer_popularity']
    categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'state', 'age_group']

    X = X[numerical_cols + categorical_cols]
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    print("--- Fitting Preprocessing Pipeline ---")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("--- Training Ridge Regression Model ---")
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_processed, y_train)

    # Evaluation
    pred_log = ridge_model.predict(X_test_processed)
    pred_raw = np.expm1(pred_log)

    mae = mean_absolute_error(y_test_raw, pred_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, pred_raw))
    r2 = r2_score(y_test_raw, pred_raw)

    print(f"Model Performance Metrics:")
    print(f"  • Test MAE:  ${mae:,.2f}")
    print(f"  • Test RMSE: ${rmse:,.2f}")
    print(f"  • Test R²:   {r2:.4f}")

    # Export assets
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    joblib.dump(ridge_model, os.path.join(output_dir, 'ridge_model.joblib'))
    joblib.dump(mfr_counts, os.path.join(output_dir, 'mfr_counts.joblib'))

    # Root mirror for backward compatibility
    joblib.dump(preprocessor, 'preprocessor.joblib')
    joblib.dump(ridge_model, 'ridge_model.joblib')
    joblib.dump(mfr_counts, 'mfr_counts.joblib')

    print(f"Assets exported successfully to '{output_dir}/' and root directory.")
    return {"mae": mae, "rmse": rmse, "r2": r2}

if __name__ == '__main__':
    train_and_export()
