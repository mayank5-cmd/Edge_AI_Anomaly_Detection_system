import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

def load_and_preprocess(data_path: str = "ai4i2020.csv", scaler_path: str = "scaler.pkl"):
    """
    Cleans raw CSV sensor data, handles missing values,
    and fits/saves a StandardScaler for edge deployment consistency.
    """
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # Extract continuous numerical features
    X = df[FEATURE_COLS].copy()
    X.fillna(X.median(), inplace=True)
    
    # Extract ground truth target if present in dataset
    y = df['Machine failure'] if 'Machine failure' in df.columns else None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for local edge pipeline
    joblib.dump(scaler, scaler_path)
    print(f"[Data Prep] Scaler saved successfully to '{scaler_path}'.")
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    if y is not None:
        X_scaled_df['target'] = y.values
        
    return X_scaled_df, scaler

if __name__ == "__main__":
    try:
        df_processed, _ = load_and_preprocess("ai4i2020.csv")
        df_processed.to_csv("processed_data.csv", index=False)
        print("[Data Prep] Processing complete. Output saved to 'processed_data.csv'.")
    except FileNotFoundError:
        print("[Data Prep] 'ai4i2020.csv' not found. Generating sample synthetic telemetry dataset...")
        np.random.seed(42)
        n_samples = 1000
        synthetic_df = pd.DataFrame({
            'Air temperature [K]': np.random.normal(300, 2, n_samples),
            'Process temperature [K]': np.random.normal(310, 2, n_samples),
            'Rotational speed [rpm]': np.random.normal(1500, 150, n_samples),
            'Torque [Nm]': np.random.normal(40, 10, n_samples),
            'Tool wear [min]': np.random.randint(0, 240, n_samples),
            'Machine failure': np.random.choice([0, 1], size=n_samples, p=[0.96, 0.04])
        })
        synthetic_df.to_csv("ai4i2020.csv", index=False)
        df_processed, _ = load_and_preprocess("ai4i2020.csv")
        df_processed.to_csv("processed_data.csv", index=False)
        print("[Data Prep] Synthetic dataset generated and preprocessed.")