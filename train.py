import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

def train_isolation_forest(data_path: str = "processed_data.csv", model_path: str = "anomaly_model.pkl"):
    """
    Trains an Isolation Forest from scratch on preprocessed sensor values
    and outputs a lightweight serialized binary for constrained hardware.
    """
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col != 'target']
    
    X = df[feature_cols].values
    y_true = df['target'].values if 'target' in df.columns else None

    contamination = float(np.mean(y_true)) if y_true is not None and np.mean(y_true) > 0 else 0.04

    print(f"[Training] Training Isolation Forest (Contamination baseline: {contamination:.3f})...")
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)

    # Predict anomalies (-1: Anomaly, 1: Normal)
    preds_raw = model.predict(X)
    preds_binary = np.where(preds_raw == -1, 1, 0) # Convert to 1 = anomaly, 0 = normal

    if y_true is not None:
        p = precision_score(y_true, preds_binary, zero_division=0)
        r = recall_score(y_true, preds_binary, zero_division=0)
        f1 = f1_score(y_true, preds_binary, zero_division=0)
        
        print("\n--- Training Performance Metrics ---")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1-Score:  {f1:.4f}\n")
        print("Classification Report:\n", classification_report(y_true, preds_binary, target_names=['Normal', 'Anomaly']))

    # Save lightweight model artifact
    joblib.dump(model, model_path)
    model_size_kb = os.path.getsize(model_path) / 1024
    print(f"[Training] Model saved to '{model_path}' | File Size: {model_size_kb:.2f} KB")

if __name__ == "__main__":
    train_isolation_forest()