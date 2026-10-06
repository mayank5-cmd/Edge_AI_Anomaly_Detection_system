import time
import joblib
import pandas as pd
from data_prep import FEATURE_COLS

class EdgeInferenceEngine:
    def __init__(self, model_path: str = "anomaly_model.pkl", scaler_path: str = "scaler.pkl"):
        """
        Lightweight runtime that loads serialized binaries
        and executes sub-millisecond on-device predictions.
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_sample(self, raw_features: dict) -> dict:
        """
        Process a single sensor streaming frame.
        """
        start_time = time.perf_counter()
        
        input_df = pd.DataFrame([raw_features])[FEATURE_COLS]
        scaled_features = self.scaler.transform(input_df)
        
        # Predict (-1 = Anomaly, 1 = Normal)
        prediction = self.model.predict(scaled_features)[0]
        anomaly_score = float(self.model.decision_function(scaled_features)[0])
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        is_anomaly = bool(prediction == -1)
        
        return {
            "is_anomaly": is_anomaly,
            "status": "CRITICAL ANOMALY" if is_anomaly else "NORMAL",
            "score": round(anomaly_score, 4),
            "latency_ms": round(latency_ms, 3)
        }

if __name__ == "__main__":
    engine = EdgeInferenceEngine()
    
    # Nominal telemetry test
    normal_sample = {
        'Air temperature [K]': 300.1,
        'Process temperature [K]': 310.2,
        'Rotational speed [rpm]': 1500,
        'Torque [Nm]': 40.0,
        'Tool wear [min]': 10
    }
    print("Normal Sample Test:", engine.predict_sample(normal_sample))

    # Severe out-of-bounds motor fault test
    fault_sample = {
        'Air temperature [K]': 328.5,
        'Process temperature [K]': 345.0,
        'Rotational speed [rpm]': 2850,
        'Torque [Nm]': 98.2,
        'Tool wear [min]': 235
    }
    print("Fault Sample Test: ", engine.predict_sample(fault_sample))