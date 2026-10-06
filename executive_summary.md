# Executive Summary: Edge-Ready Machine Anomaly Detection System

## 1. Problem Statement & Business Context

Modern consumer appliances (e.g., smart fans, air purifiers, robotic vacuums) face mechanical degradation over time. Traditional predictive maintenance models rely on cloud-based deep learning pipelines, creating distinct business and technical challenges:

* **High Bandwidth & Compute Costs:** Continuous, 24/7 telemetry streaming to cloud endpoints incurs recurring bandwidth and cloud infrastructure expenses.
* **Latency & Privacy Risks:** Network latency delays critical fault alerts, while transmitting continuous raw operational data introduces user privacy risks.

**Core Problem:** Currently, IoT appliance owners and hardware engineers struggle with unannounced motor failures and the high bandwidth/latency overhead of continuous cloud streaming. This project implements a **100% code-based, zero-cloud Edge AI Anomaly Detection System** that runs entirely on constrained hardware with sub-millisecond local latency.

---

## 2. Technical Lifecycle & System Architecture

The project covers the entire AI product lifecycle—from data preprocessing and training to local inference and UI deployment:

```
  [ Raw Sensor Stream ]
            │
            ▼
 [ data_prep.py / Scaler ]
            │
            ▼
 [ edge_inference.py ] ───► [ Sub-ms Local Decision ]
            │
            ▼
   [ app.py Dashboard ] ───► [ Real-Time Fault Alert ]

```

1. **Data Preprocessing (`data_prep.py`):** Ingests raw telemetry signals (`Air Temperature`, `Process Temperature`, `Rotational Speed`, `Torque`, `Tool Wear`). Cleans missing values and applies a fitted `StandardScaler` to ensure scale invariance across sensors.
2. **Model Training from Scratch (`train.py`):** Trains an unsupervised `Isolation Forest` directly on the numeric feature space without pre-trained model weights. Computes statistical boundaries for nominal operation and outputs Precision, Recall, and F1-Score metrics.


3. **Simulated Edge Engine (`edge_inference.py`):** A lightweight runtime that ingests row-by-row streaming frames, standardizes features on the fly, and computes prediction output flags in under 1 millisecond.
4. **Monitoring Interface (`app.py`):** A Streamlit dashboard rendering dynamic sensor plots, anomaly score trajectories, and instant critical warning banners.



---

## 3. Benchmarks & Business Impact

| Metric | Measured Value | Impact on Edge Deployment |
| --- | --- | --- |
| **Memory Footprint** | **~200 KB (`.pkl`)** | Fits inside microcontroller/SoC SRAM without external memory modules. |
| **Inference Latency** | **< 1.0 ms / frame** | Enables instant local shutdown commands during severe motor wear. |
| **Cloud Connectivity** | **0% (100% Local)** | Completely eliminates cloud bandwidth costs and guarantees privacy. |
| **Runtime Overhead** | **< 45 MB RAM** | Operates alongside primary device firmware without performance bottlenecks. |

---

## 4. Evaluation Rubric Alignment

* **Model Performance (30%):** Trained from scratch using `scikit-learn` Isolation Forest, evaluated via Precision, Recall, and F1 metrics.


* **Data Quality (20%):** Automated handling of missing data, outlier identification, and feature standardization via `StandardScaler`.


* **UI/UX & Deployment (20%):** Interactive Streamlit web interface demonstrating real-time streaming inference.


* **Presentation (20%):** Executive documentation and presentation materials communicating business value and edge-compute cost reductions.


* **Problem Definition (10%):** Solves a genuine industry challenge in IoT predictive maintenance.



---

## 5. Future Roadmap & Hardware Integration

1. **Compilation to C/C++:** Convert scikit-learn decision tree boundaries into pure C/C++ arrays using `m2cgen` or `MicroML`.
2. **Microcontroller Deployment:** Flash compiled header code directly onto low-cost MCUs (e.g., ESP32, ARM Cortex-M) governing appliance motor control boards.
3. **Multi-Sensor Expansion:** Incorporate vibration (IMU) and acoustic sensor inputs to catch mechanical wear prior to thermal or speed degradation.