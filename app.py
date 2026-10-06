import time
import pandas as pd
import numpy as np
import streamlit as st
from edge_inference import EdgeInferenceEngine, FEATURE_COLS

st.set_page_config(
    page_title="Edge AI Machine Health Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Edge-Ready Machine Anomaly Detection System")
st.caption("Simulated On-Device Machine Health Monitoring | Zero Cloud Latency")

@st.cache_resource
def load_engine():
    return EdgeInferenceEngine()

try:
    engine = load_engine()
    st.sidebar.success("Edge Model & Scaler Active (.pkl)")
except Exception as e:
    st.error(f"Error loading binaries: {e}. Execute `data_prep.py` and `train.py` first.")
    st.stop()

# Metric Card Placeholders
col1, col2, col3, col4 = st.columns(4)
metric_temp = col1.empty()
metric_speed = col2.empty()
metric_torque = col3.empty()
metric_status = col4.empty()

chart_container = st.container()

# Sidebar Interactive Controls
st.sidebar.header("Telemetry Simulation Controls")
sim_speed = st.sidebar.slider("Stream Delay (seconds)", 0.1, 2.0, 0.4)
inject_fault = st.sidebar.checkbox("Inject Simulated Component Failure")
run_stream = st.sidebar.button("Start Telemetry Stream")

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=FEATURE_COLS + ["Anomaly Score", "Status"])

if run_stream:
    for i in range(40):
        if inject_fault and i > 10:
            air_temp = np.random.normal(322.0, 3.0)
            proc_temp = np.random.normal(338.0, 4.0)
            rot_speed = np.random.normal(2700.0, 120.0)
            torque = np.random.normal(88.0, 7.0)
            tool_wear = min(240, 190 + i * 2)
        else:
            air_temp = np.random.normal(300.0, 1.0)
            proc_temp = np.random.normal(310.0, 1.0)
            rot_speed = np.random.normal(1500.0, 25.0)
            torque = np.random.normal(40.0, 2.5)
            tool_wear = min(240, i * 2)

        sample = {
            'Air temperature [K]': round(air_temp, 2),
            'Process temperature [K]': round(proc_temp, 2),
            'Rotational speed [rpm]': round(rot_speed, 1),
            'Torque [Nm]': round(torque, 2),
            'Tool wear [min]': tool_wear
        }

        res = engine.predict_sample(sample)

        # Update Live Dashboard Cards
        metric_temp.metric("Air Temperature", f"{sample['Air temperature [K]']} K")
        metric_speed.metric("Rotational Speed", f"{sample['Rotational speed [rpm]']} RPM")
        metric_torque.metric("Torque Output", f"{sample['Torque [Nm]']} Nm")
        
        if res["is_anomaly"]:
            metric_status.error(" WARNING: ANOMALY")
        else:
            metric_status.success(" SYSTEM HEALTHY")

        new_row = sample.copy()
        new_row["Anomaly Score"] = res["score"]
        new_row["Status"] = res["status"]
        st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)

        with chart_container:
            st.markdown("### Real-Time Telemetry & Anomaly Boundary Trajectory")
            c1, c2 = st.columns(2)
            with c1:
                st.line_chart(st.session_state.history[['Rotational speed [rpm]', 'Torque [Nm]']].tail(25))
            with c2:
                st.line_chart(st.session_state.history[['Anomaly Score']].tail(25))

        time.sleep(sim_speed)