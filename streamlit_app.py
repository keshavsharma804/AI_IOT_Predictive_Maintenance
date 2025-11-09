import streamlit as st
import pandas as pd
import numpy as np
from src.models.hybrid_ensemble import HybridEnsemble

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return HybridEnsemble.load("models/saved_models/hybrid")

model = load_model()

st.title("🛠️ AI-Based Predictive Maintenance Dashboard")
st.write(
    """
This dashboard uses a **Hybrid Ensemble Model** combining:

- 🧠 **Isolation Forest** → Feature-based anomaly detection  
- 🔁 **LSTM Autoencoder** → Vibration pattern reconstruction

to detect **early mechanical failures** before breakdown occurs.
"""
)

# -------------------------------
# File Upload Section
# -------------------------------
st.header("📤 Upload Machine Sensor Data")

uploaded_file = st.file_uploader(
    "Upload CSV containing `vibration_rms` column (or leave empty to use demo data)",
    type=["csv"]
)

# If no upload, load demo dataset (IMPORTANT: Use correct relative path!)
if uploaded_file is None:
    st.info("ℹ️ No file uploaded — using demo dataset.")
    try:
        data = pd.read_csv("./data/synthetic/machine_001_data.csv")
    except:
        st.error("❌ Demo dataset is missing. Please upload a CSV file.")
        st.stop()
else:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded data loaded successfully!")

# Validate required column
if "vibration_rms" not in data.columns:
    st.error("❌ The dataset must contain a column named **vibration_rms**.")
    st.stop()

# -------------------------------
# Run Model Scoring
# -------------------------------
st.subheader("🔍 Running Anomaly Detection...")

lstm_scores = model.score_sequences(data, signal_col="vibration_rms")
if_scores = np.zeros_like(lstm_scores)  # No feature data → neutral baseline
fused = model.combine_scores(if_scores, lstm_scores)
decisions = model.decision(fused, fusion_threshold=0.6)

# -------------------------------
# Metrics Summary
# -------------------------------
col1, col2 = st.columns(2)
col1.metric("✅ Healthy Windows", int((decisions == 0).sum()))
col2.metric("⚠️ Faulty Windows Detected", int((decisions == 1).sum()))

# -------------------------------
# Line Chart Visualization
# -------------------------------
st.subheader("📈 Machine Health Trend (Anomaly Score Over Time)")
st.line_chart(pd.DataFrame({"Anomaly Score": fused}))

# -------------------------------
# Fault Alerts
# -------------------------------
st.subheader("🚨 Fault Detection Summary")
fault_indices = np.where(decisions == 1)[0]

if len(fault_indices) == 0:
    st.success("✅ Machine condition is normal. No fault patterns detected.")
else:
    st.error("⚠️ Potential mechanical degradation detected!")
    st.write(f"Affected data windows: **{fault_indices.tolist()}**")
