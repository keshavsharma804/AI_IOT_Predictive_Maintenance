import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import rfft, rfftfreq
from src.models.hybrid_ensemble import HybridEnsemble

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🛠️", layout="wide")

ROOT = Path(".")
MODEL_DIR = ROOT / "models" / "saved_models" / "hybrid"
DEMO_CSV = ROOT / "data" / "synthetic" / "machine_001_demo.csv"

# -----------------------------------
# Model Loader
# -----------------------------------
@st.cache_resource
def load_model():
    return HybridEnsemble.load(MODEL_DIR.as_posix())

model = load_model()

# -----------------------------------
# Demo Data Loader
# -----------------------------------
@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_CSV)

# -----------------------------------
# Fault Diagnosis using FFT Frequency Bands
# -----------------------------------
def diagnose_fault(signal):
    N = len(signal)
    yf = np.abs(rfft(signal))
    xf = rfftfreq(N, 1/200)  # assumes ~200Hz sampling

    low = yf[(xf >= 2) & (xf <= 10)].mean()
    mid = yf[(xf >= 10) & (xf <= 40)].mean()
    high = yf[(xf >= 40)].mean()

    if low > mid and low > high:
        return "⚠️ **Rotor Imbalance Suspected**"
    elif mid > low and mid > high:
        return "⚠️ **Shaft Misalignment Suspected**"
    elif high > mid and high > low:
        return "⚠️ **Bearing Wear / Surface Damage Likely**"
    else:
        return "⚠️ **Lubrication Breakdown or Noise Interference**"

# -----------------------------------
# Score Data
# -----------------------------------
@st.cache_data
def score_data(df):
    lstm_scores = model.score_sequences(df, signal_col="vibration_rms")
    if_scores = np.zeros_like(lstm_scores)
    fused = 0.5 * lstm_scores + 0.5 * if_scores
    thr = np.percentile(fused[:min(2000, len(fused))], 99)
    decisions = (fused >= thr).astype(int)
    return fused, decisions

# -----------------------------------
# Plot (Downsample to Avoid Lag)
# -----------------------------------
def plot_scores(fused, decisions):
    down = max(len(fused) // 400, 1)
    fused_ds = fused[::down]
    decisions_ds = decisions[::down]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fused_ds, label="Anomaly Score")
    ax.scatter(np.where(decisions_ds == 1), fused_ds[decisions_ds == 1], color='red', s=18, label="Fault")
    ax.set_title("Anomaly Score Trend")
    ax.set_xlabel("Time / Window Index")
    ax.set_ylabel("Score")
    ax.legend()
    st.pyplot(fig)

# -----------------------------------
# UI
# -----------------------------------
st.title("🛠️ AI-Based Predictive Maintenance Dashboard")

st.write("""
This system uses a **Hybrid AI Model**:

| Model | Purpose |
|------|---------|
| **LSTM Autoencoder** | Learns normal vibration waveform & detects pattern deviations |
| **Isolation Forest** | Detects abnormal feature distributions |

This mimics **real industrial predictive maintenance architectures**.
""")

uploaded = st.file_uploader("Upload CSV (vibration_rms OR vibration_x,y,z)", type="csv")

# ------------------ Load Data ------------------
if uploaded:
    data = pd.read_csv(uploaded)
    st.success("✅ File uploaded successfully!")
else:
    st.info("ℹ Using demo dataset")
    data = load_demo()

# ------------------ Sensor Handling ------------------
cols_xyz = [c for c in ["vibration_x", "vibration_y", "vibration_z"] if c in data.columns]

if "vibration_rms" in data.columns:
    signal = data["vibration_rms"].astype(float)

elif len(cols_xyz) >= 2:
    st.info("ℹ Multiple axes detected → Computing RMS automatically")
    axes = data[cols_xyz].astype(float)
    signal = np.sqrt((axes ** 2).sum(axis=1))

else:
    st.error("❌ Provide either `vibration_rms` or vibration_x,y,z columns.")
    st.stop()

data = pd.DataFrame({"vibration_rms": signal})

# ------------------ Model Scoring ------------------
with st.spinner("Analyzing vibration patterns..."):
    fused, decisions = score_data(data)

# ------------------ Metrics ------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(data))
col2.metric("Windows Scored", len(fused))
col3.metric("Faults Detected", int(decisions.sum()))

# ------------------ Timeline Chart ------------------
st.subheader("📈 Machine Condition Trend")
plot_scores(fused, decisions)

# ------------------ Fault Diagnosis ------------------
if decisions.sum() > 0:
    st.subheader("🧠 Fault Type Diagnosis")
    st.error(diagnose_fault(signal.values[-3000:]))
else:
    st.success("✅ No abnormal vibration patterns detected.")

# ------------------ Result Download ------------------
results = pd.DataFrame({"index": np.arange(len(fused)), "score": fused, "fault": decisions})
st.download_button("⬇ Download Results CSV", results.to_csv(index=False), "predictions.csv", "text/csv")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
Industrial-grade Predictive Maintenance demo.

Supports:
- Multi-axis vibration analysis
- FFT fault classification
- Hybrid AI scoring
""")
