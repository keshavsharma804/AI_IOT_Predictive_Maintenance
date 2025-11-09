import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.models.hybrid_ensemble import HybridEnsemble

st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🛠️", layout="wide")

ROOT = Path(".")
MODEL_DIR = ROOT / "models" / "saved_models" / "hybrid"
DEMO_CSV = ROOT / "data" / "synthetic" / "machine_001_demo.csv"


@st.cache_resource
def load_model():
    return HybridEnsemble.load(MODEL_DIR.as_posix())


@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_CSV)


@st.cache_data
def score_data(df):
    lstm_scores = model.score_sequences(df, signal_col="vibration_rms")
    if_scores = np.zeros_like(lstm_scores)
    # fused = model.combine_scores(if_scores, lstm_scores)
    m = min(len(if_scores), len(lstm_scores))
    fused = 0.5 * if_scores[:m] + 0.5 * lstm_scores[:m]
    thr = np.percentile(fused[:min(2000, len(fused))], 99)
    decisions = (fused >= thr).astype(int)
    return fused, decisions


def plot_scores(fused, decisions):
    downsample = max(len(fused) // 500, 1)
    fused_ds = fused[::downsample]
    decisions_ds = decisions[::downsample]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fused_ds, label="Anomaly Score")
    ax.scatter(np.where(decisions_ds == 1), fused_ds[decisions_ds == 1], color='red', s=12, label="Fault")
    ax.set_title("Anomaly Score Trend")
    ax.set_xlabel("Time / Window Index")
    ax.set_ylabel("Score")
    ax.legend()
    st.pyplot(fig)


# ---------------- UI ----------------
model = load_model()

st.title("🛠️ AI-Based Predictive Maintenance Dashboard")

st.write("""
This dashboard uses a **Hybrid AI Model** combining:
- **LSTM Autoencoder** (sequence reconstruction)
- **Isolation Forest** (feature anomaly)

to detect **early machinery faults** from vibration signals.
""")

uploaded = st.file_uploader("Upload CSV (must contain `vibration_rms`)", type="csv")

if uploaded:
    try:
        data = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")
    except:
        st.error("❌ Could not read the file.")
        st.stop()
else:
    st.info("ℹ️ Using demo dataset")
    data = load_demo()

if "vibration_rms" not in data.columns:
    st.error("❌ Required column `vibration_rms` missing.")
    st.stop()

with st.spinner("Running anomaly detection..."):
    fused, decisions = score_data(data)

# ---- Metrics ----
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(data))
col2.metric("Windows Scored", len(fused))
col3.metric("Faults Detected", int((decisions == 1).sum()))

# ---- Chart ----
st.subheader("📈 Machine Condition Over Time")
plot_scores(fused, decisions)

# ---- Results Table Download ----
results_df = pd.DataFrame({
    "index": np.arange(len(fused)),
    "fused_score": fused,
    "label": decisions
})

st.download_button("Download Results as CSV", results_df.to_csv(index=False), "predictions.csv", "text/csv")

with st.sidebar:
    st.header("About")
    st.write("This system detects early machine faults using hybrid AI (LSTM + Isolation Forest).")
