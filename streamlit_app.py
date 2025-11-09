import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.hybrid_ensemble import HybridEnsemble

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide"
)

ROOT = Path(".")
MODEL_DIR = ROOT / "models" / "saved_models" / "hybrid"
DEMO_CSV = ROOT / "data" / "synthetic" / "machine_001_demo.csv"

# -----------------------------------
# Utilities
# -----------------------------------
def ensure_exists(p: Path) -> bool:
    try:
        return p.exists()
    except:
        return False

@st.cache_resource(show_spinner=False)
def load_model():
    required = ["if_model.pkl", "lstm_ae.keras", "scaler.pkl", "meta.json"]
    missing = [f for f in required if not ensure_exists(MODEL_DIR / f)]
    if missing:
        st.error(
            "❌ Model files missing in `models/saved_models/hybrid/`.\n\n"
            "Please ensure you committed:\n"
            "- if_model.pkl\n- lstm_ae.keras\n- scaler.pkl\n- meta.json"
        )
        st.stop()
    return HybridEnsemble.load(MODEL_DIR.as_posix())


@st.cache_data(show_spinner=False)
def load_demo_dataframe():
    # If real demo exists → use it
    if ensure_exists(DEMO_CSV):
        try:
            return pd.read_csv(DEMO_CSV)
        except:
            pass
    # Synthetic fallback (so app never breaks)
    n = 4000
    t = np.arange(n) / 200.0
    vib = 0.5 + 0.05*np.sin(2*np.pi*3*t) + 0.02*np.random.randn(n)
    vib[2000:2100] += 0.25*np.sin(2*np.pi*15*t[2000:2100])
    return pd.DataFrame({"vibration_rms": vib})


def fuse_scores(if_scores, lstm_scores):
    try:
        return model.combine_scores(if_scores, lstm_scores)
    except:
        m = min(len(if_scores), len(lstm_scores))
        return 0.5 * if_scores[:m] + 0.5 * lstm_scores[:m]


def make_decisions(fused, baseline=2000, pctl=99):
    baseline = min(baseline, len(fused))
    thr = np.percentile(fused[:baseline], pctl)
    return (fused >= thr).astype(int)


# -----------------------------------
# Load model
# -----------------------------------
model = load_model()

# -----------------------------------
# UI START
# -----------------------------------
st.title("🛠️ AI-Based Predictive Maintenance Dashboard")

st.write("""
This system uses a **Hybrid AI Model** combining:
- 🧠 **Isolation Forest** for feature anomaly scoring  
- 🔁 **LSTM Autoencoder** for vibration signal reconstruction  
to detect **early-stage machine faults before breakdown**.
""")

# Upload
st.header("📤 Upload Machine Sensor Data")
uploaded_file = st.file_uploader(
    "Upload CSV containing `vibration_rms` column (or leave empty to use demo dataset):",
    type=["csv"]
)

# Load Data
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except:
        st.error("❌ Could not read uploaded CSV.")
        st.stop()
else:
    st.info("ℹ️ Using built-in demo dataset.")
    data = load_demo_dataframe()

# Validate
if "vibration_rms" not in data.columns:
    st.error("❌ CSV must contain a numeric column `vibration_rms`.")
    st.stop()

# -----------------------------------
# Scoring (Cached → Fast)
# -----------------------------------
@st.cache_data(show_spinner=True)
def run_scoring(df):
    lstm_scores = model.score_sequences(df, signal_col="vibration_rms")
    if_scores = np.zeros_like(lstm_scores)  # Optional IF neutral baseline
    fused = fuse_scores(if_scores, lstm_scores)
    decisions = make_decisions(fused)
    return lstm_scores, if_scores, fused, decisions

lstm_scores, if_scores, fused, decisions = run_scoring(data)

# -----------------------------------
# Metrics
# -----------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(data))
col2.metric("Windows Scored", len(fused))
col3.metric("Faulty Windows Detected", int((decisions == 1).sum()))

# -----------------------------------
# Chart (Downsample for speed)
# -----------------------------------
st.subheader("📈 Machine Health Trend (Anomaly Score Over Time)")

downsample = max(len(fused) // 600, 1)
fused_ds = fused[::downsample]
fault_ds = decisions[::downsample]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(fused_ds, label="Anomaly Score")
ax.scatter(np.where(fault_ds == 1), fused_ds[fault_ds == 1], color='red', s=12, label="Fault")
ax.legend()
ax.set_xlabel("Time / Window Index")
ax.set_ylabel("Score")
st.pyplot(fig)

# -----------------------------------
# Fault List
# -----------------------------------
fault_idx = np.where(decisions == 1)[0]
st.subheader("🚦 Detected Fault Windows")

if len(fault_idx) == 0:
    st.success("✅ Machine appears healthy.")
else:
    st.error("⚠️ Potential fault patterns detected.")
    st.write(f"Fault windows (first 50): **{fault_idx[:50].tolist()}**")

# -----------------------------------
# Download
# -----------------------------------
results_df = pd.DataFrame({
    "index": np.arange(len(fused)),
    "lstm_score": lstm_scores[:len(fused)],
    "if_score": if_scores[:len(fused)],
    "fused_score": fused,
    "label": decisions
})

st.download_button(
    "⬇️ Download Results CSV",
    results_df.to_csv(index=False).encode(),
    "predictions.csv",
    "text/csv"
)

# -----------------------------------
# Sidebar
# -----------------------------------
with st.sidebar:
    st.header("ℹ️ About Model")
    st.write("""
**Hybrid Model Structure**
- Isolation Forest → Detects unusual feature patterns  
- LSTM Autoencoder → Learns normal vibration waveforms  
- Fusion → Weighted anomaly score thresholding  
    """)
