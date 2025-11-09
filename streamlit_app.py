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
    except Exception:
        return False

@st.cache_resource(show_spinner=False)
def load_model():
    required = ["if_model.pkl", "lstm_ae.keras", "scaler.pkl", "meta.json"]
    missing = [f for f in required if not ensure_exists(MODEL_DIR / f)]
    if missing:
        st.error(
            "❌ Model is missing one or more required files:\n\n"
            + "\n".join(f"- {f}" for f in missing)
        )
        st.stop()
    return HybridEnsemble.load(MODEL_DIR.as_posix())

@st.cache_data(show_spinner=False)
def load_demo_dataframe() -> pd.DataFrame:
    if ensure_exists(DEMO_CSV):
        return pd.read_csv(DEMO_CSV)

    # fallback synthetic data
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
    if len(fused) == 0:
        return np.array([], dtype=int)
    thr = np.percentile(fused[:min(baseline, len(fused))], pctl)
    return (fused >= thr).astype(int)

# -----------------------------------
# Load model
# -----------------------------------
model = load_model()

# -----------------------------------
# UI
# -----------------------------------
st.title("🛠️ AI-Based Predictive Maintenance Dashboard")
st.write("""
This dashboard uses a **Hybrid Ensemble** of:
- **Isolation Forest** (feature-based anomaly scores)
- **LSTM Autoencoder** (sequence reconstruction error)

to detect **early-stage mechanical faults** in vibration signals.
""")

# Upload
st.header("📤 Upload Machine Sensor Data")
uploaded_file = st.file_uploader(
    "Upload CSV containing a `vibration_rms` column (or leave empty to use demo)",
    type=["csv"]
)

# Load Data
if uploaded_file is None:
    st.info("ℹ️ Using built
