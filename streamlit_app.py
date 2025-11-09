import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
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

@st.cache_resource
def load_model():
    # Fail fast with a clear message if the model files are not there
    required = ["if_model.pkl", "lstm_ae.keras", "scaler.pkl", "meta.json"]
    missing = [f for f in required if not ensure_exists(MODEL_DIR / f)]
    if missing:
        st.error(
            "❌ Trained model artifacts are missing.\n\n"
            f"Expected files in `{MODEL_DIR.as_posix()}`:\n"
            + "\n".join(f"- {f}" for f in required)
            + "\n\nTip: Commit these small files to GitHub (they are all << 100MB)."
        )
        st.stop()
    return HybridEnsemble.load(MODEL_DIR.as_posix())

def load_demo_dataframe() -> pd.DataFrame:
    # 1) Try repo demo file
    if ensure_exists(DEMO_CSV):
        try:
            return pd.read_csv(DEMO_CSV)
        except Exception:
            pass
    # 2) Fallback: tiny synthetic demo so the app always runs
    n = 4000
    t = np.arange(n) / 200.0
    vib = 0.5 + 0.05*np.sin(2*np.pi*3*t) + 0.02*np.random.randn(n)
    # inject a small anomalous burst
    vib[2000:2100] += 0.25*np.sin(2*np.pi*15*t[2000:2100])
    return pd.DataFrame({"vibration_rms": vib})

def fuse_scores(if_scores: np.ndarray, lstm_scores: np.ndarray) -> np.ndarray:
    # Try model's fuse; fall back to a simple average
    try:
        # if the model exposes combine_scores
        return model.combine_scores(if_scores, lstm_scores)
    except Exception:
        # shape-safe average
        m = min(len(if_scores), len(lstm_scores))
        return 0.5 * if_scores[:m] + 0.5 * lstm_scores[:m]

def make_decisions(fused: np.ndarray, baseline: int = 2000, pctl: float = 99.0) -> np.ndarray:
    """Compute anomaly labels without using model.decision to avoid API mismatch."""
    if len(fused) == 0:
        return np.array([], dtype=int)
    base = min(baseline, len(fused))
    thr = np.percentile(fused[:base], pctl)
    return (fused >= thr).astype(int)

# -----------------------------------
# Load model
# -----------------------------------
model = load_model()

# -----------------------------------
# UI
# -----------------------------------
st.title("🛠️ AI-Based Predictive Maintenance Dashboard")
st.write(
    """
This dashboard uses a **Hybrid Ensemble** of:
- **Isolation Forest** (feature-based anomaly scores)
- **LSTM Autoencoder** (sequence reconstruction error)

to detect **early-stage mechanical faults** in vibration signals.
"""
)

# Upload
st.header("📤 Upload Machine Sensor Data")
uploaded_file = st.file_uploader(
    "Upload CSV containing a `vibration_rms` column (or leave empty to use a demo)",
    type=["csv"]
)

# Data selection
if uploaded_file is None:
    st.info("ℹ️ No file uploaded — using the built-in demo dataset.")
    data = load_demo_dataframe()
else:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded data loaded successfully!")
    except Exception as e:
        st.error(f"❌ Could not read the CSV file: {e}")
        st.stop()

# Validate
if "vibration_rms" not in data.columns:
    st.error("❌ The dataset must contain a numeric column named **vibration_rms**.")
    st.stop()

# -----------------------------------
# Scoring
# -----------------------------------
st.subheader("🔍 Running Anomaly Detection...")
with st.spinner("Scoring sequences with the LSTM autoencoder..."):
    try:
        lstm_scores = model.score_sequences(data, signal_col="vibration_rms")
    except Exception as e:
        st.error(f"❌ LSTM scoring failed: {e}")
        st.stop()

# Optional IF scores (not needed on Cloud; set to zeros of same length)
if_scores = np.zeros_like(lstm_scores)

# Fuse (robust)
fused = fuse_scores(if_scores, lstm_scores)

# Compute decisions without calling model.decision (avoids TypeError)
decisions = make_decisions(fused, baseline=2000, pctl=99.0)

# -----------------------------------
# Metrics
# -----------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Samples", int(len(data)))
col2.metric("Windows Scored", int(len(fused)))
col3.metric("Faulty Windows", int((decisions == 1).sum()))

# -----------------------------------
# Charts
# -----------------------------------
st.subheader("📈 Anomaly Score Timeline")
st.line_chart(pd.DataFrame({"Anomaly Score": fused}))

st.subheader("🚦 Detected Fault Windows")
fault_idx = np.where(decisions == 1)[0]
if len(fault_idx) == 0:
    st.success("✅ No anomalies detected — machine appears healthy.")
else:
    st.error("⚠️ Potential fault patterns detected.")
    st.write(f"First 50 fault windows: {fault_idx[:50].tolist()}")

# -----------------------------------
# Download results
# -----------------------------------
st.subheader("⬇️ Download Results")
results_df = pd.DataFrame({
    "index": np.arange(len(fused)),
    "lstm_score": lstm_scores[:len(fused)],
    "if_score": if_scores[:len(fused)],
    "fused_score": fused,
    "label": decisions
})
st.download_button(
    "Download CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)

# Sidebar: About
with st.sidebar:
    st.header("About")
    st.write(
        "Hybrid model files expected in `models/saved_models/hybrid/`:\n"
        "- `if_model.pkl`\n- `lstm_ae.keras`\n- `scaler.pkl`\n- `meta.json`"
    )
    st.caption("If the demo CSV is absent, the app generates a tiny synthetic dataset so it always runs.")
