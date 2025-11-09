import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import seaborn as sns
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
    try: return p.exists()
    except: return False

@st.cache_resource
def load_model():
    required = ["if_model.pkl", "lstm_ae.keras", "scaler.pkl", "meta.json"]
    missing = [f for f in required if not ensure_exists(MODEL_DIR / f)]
    if missing:
        st.error("❌ Model artifacts missing in models/saved_models/hybrid/")
        st.stop()
    return HybridEnsemble.load(MODEL_DIR.as_posix())

def load_demo_data():
    if ensure_exists(DEMO_CSV):
        return pd.read_csv(DEMO_CSV)
    # Auto-synthetic fallback dataset
    t = np.arange(4000) / 200.0
    vib = 0.5 + 0.05*np.sin(2*np.pi*3*t) + 0.02*np.random.randn(len(t))
    vib[2000:2100] += 0.25*np.sin(2*np.pi*15*t[2000:2100])
    return pd.DataFrame({"vibration_rms": vib})

def butter_lowpass_filter(x, cutoff=50, fs=1000, order=4):
    b, a = butter(order, cutoff/(0.5*fs), btype="low")
    return filtfilt(b, a, x)

def fuse_scores(if_scores, lstm_scores):
    try:
        return model.combine_scores(if_scores, lstm_scores)
    except:
        m = min(len(if_scores), len(lstm_scores))
        return 0.5 * if_scores[:m] + 0.5 * lstm_scores[:m]

def make_decisions(fused, baseline=2000, pctl=99.0):
    thr = np.percentile(fused[:min(baseline, len(fused))], pctl)
    return (fused >= thr).astype(int)

# -----------------------------------
# Load model
# -----------------------------------
model = load_model()

# -----------------------------------
# UI Header
# -----------------------------------
st.title("🛠️ AI-Based Predictive Maintenance Dashboard")
st.write("""
This dashboard analyzes **vibration sensor data** to detect **early machine faults**, using:

- 🧠 *LSTM Autoencoder* → Learns normal vibration behavior  
- 🌲 *Isolation Forest* → Detects statistical anomalies  
""")

# Upload Section
st.header("📤 Upload Machine Sensor Data")
uploaded = st.file_uploader("Upload CSV (supports vibration X/Y/Z, temperature, acoustic)", type=["csv"])

if uploaded:
    data = pd.read_csv(uploaded)
    st.success("✅ File loaded successfully!")
else:
    st.info("ℹ️ Using demo dataset.")
    data = load_demo_data()

if "vibration_rms" not in data.columns:
    st.error("Dataset must contain a `vibration_rms` column.")
    st.stop()

# -----------------------------------
# Raw Signal Visualization
# -----------------------------------
st.subheader("📊 Raw Sensor Signal(s)")
possible_signals = [c for c in data.columns if c not in ["timestamp"]]
selected_signals = st.multiselect("Select signals to visualize:", possible_signals, default=["vibration_rms"])
if selected_signals:
    st.line_chart(data[selected_signals])

# -----------------------------------
# Filtering & FFT
# -----------------------------------
st.subheader("🔧 Filtered Signal (Noise Reduction)")
data["filtered"] = butter_lowpass_filter(data["vibration_rms"])
st.line_chart(data[["vibration_rms","filtered"]])

st.subheader("⚡ Frequency Spectrum (FFT)")
signal = data["vibration_rms"].values
freq = np.fft.rfftfreq(len(signal), 1/1000)
amp = np.abs(np.fft.rfft(signal))
st.line_chart(pd.DataFrame({"Amplitude": amp}, index=freq))

# -----------------------------------
# Feature Extraction
# -----------------------------------
st.subheader("📐 Extracted Signal Features")
features = pd.DataFrame({
    "RMS": [np.sqrt(np.mean(signal**2))],
    "Peak": [np.max(np.abs(signal))],
    "Kurtosis": [kurtosis(signal)],
    "Skewness": [skew(signal)]
})
st.table(features)

# -----------------------------------
# Run LSTM Model
# -----------------------------------
st.subheader("🔍 Running Anomaly Detection...")
lstm_scores = model.score_sequences(data, "vibration_rms")
if_scores = np.zeros_like(lstm_scores)
fused = fuse_scores(if_scores, lstm_scores)
decisions = make_decisions(fused)

# Metrics
st.metric("Samples Processed", len(data))
st.metric("Faulty Windows", int(decisions.sum()))

# -----------------------------------
# LSTM Reconstruction Visualization
# -----------------------------------
st.subheader("🧠 LSTM Reconstruction Comparison")
try:
    reconstructed = model.reconstruct_sequences(data, "vibration_rms")
    comp = pd.DataFrame({"Original": data["vibration_rms"][:len(reconstructed)],
                         "Reconstructed": reconstructed})
    st.line_chart(comp)
except:
    st.info("Reconstruction visualization is not available for this model.")

# -----------------------------------
# Fault Heatmap
# -----------------------------------
st.subheader("🔥 Fault Heatmap")
fig, ax = plt.subplots(figsize=(12,2))
sns.heatmap([fused], cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------------
# Download Results
# -----------------------------------
st.download_button("⬇️ Download Predictions CSV", 
                   results_df := pd.DataFrame({"fused_score": fused, "label": decisions}).to_csv(index=False),
                   "predictions.csv", "text/csv")
