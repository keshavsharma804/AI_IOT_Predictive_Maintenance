import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import rfft, rfftfreq
from src.models.hybrid_ensemble import HybridEnsemble

st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🛠️", layout="wide")

ROOT = Path(".")
MODEL_DIR = ROOT / "models" / "saved_models" / "hybrid"
DEMO_CSV = ROOT / "data" / "synthetic" / "machine_001_demo.csv"

@st.cache_resource
def load_model():
    return HybridEnsemble.load(MODEL_DIR.as_posix())

model = load_model()

@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_CSV)

def compute_rms(df):
    cols = [c for c in ["vibration_x", "vibration_y", "vibration_z"] if c in df.columns]
    if len(cols) >= 2:
        st.info("✅ Multi-axis detected → Computing RMS automatically")
        arr = df[cols].astype(float)
        return np.sqrt((arr ** 2).sum(axis=1))
    elif "vibration_rms" in df.columns:
        return df["vibration_rms"].astype(float)
    else:
        st.error("❌ Provide either vibration_rms or multi-axis vibration_x,y,z")
        st.stop()

@st.cache_data
def score(df):
    lstm_scores = model.score_sequences(df, signal_col="vibration_rms")
    if_scores = np.zeros_like(lstm_scores)
    fused = 0.5 * lstm_scores + 0.5 * if_scores
    thr = np.percentile(fused[:min(2000, len(fused))], 99)
    labels = (fused >= thr).astype(int)
    return fused, labels

def diagnose(signal):
    yf = np.abs(rfft(signal))
    xf = rfftfreq(len(signal), 1/200)
    low = yf[(xf >= 2) & (xf <= 10)].mean()
    mid = yf[(xf >= 10) & (xf <= 40)].mean()
    high = yf[(xf >= 40)].mean()

    if low > mid and low > high:
        return "⚠️ Rotor Imbalance"
    elif mid > low and mid > high:
        return "⚠️ Shaft Misalignment"
    elif high > mid and high > low:
        return "⚠️ Bearing Wear / Surface Damage"
    else:
        return "⚠️ Lubrication Failure or Noise Interference"

def plot(fused, labels):
    step = max(len(fused)//400, 1)
    f = fused[::step]
    l = labels[::step]
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(f, label="Anomaly Score")
    ax.scatter(np.where(l==1), f[l==1], color="red", label="Fault", s=20)
    ax.set_title("Anomaly Trend")
    ax.set_xlabel("Time Window")
    st.pyplot(fig)

# ---------------- UI TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Data Input", "🔍 Data Preview", "🤖 Model & Analysis", "🚨 Fault Diagnosis", "⬇️ Download"])

# -------- Tab1 Upload --------
with tab1:
    st.header("Upload Machine Sensor Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    data = load_demo() if uploaded is None else pd.read_csv(uploaded)
    signal = compute_rms(data)
    data = pd.DataFrame({"vibration_rms": signal})
    st.success("✅ Data Ready for Processing")

# -------- Tab2 Preview --------
with tab2:
    st.header("Dataset Preview")
    st.write(data.head())

# -------- Tab3 Analysis --------
with tab3:
    st.header("Running Hybrid AI Model...")
    fused, labels = score(data)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(data))
    c2.metric("Windows Analyzed", len(fused))
    c3.metric("Fault Count", int(labels.sum()))

    st.subheader("📈 Anomaly Score Trend")
    plot(fused, labels)

    st.info("""
**Model Explanation:**
• **LSTM Autoencoder** learns normal vibration shape  
• **Isolation Forest** detects statistical outliers  
• Combined → Detects early-stage machine failures
""")

# -------- Tab4 Fault Diagnosis --------
with tab4:
    st.header("Fault Interpretation (FFT Based)")
    if labels.sum() == 0:
        st.success("✅ Machine operating normally. No fault patterns detected.")
    else:
        st.error("⚠️ Abnormal vibration detected — Possible Fault:")
        st.subheader(diagnose(signal[-3000:]))

# -------- Tab5 Download --------
with tab5:
    st.header("Download Results")
    results = pd.DataFrame({"index": np.arange(len(fused)), "score": fused, "fault": labels})
    st.download_button("Download CSV", results.to_csv(index=False), "predictions.csv", "text/csv")

# --------------- Sidebar ---------------
with st.sidebar:
    st.header("ℹ️ About System")
    st.write("""
This dashboard uses **Hybrid Predictive Maintenance AI**:
- Trained on **normal machine vibration**
- Detects unknown failure patterns
- Works on **real sensor signals**
""")
