import json, time, math
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional imports
try:
    from scipy.signal import butter, filtfilt
    from scipy.stats import kurtosis, skew
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

MQTT_OK = True
try:
    import paho.mqtt.client as mqtt
except Exception:
    MQTT_OK = False

from src.models.hybrid_ensemble import HybridEnsemble
from src.utils.telegram_alert import send_alert

# ────────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="🛠️", layout="wide")

ROOT = Path(".")
MODEL_DIR = ROOT / "models" / "saved_models" / "hybrid"
DEMO_CSV = ROOT / "data" / "synthetic" / "machine_001_demo.csv"

# ────────────────────────────────────────────────────────────────────────────────
# Thread-safe buffer management
# ────────────────────────────────────────────────────────────────────────────────
if "buffer_lock" not in st.session_state:
    st.session_state.buffer_lock = Lock()

# ────────────────────────────────────────────────────────────────────────────────
# Initialize session state
# ────────────────────────────────────────────────────────────────────────────────
if "live_buffer" not in st.session_state:
    st.session_state.live_buffer = deque(maxlen=6000)
if "live_rpm" not in st.session_state:
    st.session_state.live_rpm = deque(maxlen=6000)
if "live_temp" not in st.session_state:
    st.session_state.live_temp = deque(maxlen=6000)
if "live_acoustic" not in st.session_state:
    st.session_state.live_acoustic = deque(maxlen=6000)
if "live_magnetic" not in st.session_state:
    st.session_state.live_magnetic = deque(maxlen=6000)
if "live_current" not in st.session_state:
    st.session_state.live_current = deque(maxlen=6000)

for axis in ("axial", "horizontal", "vertical"):
    key = f"vibration_{axis}"
    if key not in st.session_state:
        st.session_state[key] = deque(maxlen=6000)

if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False
if "mqtt_last_err" not in st.session_state:
    st.session_state.mqtt_last_err = ""
if "asset_name" not in st.session_state:
    st.session_state.asset_name = "Motor-001"
if "data_counter" not in st.session_state:
    st.session_state.data_counter = 0
if "last_message" not in st.session_state:
    st.session_state.last_message = {}

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────
def ensure_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

@st.cache_resource
def load_model():
    required = ["if_model.pkl", "lstm_ae.keras", "scaler.pkl", "meta.json"]
    missing = [f for f in required if not ensure_exists(MODEL_DIR / f)]
    if missing:
        st.error("❌ Trained model files missing")
        st.stop()
    return HybridEnsemble.load(MODEL_DIR.as_posix())

@st.cache_data
def load_demo_dataframe() -> pd.DataFrame:
    if ensure_exists(DEMO_CSV):
        return pd.read_csv(DEMO_CSV)
    n = 5000
    t = np.arange(n) / 200.0
    x = 0.5 + 0.05*np.sin(2*np.pi*3*t) + 0.02*np.random.randn(n)
    y = 0.5 + 0.04*np.sin(2*np.pi*3.2*t + 0.3) + 0.02*np.random.randn(n)
    z = 0.5 + 0.06*np.sin(2*np.pi*2.8*t - 0.2) + 0.02*np.random.randn(n)
    z[2000:2100] += 0.25*np.sin(2*np.pi*15*t[2000:2100])
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    df["vibration_rms"] = np.sqrt((df["x"]**2 + df["y"]**2 + df["z"]**2)/3.0)
    return df

def push_sample_data(x, y, z, rpm=None, temp=None, acoustic=None, magnetic=None, current=None):
    """Thread-safe data pushing"""
    with st.session_state.buffer_lock:
        st.session_state.vibration_axial.append(float(x))
        st.session_state.vibration_horizontal.append(float(y))
        st.session_state.vibration_vertical.append(float(z))
        
        rms = math.sqrt((x*x + y*y + z*z) / 3.0)
        st.session_state.live_buffer.append(rms)
        
        if rpm is not None:
            st.session_state.live_rpm.append(float(rpm))
        if temp is not None:
            st.session_state.live_temp.append(float(temp))
        if acoustic is not None:
            st.session_state.live_acoustic.append(float(acoustic))
        if magnetic is not None:
            st.session_state.live_magnetic.append(float(magnetic))
        if current is not None:
            st.session_state.live_current.append(float(current))
        
        st.session_state.data_counter += 1

def score_live_window(arr: np.ndarray, model):
    if arr.size < 32:
        return np.array([]), np.array([]), 0.0
    df = pd.DataFrame({"vibration_rms": arr})
    lstm = model.score_sequences(df, "vibration_rms")
    ifs = np.zeros_like(lstm)
    fused = 0.5 * lstm + 0.5 * ifs
    thr = float(np.percentile(fused[:min(200, len(fused))], 99))
    dec = (fused >= thr).astype(int)
    return fused, dec, thr

# ────────────────────────────────────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────────────────────────────────────
model = load_model()

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Data Source")
    source = st.radio(
        "Choose input",
        ["Simulated Stream", "MQTT Live"],
        index=0
    )
    st.text_input("Asset Name", value=st.session_state.asset_name, key="asset_name")
    
    st.divider()
    st.header("⚙️ Settings")
    max_points = st.slider("Max chart points", 200, 2000, 500, 100)
    update_interval = st.slider("Update interval (ms)", 100, 2000, 500, 100)

# ────────────────────────────────────────────────────────────────────────────────
# Main Dashboard
# ────────────────────────────────────────────────────────────────────────────────
st.title("🟢 Live Monitoring Dashboard")

left, mid, right = st.columns([2, 2, 1.4])

with left:
    st.write(f"**Mode:** {source}")

with mid:
    sensor_choice = st.selectbox(
        "Signal",
        ["Vibration", "Temperature", "RPM", "Acoustics", "Magnetic Flux", "Current"]
    )

with right:
    time_range = st.radio("Time Range", ["Last 100", "Last 500", "All"], horizontal=False)

axis_choice = None
if sensor_choice == "Vibration":
    axis_choice = st.radio("Axis", ["All (RMS)", "Axial", "Horizontal", "Vertical"], horizontal=True)

threshold = st.slider("Alert Threshold", 0.0, 2.0, 0.8, 0.1)

# ────────────────────────────────────────────────────────────────────────────────
# MODE: Simulated Stream
# ────────────────────────────────────────────────────────────────────────────────
if source == "Simulated Stream":
    st.caption("📡 Simulating live sensor data")
    
    sim_rate = st.slider("Samples per update", 1, 50, 10, 1)
    
    c1, c2, c3 = st.columns(3)
    if c1.button("▶️ Start Stream"):
        st.session_state.live_running = True
        st.rerun()
    
    if c2.button("⏸️ Pause"):
        st.session_state.live_running = False
    
    if c3.button("🛑 Reset"):
        st.session_state.live_running = False
        with st.session_state.buffer_lock:
            for key in ("live_buffer","live_rpm","live_temp","live_acoustic","live_magnetic","live_current",
                        "vibration_axial","vibration_horizontal","vibration_vertical"):
                st.session_state[key].clear()
            st.session_state.pop("sim_idx", None)
            st.session_state.data_counter = 0
        st.rerun()
    
    if st.session_state.live_running:
        demo = load_demo_dataframe()
        
        if {"x","y","z"}.issubset(demo.columns):
            xs, ys, zs = demo["x"].values, demo["y"].values, demo["z"].values
        else:
            rms = demo["vibration_rms"].values
            xs, ys, zs = rms*0.97, rms*1.01, rms*1.03
        
        if "sim_idx" not in st.session_state:
            st.session_state.sim_idx = 0
        
        i0, i1 = st.session_state.sim_idx, st.session_state.sim_idx + sim_rate
        
        for i in range(i0, i1):
            j = i % len(xs)
            rpm = 1500 + 25*np.sin(j/180) + np.random.randn()*5
            temp = 65 + 1.5*np.sin(j/360) + np.random.randn()*0.5
            acoustic = 0.2 + 0.02*np.sin(j/140) + np.random.randn()*0.01
            magnetic = 0.1 + 0.01*np.sin(j/200) + np.random.randn()*0.005
            current = 3.0 + 0.1*np.sin(j/220) + np.random.randn()*0.05
            
            push_sample_data(xs[j], ys[j], zs[j], rpm=rpm, temp=temp,
                           acoustic=acoustic, magnetic=magnetic, current=current)
        
        st.session_state.sim_idx = i1
        time.sleep(update_interval/1000.0)
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# MODE: MQTT Live - FIXED for port 1883
# ────────────────────────────────────────────────────────────────────────────────
elif source == "MQTT Live":
    st.caption("🌍 MQTT broker: broker.hivemq.com:1883 (TCP)")
    st.caption("📝 Topic: machine/vibration/data")
    
    if not MQTT_OK:
        st.error("Install MQTT: `pip install paho-mqtt`")
    else:
        # Show last received message
        if st.session_state.last_message:
            with st.expander("📨 Last Received Message", expanded=True):
                st.json(st.session_state.last_message)
        
        def on_connect(client, userdata, flags, rc, properties=None):
            """Callback when connected - NO Streamlit calls here"""
            if rc == 0:
                st.session_state.mqtt_connected = True
                st.session_state.mqtt_last_err = ""
                client.subscribe("machine/vibration/data", qos=0)
            else:
                st.session_state.mqtt_connected = False
                st.session_state.mqtt_last_err = f"Connect failed (rc={rc})"
        
        def on_message(client, userdata, msg):
            """Callback when message received - NO Streamlit calls here"""
            try:
                payload = msg.payload.decode("utf-8")
                j = json.loads(payload)
                
                # Store last message for display
                st.session_state.last_message = j
                
                # Support both naming conventions
                ax = float(j.get("axial", j.get("x", 0.0)))
                hz = float(j.get("horizontal", j.get("y", 0.0)))
                vt = float(j.get("vertical", j.get("z", 0.0)))
                
                rpm = float(j.get("rpm", 1500))
                temp = float(j.get("temp", 65))
                acoustic = float(j.get("acoustic", 0.2))
                magnetic = float(j.get("mag", 0.1))
                current = float(j.get("current", 3.0))
                
                # Thread-safe push
                push_sample_data(ax, hz, vt, rpm=rpm, temp=temp,
                               acoustic=acoustic, magnetic=magnetic, current=current)
                
                st.session_state.mqtt_last_err = ""
                
            except Exception as e:
                st.session_state.mqtt_last_err = f"Parse error: {e}"
        
        ca, cb, cc = st.columns(3)
        
        if ca.button("🔌 Connect MQTT"):
            try:
                # Use standard TCP connection (port 1883)
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                client.on_connect = on_connect
                client.on_message = on_message
                
                # Connect to standard MQTT port
                client.connect("broker.hivemq.com", 1883, 60)
                client.loop_start()
                
                st.session_state.mqtt_client = client
                st.success("Connecting to broker.hivemq.com:1883...")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.session_state.mqtt_last_err = str(e)
        
        if cb.button("🔕 Disconnect"):
            c = st.session_state.get("mqtt_client")
            if c:
                try:
                    c.loop_stop()
                    c.disconnect()
                except:
                    pass
            st.session_state.mqtt_connected = False
            st.session_state.last_message = {}
            st.rerun()
        
        if cc.button("🧹 Clear Buffers"):
            with st.session_state.buffer_lock:
                for key in ("live_buffer","live_rpm","live_temp","live_acoustic","live_magnetic","live_current",
                            "vibration_axial","vibration_horizontal","vibration_vertical"):
                    st.session_state[key].clear()
                st.session_state.data_counter = 0
                st.session_state.last_message = {}
            st.rerun()
        
        # Status indicators
        col_status1, col_status2 = st.columns(2)
        
        with col_status1:
            if st.session_state.mqtt_connected:
                st.success("✅ MQTT Connected")
            else:
                st.info("⏸️ Not connected")
        
        with col_status2:
            st.metric("Messages Received", st.session_state.data_counter)
        
        if st.session_state.mqtt_last_err:
            st.warning(f"⚠️ {st.session_state.mqtt_last_err}")
        
        # Auto-refresh when connected
        if st.session_state.mqtt_connected:
            time.sleep(update_interval / 1000.0)
            st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# Data Visualization (Common for both modes)
# ────────────────────────────────────────────────────────────────────────────────
def pick_series():
    with st.session_state.buffer_lock:
        if sensor_choice == "Vibration":
            if axis_choice == "Axial":
                return np.array(list(st.session_state.vibration_axial))
            elif axis_choice == "Horizontal":
                return np.array(list(st.session_state.vibration_horizontal))
            elif axis_choice == "Vertical":
                return np.array(list(st.session_state.vibration_vertical))
            else:
                return np.array(list(st.session_state.live_buffer))
        elif sensor_choice == "Temperature":
            return np.array(list(st.session_state.live_temp))
        elif sensor_choice == "RPM":
            return np.array(list(st.session_state.live_rpm))
        elif sensor_choice == "Acoustics":
            return np.array(list(st.session_state.live_acoustic))
        elif sensor_choice == "Magnetic Flux":
            return np.array(list(st.session_state.live_magnetic))
        elif sensor_choice == "Current":
            return np.array(list(st.session_state.live_current))
        return np.array([])

series = pick_series()

# Apply time range
if time_range == "Last 100":
    series = series[-100:]
elif time_range == "Last 500":
    series = series[-500:]

st.divider()

# KPIs
k1, k2, k3, k4 = st.columns(4)

if series.size > 0:
    current_val = float(series[-1])
    avg_val = float(np.mean(series))
    max_val = float(np.max(series))
    min_val = float(np.min(series))
    
    faults = 0
    if sensor_choice == "Vibration" and series.size >= 32:
        fused, dec, thr = score_live_window(series, model)
        faults = int((dec == 1).sum()) if dec.size else 0
        
        if faults > 0:
            msg = f"⚠️ Fault Detected!\nAsset: {st.session_state.asset_name}\nFaults: {faults}"
            try:
                send_alert(msg)
            except:
                pass
    
    k1.metric("Current Value", f"{current_val:.3f}", 
              delta=f"{current_val - avg_val:.3f}" if series.size > 1 else None)
    k2.metric("Average", f"{avg_val:.3f}")
    k3.metric("Max", f"{max_val:.3f}")
    k4.metric("Samples", len(series))
    
    if faults > 0:
        st.error(f"🚨 {faults} anomalies detected!")
    
    # Chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=series,
        mode="lines",
        name=sensor_choice,
        line=dict(color="steelblue", width=2),
        fill='tozeroy',
        fillcolor='rgba(70, 130, 180, 0.2)'
    ))
    
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text="Threshold"
    )
    
    if sensor_choice == "Vibration" and faults > 0:
        anomaly_mask = series > threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        if len(anomaly_indices) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_indices,
                y=series[anomaly_indices],
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name="Anomaly"
            ))
    
    fig.update_layout(
        title=f"{sensor_choice} — Live Stream (Updates: {st.session_state.data_counter})",
        xaxis_title="Sample Index",
        yaxis_title="Value",
        template="plotly_white",
        height=450,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📊 Live Statistics")
    stats_df = pd.DataFrame({
        "Metric": ["Current", "Mean", "Std Dev", "Min", "Max", "Range"],
        "Value": [
            f"{current_val:.4f}",
            f"{avg_val:.4f}",
            f"{float(np.std(series)):.4f}",
            f"{min_val:.4f}",
            f"{max_val:.4f}",
            f"{max_val - min_val:.4f}"
        ]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
else:
    st.info("⏳ Waiting for data... Start the stream or connect to MQTT")
    k1.metric("Samples", 0)
    k2.metric("Status", "No Data")
    k3.metric("Total Updates", st.session_state.data_counter)
    k4.metric("Threshold", threshold)
