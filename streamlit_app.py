import json, time, math
from collections import deque
from pathlib import Path
from threading import Lock
import queue

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
# Initialize session state FIRST
# ────────────────────────────────────────────────────────────────────────────────
if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue()

if "live_buffer" not in st.session_state:
    st.session_state.live_buffer = deque(maxlen=6000)
if "live_rpm" not in st.session_state:
    st.session_state.live_rpm = deque(maxlen=6000)
if "live_temp" not in st.session_state:
    st.session_state.live_temp = deque(maxlen=6000)

for axis in ("axial", "horizontal", "vertical"):
    key = f"vibration_{axis}"
    if key not in st.session_state:
        st.session_state[key] = deque(maxlen=6000)

if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False
if "mqtt_status" not in st.session_state:
    st.session_state.mqtt_status = "Not connected"
if "asset_name" not in st.session_state:
    st.session_state.asset_name = "Motor-001"
if "data_counter" not in st.session_state:
    st.session_state.data_counter = 0
if "last_message" not in st.session_state:
    st.session_state.last_message = {}
if "connection_attempts" not in st.session_state:
    st.session_state.connection_attempts = 0

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

def process_message_queue():
    """Process all messages from queue - called from main thread"""
    processed = 0
    while not st.session_state.message_queue.empty():
        try:
            msg_data = st.session_state.message_queue.get_nowait()
            
            x = float(msg_data.get("x", 0.0))
            y = float(msg_data.get("y", 0.0))
            z = float(msg_data.get("z", 0.0))
            rpm = float(msg_data.get("rpm", 1500))
            temp = float(msg_data.get("temp", 65))
            
            # Add to buffers
            st.session_state.vibration_axial.append(x)
            st.session_state.vibration_horizontal.append(y)
            st.session_state.vibration_vertical.append(z)
            
            rms = math.sqrt((x*x + y*y + z*z) / 3.0)
            st.session_state.live_buffer.append(rms)
            st.session_state.live_rpm.append(rpm)
            st.session_state.live_temp.append(temp)
            
            st.session_state.data_counter += 1
            st.session_state.last_message = msg_data
            processed += 1
            
        except queue.Empty:
            break
        except Exception as e:
            st.session_state.mqtt_status = f"Process error: {e}"
    
    return processed

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
# MQTT Setup with proper callbacks
# ────────────────────────────────────────────────────────────────────────────────
def create_mqtt_client():
    """Create MQTT client with callbacks that only use queue"""
    
    def on_connect(client, userdata, flags, rc):
        """Connection callback - thread-safe"""
        if rc == 0:
            st.session_state.mqtt_connected = True
            st.session_state.mqtt_status = "✅ Connected to broker"
            client.subscribe("machine/vibration/data", qos=0)
        else:
            st.session_state.mqtt_connected = False
            st.session_state.mqtt_status = f"❌ Connection failed (code: {rc})"
    
    def on_disconnect(client, userdata, rc):
        """Disconnection callback"""
        st.session_state.mqtt_connected = False
        if rc != 0:
            st.session_state.mqtt_status = f"⚠️ Unexpected disconnect (code: {rc})"
        else:
            st.session_state.mqtt_status = "Disconnected"
    
    def on_message(client, userdata, msg):
        """Message callback - just queue it, no state manipulation"""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            st.session_state.message_queue.put(payload)
        except Exception as e:
            pass  # Silent fail in callback
    
    # Create client (compatible with old API)
    try:
        # Try new API first
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except:
        # Fallback to old API
        client = mqtt.Client()
    
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    return client

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
        index=1,  # Default to MQTT
        key="data_source"
    )
    st.text_input("Asset Name", value=st.session_state.asset_name, key="asset_name")
    
    st.divider()
    st.header("⚙️ Settings")
    update_interval = st.slider("Update interval (ms)", 200, 2000, 500, 100)
    
    if st.button("🧹 Clear All Data"):
        for key in ("live_buffer","live_rpm","live_temp","vibration_axial","vibration_horizontal","vibration_vertical"):
            st.session_state[key].clear()
        st.session_state.data_counter = 0
        st.session_state.last_message = {}
        while not st.session_state.message_queue.empty():
            st.session_state.message_queue.get()
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# Main Dashboard
# ────────────────────────────────────────────────────────────────────────────────
st.title("🟢 Live Monitoring Dashboard")

# ────────────────────────────────────────────────────────────────────────────────
# MODE: Simulated Stream
# ────────────────────────────────────────────────────────────────────────────────
if source == "Simulated Stream":
    st.caption("📡 Simulating live sensor data")
    
    sim_rate = st.slider("Samples per update", 1, 50, 10, 1)
    
    c1, c2, c3 = st.columns(3)
    if c1.button("▶️ Start"):
        st.session_state.live_running = True
        st.rerun()
    
    if c2.button("⏸️ Pause"):
        st.session_state.live_running = False
    
    if c3.button("🔄 Reset"):
        st.session_state.live_running = False
        st.session_state.pop("sim_idx", None)
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
            
            st.session_state.vibration_axial.append(float(xs[j]))
            st.session_state.vibration_horizontal.append(float(ys[j]))
            st.session_state.vibration_vertical.append(float(zs[j]))
            
            rms = math.sqrt((xs[j]**2 + ys[j]**2 + zs[j]**2) / 3.0)
            st.session_state.live_buffer.append(rms)
            st.session_state.live_rpm.append(rpm)
            st.session_state.live_temp.append(temp)
            st.session_state.data_counter += 1
        
        st.session_state.sim_idx = i1
        time.sleep(update_interval/1000.0)
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# MODE: MQTT Live
# ────────────────────────────────────────────────────────────────────────────────
elif source == "MQTT Live":
    st.caption("🌍 **Broker:** broker.hivemq.com:1883 | **Topic:** machine/vibration/data")
    
    if not MQTT_OK:
        st.error("❌ Install MQTT: `pip install paho-mqtt`")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔌 Connect", type="primary", disabled=st.session_state.mqtt_connected):
                try:
                    # Create new client
                    client = create_mqtt_client()
                    client.connect("broker.hivemq.com", 1883, 60)
                    client.loop_start()
                    
                    st.session_state.mqtt_client = client
                    st.session_state.connection_attempts += 1
                    st.session_state.mqtt_status = "Connecting..."
                    
                    time.sleep(1)  # Give it a moment
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.mqtt_status = f"❌ Error: {str(e)}"
                    st.session_state.mqtt_connected = False
        
        with col2:
            if st.button("🔕 Disconnect", disabled=not st.session_state.mqtt_connected):
                client = st.session_state.get("mqtt_client")
                if client:
                    try:
                        client.loop_stop()
                        client.disconnect()
                    except:
                        pass
                st.session_state.mqtt_connected = False
                st.session_state.mqtt_status = "Disconnected"
                st.rerun()
        
        with col3:
            st.metric("Connection Attempts", st.session_state.connection_attempts)
        
        # Status display
        st.info(st.session_state.mqtt_status)
        
        # Process queued messages
        if st.session_state.mqtt_connected:
            processed = process_message_queue()
            if processed > 0:
                st.success(f"📨 Processed {processed} messages")
        
        # Display last message
        if st.session_state.last_message:
            with st.expander("📩 Last Received Message", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                msg = st.session_state.last_message
                col_a.metric("X", f"{msg.get('x', 0):.4f}")
                col_b.metric("Y", f"{msg.get('y', 0):.4f}")
                col_c.metric("Z", f"{msg.get('z', 0):.4f}")
                
                col_d, col_e = st.columns(2)
                col_d.metric("RPM", f"{msg.get('rpm', 0):.2f}")
                col_e.metric("Temp", f"{msg.get('temp', 0):.2f}")
        
        # Auto-refresh when connected
        if st.session_state.mqtt_connected:
            time.sleep(update_interval / 1000.0)
            st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# Data Visualization (Common)
# ────────────────────────────────────────────────────────────────────────────────
st.divider()

# Controls
left, right = st.columns(2)
with left:
    sensor_choice = st.selectbox(
        "Signal to Display",
        ["Vibration (RMS)", "Vibration X", "Vibration Y", "Vibration Z", "RPM", "Temperature"]
    )
with right:
    max_points = st.slider("Display points", 50, 1000, 200, 50)

threshold = st.slider("Alert Threshold", 0.0, 2.0, 0.8, 0.05)

# Select data series
def get_series():
    if sensor_choice == "Vibration (RMS)":
        return np.array(list(st.session_state.live_buffer))
    elif sensor_choice == "Vibration X":
        return np.array(list(st.session_state.vibration_axial))
    elif sensor_choice == "Vibration Y":
        return np.array(list(st.session_state.vibration_horizontal))
    elif sensor_choice == "Vibration Z":
        return np.array(list(st.session_state.vibration_vertical))
    elif sensor_choice == "RPM":
        return np.array(list(st.session_state.live_rpm))
    elif sensor_choice == "Temperature":
        return np.array(list(st.session_state.live_temp))
    return np.array([])

series = get_series()[-max_points:]

# KPIs
k1, k2, k3, k4 = st.columns(4)

if series.size > 0:
    current_val = float(series[-1])
    avg_val = float(np.mean(series))
    max_val = float(np.max(series))
    min_val = float(np.min(series))
    
    k1.metric("Current", f"{current_val:.4f}", 
              delta=f"{current_val - avg_val:.4f}")
    k2.metric("Average", f"{avg_val:.4f}")
    k3.metric("Range", f"{max_val - min_val:.4f}")
    k4.metric("Total Samples", st.session_state.data_counter)
    
    # Anomaly detection for vibration
    if "Vibration" in sensor_choice and series.size >= 32:
        fused, dec, thr = score_live_window(series, model)
        faults = int((dec == 1).sum()) if dec.size else 0
        
        if faults > 0:
            st.error(f"🚨 {faults} anomalies detected in window!")
    
    # Chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=series,
        mode="lines",
        name=sensor_choice,
        line=dict(color="#1f77b4", width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        opacity=0.6,
        annotation_text=f"Threshold: {threshold}"
    )
    
    # Highlight over-threshold points
    over_threshold = series > threshold
    if np.any(over_threshold):
        indices = np.where(over_threshold)[0]
        fig.add_trace(go.Scatter(
            x=indices,
            y=series[indices],
            mode="markers",
            marker=dict(color="red", size=8, symbol="x"),
            name="Alert"
        ))
    
    fig.update_layout(
        title=f"{sensor_choice} — Live Stream (Total: {st.session_state.data_counter} samples)",
        xaxis_title="Sample Index",
        yaxis_title="Value",
        template="plotly_white",
        height=450,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 Statistics")
        stats_df = pd.DataFrame({
            "Metric": ["Mean", "Std Dev", "Min", "Max", "Median"],
            "Value": [
                f"{avg_val:.4f}",
                f"{float(np.std(series)):.4f}",
                f"{min_val:.4f}",
                f"{max_val:.4f}",
                f"{float(np.median(series)):.4f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    with col_right:
        st.subheader("📈 Recent Values")
        recent = series[-10:][::-1]  # Last 10, newest first
        recent_df = pd.DataFrame({
            "Index": range(len(recent)),
            "Value": [f"{v:.4f}" for v in recent]
        })
        st.dataframe(recent_df, hide_index=True, use_container_width=True)
    
else:
    st.info("⏳ Waiting for data...")
    st.write("**To start:**")
    if source == "Simulated Stream":
        st.write("1. Click '▶️ Start' button above")
    else:
        st.write("1. Click '🔌 Connect' button above")
        st.write("2. Make sure your data publisher is running")
        st.write("3. Check that publisher is sending to: `broker.hivemq.com:1883`")
    
    k1.metric("Samples", 0)
    k2.metric("Status", "No Data")
    k3.metric("Mode", source)
    k4.metric("Ready", "✓" if MQTT_OK else "✗")
