# streamlit_app.py
import json
import time
import math
import queue
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# Optional imports if you have them
try:
    from scipy.stats import kurtosis, skew
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# MQTT client import; graceful fallback
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except Exception:
    MQTT_AVAILABLE = False

# ---------------------------
# Telegram bot (set your values)
# ---------------------------
BOT_TOKEN = "REPLACE_WITH_YOUR_BOT_TOKEN"
CHAT_ID   = "REPLACE_WITH_YOUR_CHAT_ID"

def send_telegram_alert(message: str):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        # Use markdown for readability
        requests.get(url, params={"chat_id": CHAT_ID, "text": message})
    except Exception as e:
        st.session_state.mqtt_last_err = f"Telegram error: {e}"

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Live Predictive Maintenance", layout="wide")

ROOT = Path(".")
DEMO_CSV = ROOT / "data" / "synthetic" / "machine_001_demo.csv"

# ---------------------------
# Helpers: demo data
# ---------------------------
@st.cache_data
def load_demo_dataframe():
    if DEMO_CSV.exists():
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

# ---------------------------
# Session-state init
# ---------------------------
if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue()

for k in ("vibration_x","vibration_y","vibration_z","rms_buffer","rpm_buffer","temp_buffer"):
    if k not in st.session_state:
        st.session_state[k] = deque(maxlen=6000)

if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False
if "mqtt_last_err" not in st.session_state:
    st.session_state.mqtt_last_err = ""

# Alert log (last 10)
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []  # list of (timestamp, message)
if "alert_mode" not in st.session_state:
    st.session_state.alert_mode = False
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = datetime.min

# ---------------------------
# UI - Sidebar
# ---------------------------
with st.sidebar:
    st.title("Data Source & Controls")
    mode = st.radio("Mode", ["Simulated Stream", "MQTT Live"])
    st.markdown("---")
    update_interval = st.slider("Update interval (ms)", 100, 2000, 500, 50)
    display_points = st.slider("Display points", 100, 2000, 500, 50)
    st.markdown("---")
    st.write("Telegram bot: set token & chat id in file if you want alerts.")

# ---------------------------
# MQTT client creation (non-UI callbacks)
# ---------------------------
def create_mqtt_client(broker="broker.hivemq.com", port=1883, topic="machine/vibration/data"):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            st.session_state.mqtt_connected = True
            client.subscribe(topic, qos=0)
        else:
            st.session_state.mqtt_last_err = f"MQTT connect failed rc={rc}"

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            st.session_state.message_queue.put(payload)
        except Exception:
            # ignore malformed payloads
            pass

    # Use default client (TCP)
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(broker, port, 60)
        client.loop_start()
        return client
    except Exception as e:
        st.session_state.mqtt_last_err = f"MQTT create error: {e}"
        return None

# ---------------------------
# Process queued messages (main thread)
# ---------------------------
def process_message_queue():
    processed = 0
    q = st.session_state.message_queue
    while not q.empty():
        try:
            msg = q.get_nowait()
            # support both formats
            x = float(msg.get("x", msg.get("axial", 0.0)))
            y = float(msg.get("y", msg.get("horizontal", 0.0)))
            z = float(msg.get("z", msg.get("vertical", 0.0)))
            rpm = float(msg.get("rpm", 1500))
            temp = float(msg.get("temp", 65))

            rms = math.sqrt((x*x + y*y + z*z) / 3.0)

            st.session_state.vibration_x.append(x)
            st.session_state.vibration_y.append(y)
            st.session_state.vibration_z.append(z)
            st.session_state.rms_buffer.append(rms)
            st.session_state.rpm_buffer.append(rpm)
            st.session_state.temp_buffer.append(temp)

            processed += 1
        except Exception:
            break
    return processed

# ---------------------------
# UI layout - main
# ---------------------------
st.header("🛠️ Predictive Maintenance — Live Dashboard")

col1, col2 = st.columns([3,1])
with col2:
    st.subheader("Status")
    st.write(f"MQTT connected: {st.session_state.mqtt_connected}")
    if st.session_state.mqtt_last_err:
        st.error(st.session_state.mqtt_last_err)

# Connect / Disconnect controls for MQTT
if mode == "MQTT Live":
    if not MQTT_AVAILABLE:
        st.error("paho-mqtt not installed. Install: pip install paho-mqtt")
    else:
        c1, c2 = st.columns(2)
        if c1.button("🔌 Connect (MQTT)", disabled=st.session_state.mqtt_connected):
            client = create_mqtt_client()
            if client:
                st.session_state.mqtt_client = client
                st.success("MQTT client started")
        if c2.button("🔕 Disconnect", disabled=not st.session_state.mqtt_connected):
            c = st.session_state.get("mqtt_client")
            if c:
                try:
                    c.loop_stop(); c.disconnect()
                except Exception:
                    pass
            st.session_state.mqtt_client = None
            st.session_state.mqtt_connected = False
            st.success("Disconnected")

# Simulated stream controls
if mode == "Simulated Stream":
    demo = load_demo_dataframe()
    sim_rate = st.slider("Sim samples per tick", 1, 200, 20)
    s1, s2 = st.columns(2)
    if s1.button("▶️ Start simulation"):
        st.session_state.sim_running = True
    if s2.button("⏸️ Stop simulation"):
        st.session_state.sim_running = False

    if st.session_state.get("sim_running", False):
        # push sim_rate messages into queue (non-blocking)
        if {"x","y","z"}.issubset(demo.columns):
            xs, ys, zs = demo["x"].values, demo["y"].values, demo["z"].values
        else:
            rms = demo["vibration_rms"].values
            xs, ys, zs = rms*0.97, rms*1.01, rms*1.03

        if "sim_idx" not in st.session_state:
            st.session_state.sim_idx = 0
        for _ in range(sim_rate):
            j = st.session_state.sim_idx % len(xs)
            # For testing anomalies you can scale z sometimes
            z_val = float(zs[j])
            # optional forced anomaly every 500 samples:
            if (st.session_state.sim_idx % 500) == 0:
                z_val = z_val * 3.5
            msg = {"x": float(xs[j]), "y": float(ys[j]), "z": z_val,
                   "rpm": 1500 + 10*np.sin(j/200), "temp": 65 + 1*np.sin(j/350)}
            st.session_state.message_queue.put(msg)
            st.session_state.sim_idx += 1

# Process incoming messages into buffers
processed = process_message_queue()
if processed:
    st.info(f"Processed {processed} messages", icon="📨")

# Build series (most recent values)
def get_series(name):
    if name == "RMS":
        return np.array(st.session_state.rms_buffer)
    if name == "X":
        return np.array(st.session_state.vibration_x)
    if name == "Y":
        return np.array(st.session_state.vibration_y)
    if name == "Z":
        return np.array(st.session_state.vibration_z)
    if name == "RPM":
        return np.array(st.session_state.rpm_buffer)
    if name == "Temp":
        return np.array(st.session_state.temp_buffer)
    return np.array([])

# Plot area
plot_col, alert_col = st.columns([3,1])
with plot_col:
    series_choice = st.selectbox("Plot series", ["RMS","X","Y","Z","RPM","Temp"])
    series = get_series(series_choice)[-display_points:]
    if series.size:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series, mode="lines", name=series_choice))
        fig.update_layout(title=f"{series_choice} — Live", xaxis_title="Samples (newest → right)",
                          yaxis_title="Value", template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet (start simulator or connect MQTT)")

# Alerts: basic RMS thresholding with cooldown + last-10 log
with alert_col:
    st.subheader("Alerts")
    alert_thresh = st.number_input("RMS alert threshold", value=0.85, step=0.05)
    cooldown = timedelta(seconds=30)

    current_rms = float(st.session_state.rms_buffer[-1]) if len(st.session_state.rms_buffer) else 0.0
    now = datetime.utcnow()

    # trigger alert
    if current_rms > alert_thresh and (now - st.session_state.last_alert_time) > cooldown:
        msg = f"🚨 ALERT | RMS={current_rms:.4f} > {alert_thresh} | {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        send_telegram_alert(msg)
        st.session_state.alert_log.append((now, msg))
        st.session_state.alert_log = st.session_state.alert_log[-10:]
        st.session_state.last_alert_time = now
        st.success("Alert sent")

    # display last 10 alerts
    if st.session_state.alert_log:
        df_alerts = pd.DataFrame([(t.strftime("%Y-%m-%d %H:%M:%S"), m) for t, m in st.session_state.alert_log],
                                 columns=["Timestamp", "Message"])
        st.table(df_alerts)
    else:
        st.write("No alerts")

# small info panel
st.sidebar.write("---")
st.sidebar.metric("Buffered samples (RMS)", len(st.session_state.rms_buffer))

# refresh loop (do not block long)
time.sleep(update_interval / 1000.0)
# trigger a refresh without hard rerun (works reliably)
st.experimental_set_query_params(ts=str(time.time()))



