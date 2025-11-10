import json, time, math, queue
from collections import deque
from threading import Lock

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import paho.mqtt.client as mqtt
    MQTT_OK = True
except:
    MQTT_OK = False

# ------------------ CONFIG -------------------
TOPIC = "machine/vibration/data"
BROKER = "broker.hivemq.com"
PORT   = 1883
MAX_LEN = 3000     # how many samples buffer holds
UPDATE_MS = 400    # UI refresh speed (ms)
# ---------------------------------------------

st.set_page_config(page_title="Live Machine Health Dashboard", layout="wide")

# ------------------ STATE INIT ----------------
if "queue" not in st.session_state:
    st.session_state.queue = queue.Queue()

def init_buffer(key):
    if key not in st.session_state:
        st.session_state[key] = deque(maxlen=MAX_LEN)

for k in ["ax","ay","az","rms","rpm","temp"]:
    init_buffer(k)

if "connected" not in st.session_state:
    st.session_state.connected = False
if "last_msg" not in st.session_state:
    st.session_state.last_msg = {}

# ------------------ DATA UPDATE ----------------
def push(msg):
    try:
        x = float(msg.get("x", 0))
        y = float(msg.get("y", 0))
        z = float(msg.get("z", 0))
        rpm = float(msg.get("rpm", 0))
        temp = float(msg.get("temp", 0))

        st.session_state.ax.append(x)
        st.session_state.ay.append(y)
        st.session_state.az.append(z)

        rms = math.sqrt((x*x + y*y + z*z) / 3)
        st.session_state.rms.append(rms)
        st.session_state.rpm.append(rpm)
        st.session_state.temp.append(temp)

        st.session_state.last_msg = msg

    except Exception:
        pass


def process_queue():
    while not st.session_state.queue.empty():
        msg = st.session_state.queue.get()
        push(msg)

# ------------------ MQTT CLIENT ----------------
def create_client():
    def on_connect(client, userdata, flags, rc, *args):
        if rc == 0:
            st.session_state.connected = True
            client.subscribe(TOPIC)
        else:
            st.session_state.connected = False

    def on_message(client, userdata, msg):
        try:
            st.session_state.queue.put(json.loads(msg.payload.decode()))
        except:
            pass

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except:
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message
    return client


# ------------------ UI ----------------
st.title("🟢 Real-Time Machine Condition Dashboard")

col1, col2 = st.columns(2)
with col1:
    if st.button("🔌 Connect", disabled=st.session_state.connected):
        try:
            client = create_client()
            client.connect(BROKER, PORT, 60)
            client.loop_start()
            st.session_state.mqtt = client
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Connection error: {e}")

with col2:
    if st.button("🔕 Disconnect", disabled=not st.session_state.connected):
        c = st.session_state.get("mqtt")
        if c:
            c.loop_stop()
            c.disconnect()
        st.session_state.connected = False
        st.rerun()

st.write(f"**Status:** {'✅ Connected' if st.session_state.connected else '❌ Not Connected'}")

# Process MQTT incoming messages
process_queue()

# ------------------ CHARTS ----------------
sensor = st.selectbox("Select Signal", ["RMS Vibration", "X Axis", "Y Axis", "Z Axis", "RPM", "Temperature"])

data_map = {
    "RMS Vibration": st.session_state.rms,
    "X Axis": st.session_state.ax,
    "Y Axis": st.session_state.ay,
    "Z Axis": st.session_state.az,
    "RPM": st.session_state.rpm,
    "Temperature": st.session_state.temp,
}

series = np.array(data_map[sensor])

if len(series) > 5:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series, mode="lines", line=dict(width=2)))
    fig.update_layout(
        title=f"{sensor} — Live Stream",
        xaxis_title="Sample Index",
        yaxis_title=sensor,
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Waiting for data...")

if st.session_state.connected:
    time.sleep(UPDATE_MS / 1000)
    st.rerun()
