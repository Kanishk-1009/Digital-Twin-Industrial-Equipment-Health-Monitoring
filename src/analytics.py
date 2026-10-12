"""Real-time analytics service — ML inference + InfluxDB persistence.

Subscribes to telemetry and truth MQTT topics, maintains a 30-sample
sliding window, computes health index / fault class / RUL, then
publishes results to the health topic and writes to InfluxDB.

Run with:  python -m src.analytics
Start this BEFORE the simulator so the window fills from the first sample.
"""

import json
import warnings
import joblib
import numpy as np
from collections import deque

# suppress sklearn feature-name warnings (we pass numpy arrays, not DataFrames)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from src.config import (
    MQTT_HOST, MQTT_PORT,
    INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET,
    ASSET_ID, SIGNALS, WINDOW, FAIL_THRESHOLD, MODEL_PATH,
    TOPIC_TELEMETRY, TOPIC_TRUTH, TOPIC_HEALTH,
)
from src.features import window_features

# load trained models
M = joblib.load(MODEL_PATH)
scaler, iso, clf = M["scaler"], M["iso"], M["clf"]
s_good, s_bad    = M["s_good"], M["s_bad"]

# runtime state
buf   = deque(maxlen=WINDOW)   # raw sample window
hist  = deque(maxlen=90)       # health history for RUL trend
state = {"ema": None}          # EMA-smoothed health index

# influx client
influx  = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
writer  = influx.write_api(write_options=SYNCHRONOUS)

ALERT_LEVELS = ["OK", "WARNING", "CRITICAL"]


def alert_level(h: float) -> int:
    """0=OK, 1=WARNING, 2=CRITICAL based on health index thresholds."""
    return 0 if h >= 70 else (1 if h >= 40 else 2)


def estimate_rul(h_hist) -> float:
    """Extrapolate health trend to failure threshold. Returns -1 if stable."""
    if len(h_hist) < 45:
        return -1.0
    y     = np.array(h_hist)
    slope = np.polyfit(np.arange(len(y)), y, 1)[0]
    if slope > -0.05:
        return -1.0
    return float(max(0.0, (y[-10:].mean() - FAIL_THRESHOLD) / -slope))


def on_connect(client, userdata, flags, rc, properties):
    print(f"[analytics] connected to broker, rc={rc}")
    client.subscribe([(TOPIC_TELEMETRY, 1), (TOPIC_TRUTH, 1)])


def on_message(client, userdata, msg):
    d = json.loads(msg.payload)

    try:
        # truth messages go straight to InfluxDB for the validation panel
        if msg.topic == TOPIC_TRUTH:
            pt = (
                Point("truth")
                .tag("asset_id", ASSET_ID)
                .field("health_true", float(d["health_true"]))
                .field("severity",    float(d["severity"]))
                .field("fault_true",  str(d["fault"]))
                .field("load",        float(d["load"]))
            )
            writer.write(INFLUX_BUCKET, INFLUX_ORG, pt)
            return

        # telemetry: persist raw values
        pt = Point("telemetry").tag("asset_id", ASSET_ID)
        for s in SIGNALS:
            pt = pt.field(s, float(d[s]))
        writer.write(INFLUX_BUCKET, INFLUX_ORG, pt)

        # update sliding window
        buf.append([d[s] for s in SIGNALS])
        if len(buf) < WINDOW:
            return

        # feature extraction + inference
        x = window_features(buf).reshape(1, -1)

        # anomaly score -> health index (EMA smoothed)
        score = float(iso.decision_function(scaler.transform(x))[0])
        raw   = float(np.clip((score - s_bad) / (s_good - s_bad), 0, 1) * 100)
        state["ema"] = raw if state["ema"] is None else 0.8 * state["ema"] + 0.2 * raw
        health = state["ema"]

        # fault classification
        proba = clf.predict_proba(x)[0]
        k     = int(proba.argmax())
        fault = str(clf.classes_[k])
        conf  = float(proba[k])

        # RUL + alert
        hist.append(health)
        rul = estimate_rul(hist)
        lvl = alert_level(health)

        # publish health JSON
        out = {
            "ts":            d["ts"],
            "asset_id":      ASSET_ID,
            "health_index":  round(health, 1),
            "anomaly_score": round(score, 4),
            "fault_class":   fault,
            "confidence":    round(conf, 3),
            "rul_s":         round(rul, 0),
            "alert":         ALERT_LEVELS[lvl],
        }
        client.publish(TOPIC_HEALTH, json.dumps(out), qos=1)

        # persist to InfluxDB
        writer.write(
            INFLUX_BUCKET, INFLUX_ORG,
            Point("health")
            .tag("asset_id", ASSET_ID)
            .field("health_index",  health)
            .field("anomaly_score", score)
            .field("fault_class",   fault)
            .field("confidence",    conf)
            .field("rul_s",         rul)
            .field("alert_level",   float(lvl)),
        )
        print(out)

    except Exception as e:
        print(f"[analytics] error: {e}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pump01-analytics")
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, MQTT_PORT)
print("[analytics] waiting for telemetry...")
client.loop_forever()
