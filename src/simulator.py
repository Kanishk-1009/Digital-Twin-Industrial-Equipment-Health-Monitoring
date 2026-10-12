"""Live digital-twin simulator — publishes 1 Hz telemetry over MQTT.

Usage:
    python -m src.simulator          # real-time
    python -m src.simulator 2        # 2x speed

Publishes to: telemetry, truth, status topics
Subscribes to: cmd topic (inject/repair commands)
"""

import json
import sys
import time

import paho.mqtt.client as mqtt

from src.config import (
    MQTT_HOST, MQTT_PORT,
    TOPIC_TELEMETRY, TOPIC_TRUTH, TOPIC_CMD, TOPIC_STATUS,
)
from src.asset_model import PumpTwin

twin  = PumpTwin(seed=42)
speed = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
sleep_s = 1.0 / speed


def on_connect(client, userdata, flags, rc, properties):
    print(f"[simulator] connected to broker, rc={rc}")
    client.subscribe(TOPIC_CMD, qos=1)
    client.publish(TOPIC_STATUS, "online", retain=True)


def on_message(client, userdata, msg):
    try:
        cmd = json.loads(msg.payload)
    except (ValueError, TypeError):
        print("[simulator] bad cmd payload, ignoring")
        return

    action = cmd.get("action")
    if action == "inject":
        fault      = cmd.get("fault", "")
        rate_mult  = float(cmd.get("rate_mult", 1.0))
        twin.set_fault(fault, rate_mult)
        print(f"[simulator] FAULT INJECTED: {cmd}")
    elif action == "repair":
        twin.repair()
        print("[simulator] REPAIRED, all severities reset")
    else:
        print(f"[simulator] unknown action: {action}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pump01-sim")
client.will_set(TOPIC_STATUS, "offline", retain=True)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, MQTT_PORT)
client.loop_start()

print(f"[simulator] starting at {speed}x speed (sleep={sleep_s:.3f} s/step)")
try:
    while True:
        tel, truth = twin.step()
        client.publish(TOPIC_TELEMETRY, json.dumps(tel),   qos=1)
        client.publish(TOPIC_TRUTH,     json.dumps(truth), qos=1)
        time.sleep(sleep_s)
except KeyboardInterrupt:
    print("\n[simulator] shutting down")
finally:
    client.publish(TOPIC_STATUS, "offline", retain=True)
    client.loop_stop()
    client.disconnect()
