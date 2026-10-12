"""Fault injection / repair CLI.

Usage:
    python -m src.control inject bearing_wear
    python -m src.control inject imbalance 1.5
    python -m src.control inject cavitation 2.0
    python -m src.control inject cooling_failure
    python -m src.control repair
"""

import json
import sys

import paho.mqtt.publish as publish

from src.config import MQTT_HOST, MQTT_PORT, TOPIC_CMD

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

action = sys.argv[1].lower()

if action == "inject":
    if len(sys.argv) < 3:
        print("Usage: python -m src.control inject <fault_name> [rate_mult]")
        sys.exit(1)
    msg = {
        "action":    "inject",
        "fault":     sys.argv[2],
        "rate_mult": float(sys.argv[3]) if len(sys.argv) > 3 else 1.0,
    }
elif action == "repair":
    msg = {"action": "repair"}
else:
    print(f"Unknown action '{action}'. Use 'inject' or 'repair'.")
    sys.exit(1)

publish.single(TOPIC_CMD, json.dumps(msg), hostname=MQTT_HOST, port=MQTT_PORT)
print("Sent:", msg)
