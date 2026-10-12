import numpy as np
from datetime import datetime, timezone
from src.config import ASSET_ID

# severity gained per second at rate_mult = 1
FAULT_RATES = {
    "bearing_wear":    0.004,
    "imbalance":       0.005,
    "cavitation":      0.005,
    "cooling_failure": 0.004,
}


class PumpTwin:
    """Physics-inspired digital twin of a motor-pump asset.

    Generates multi-sensor telemetry with optional fault injection.
    Each fault accumulates severity (0 -> 1) over time. Sensor readings
    are deterministic functions of load and severity plus Gaussian noise.
    Temperature uses a first-order thermal lag (tau ~ 40s) for realism.
    """

    def __init__(self, seed=None, dt=1.0):
        self.rng   = np.random.default_rng(seed)
        self.dt    = dt
        self.t     = 0.0
        self.sev   = {k: 0.0 for k in FAULT_RATES}
        self.active    = None
        self.rate_mult = 1.0
        self.temp  = 55.0   # initial bearing temperature
        self.phase = self.rng.uniform(0, 2 * np.pi)

    def set_fault(self, name, rate_mult=1.0):
        """Activate a fault type. Ignored if name is invalid."""
        if name in FAULT_RATES:
            self.active, self.rate_mult = name, rate_mult
        else:
            self.active = None

    def repair(self):
        """Clear all faults and reset severities."""
        self.active = None
        self.sev = {k: 0.0 for k in FAULT_RATES}

    def step(self):
        """Advance by one time-step, returns (telemetry_dict, truth_dict)."""
        d, n = self.dt, self.rng.normal
        self.t += d

        # advance active fault severity
        if self.active:
            k = self.active
            self.sev[k] = min(1.0, self.sev[k] + FAULT_RATES[k] * self.rate_mult * d)

        w = self.sev["bearing_wear"]
        i = self.sev["imbalance"]
        c = self.sev["cavitation"]
        k = self.sev["cooling_failure"]

        # process load: slow sinusoidal cycle + noise
        L = float(np.clip(
            0.7 + 0.2 * np.sin(2 * np.pi * self.t / 300 + self.phase) + n(0, 0.01),
            0.4, 1.0
        ))

        # sensor physics
        rpm      = 1480 - 30 * L - 20 * c + n(0, 3 + 6 * i)
        current  = 6 + 12 * L - 2.5 * c + 2.0 * w + n(0, 0.25 + 0.6 * c)
        flow     = 60 * L * (1 - 0.4 * c) + n(0, 0.8 + 2 * c)
        pressure = 6 - 1.5 * L - 2.0 * c + n(0, 0.05 + 0.25 * c)
        vib      = (1.0 + 0.8 * L + 5.0 * w + 3.5 * i + 2.5 * c + 1.0 * k
                    + n(0, 0.08 + 0.15 * (w + i + c)))

        # thermal lag: first-order with tau ~ 40s
        target_T = 40 + 22 * L + 25 * w + 6 * i + 35 * k
        self.temp += (target_T - self.temp) * d / 40.0
        temp = self.temp + n(0, 0.2)

        ts = datetime.now(timezone.utc).isoformat()

        tel = {
            "ts": ts, "asset_id": ASSET_ID,
            "rpm": rpm, "current_a": current,
            "flow_m3h": flow, "pressure_bar": pressure,
            "vibration_mms": vib, "temp_c": temp,
        }

        top = max(self.sev, key=self.sev.get)
        s   = self.sev[top]
        truth = {
            "ts": ts,
            "fault":       top if s >= 0.2 else "healthy",
            "severity":    s,
            "health_true": 100 * (1 - s),
            "load":        L,
        }
        return tel, truth
