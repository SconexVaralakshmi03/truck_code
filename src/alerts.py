"""
src/alerts.py
==============
Simulated alert actions for the POC. None of these touch real hardware,
send real notifications, or integrate with any external system — they
print what WOULD happen, and return a small dict describing the simulated
action so it can also be shown on the video overlay / logged.

Wiring these to a real buzzer, vibration motor, or SMS/push notification
service is explicitly out of scope for V1 (see prompt rule #21).
"""

import time


def trigger_driver_alarm(reason: str) -> dict:
    msg = f"[DRIVER ALARM] {reason}"
    print(msg)
    return {"channel": "driver_alarm", "state": "ON", "reason": reason, "time": time.time()}


def trigger_digital_ring() -> dict:
    msg = "[DIGITAL RING] VIBRATION TRIGGERED (simulated)"
    print(msg)
    return {"channel": "digital_ring", "state": "ON", "time": time.time()}


def notify_owner(reason: str) -> dict:
    msg = f"[OWNER ALERT] NOTIFICATION GENERATED (simulated): {reason}"
    print(msg)
    return {"channel": "owner_alert", "state": "SENT", "reason": reason, "time": time.time()}


def clear_alarms() -> dict:
    return {"driver_alarm": "OFF", "digital_ring": "OFF", "owner_alert": "OFF"}


def fire_all(reason: str, severity: str) -> dict:
    """
    Convenience helper: fires all three simulated channels for HIGH/CRITICAL
    severity events, matching the spec's example output block.
    """
    state = {"driver_alarm": "OFF", "digital_ring": "OFF", "owner_alert": "OFF"}
    if severity in ("HIGH", "CRITICAL"):
        trigger_driver_alarm(reason)
        trigger_digital_ring()
        notify_owner(reason)
        state = {"driver_alarm": "ON", "digital_ring": "ON", "owner_alert": "SENT"}
    return state
