"""
src/event_manager.py
=====================
Implements the temporal state machine required by the spec:

    CONDITION_MET (per-frame, e.g. drowsy conf > threshold)
            v
    sustained for >= DURATION seconds
            v
    EVENT START  -> logged once
            v
    EVENT ACTIVE (no repeated logging while still true)
            v
    condition no longer met
            v
    EVENT END -> logged once
            v
    COOLDOWN before the same event type can START again

This is what prevents "DROWSINESS / DROWSINESS / DROWSINESS" spam and
instead produces one clean START/END pair per real occurrence.
"""

import os
import json
import csv
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from src.video_utils import format_timestamp


@dataclass
class TemporalFlag:
    """
    Tracks one boolean condition (e.g. "is this frame drowsy?") over time
    and decides when it has been sustained long enough to be a real event,
    with a cooldown before it can fire again.
    """
    name: str
    duration_required: float     # seconds condition must hold to trigger START
    cooldown: float               # seconds after END before it can START again
    # Seconds of consecutive "condition_true=False" frames that are
    # tolerated WITHOUT resetting the sustained-duration timer. 0.0 keeps
    # the original strict "any single false frame resets it" behavior.
    # Non-zero makes the timer robust to the kind of one-frame flicker a
    # live/WebSocket feed produces (dropped frame, momentary misread) that
    # would otherwise keep a real, ongoing condition from ever reaching
    # duration_required, or cause rapid START/END flapping once it does.
    grace_period: float = 0.0

    _condition_since: Optional[float] = field(default=None, init=False)
    _false_since: Optional[float] = field(default=None, init=False)
    _active: bool = field(default=False, init=False)
    _last_end_time: Optional[float] = field(default=None, init=False)
    _last_confidence: float = field(default=0.0, init=False)
    _peak_confidence: float = field(default=0.0, init=False)

    def update(self, condition_true: bool, video_time: float, confidence: float = 0.0):
        """
        Call once per processed frame.
        Returns one of: None, "START", "END"
        """
        if condition_true:
            self._false_since = None  # any true frame clears a pending gap
            self._last_confidence = confidence
            self._peak_confidence = max(self._peak_confidence, confidence)

            if self._condition_since is None:
                self._condition_since = video_time

            sustained = video_time - self._condition_since

            if not self._active and sustained >= self.duration_required:
                cooldown_ok = (
                    self._last_end_time is None
                    or (video_time - self._last_end_time) >= self.cooldown
                )
                if cooldown_ok:
                    self._active = True
                    return "START"
            return None
        else:
            if self.grace_period > 0 and self._condition_since is not None:
                # Within grace: treat this false frame as a blip, not a
                # real end -- keep the sustained-timer running instead of
                # resetting it, as long as the false streak itself stays
                # within grace_period.
                if self._false_since is None:
                    self._false_since = video_time
                if video_time - self._false_since < self.grace_period:
                    return None
                # Gap outlasted the grace period -- genuinely ended.

            self._condition_since = None
            self._false_since = None
            if self._active:
                self._active = False
                self._last_end_time = video_time
                return "END"
            return None

    @property
    def is_active(self):
        return self._active

    @property
    def confidence(self):
        return self._last_confidence

    @property
    def peak_confidence(self):
        return self._peak_confidence

    @property
    def sustained_duration(self):
        if self._condition_since is None:
            return 0.0
        return None  # caller should compute using current video_time - _condition_since if needed


class EventManager:
    """
    Collects START/END events from any number of TemporalFlag instances and
    writes them to logs/events.json and logs/events.csv.
    """

    FIELDNAMES = [
        "timestamp", "video_time", "event", "state", "severity",
        "confidence", "duration", "source",
    ]

    def __init__(self, json_path: str, csv_path: str, source_label: str):
        self.json_path = json_path
        self.csv_path = csv_path
        self.source_label = source_label
        self.events: List[Dict[str, Any]] = []
        self._event_start_times: Dict[str, float] = {}

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Start CSV fresh with header for this run; JSON accumulates a list.
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r") as f:
                    existing = json.load(f)
                    if isinstance(existing, list):
                        self.events = existing
            except Exception:
                self.events = []

    def record(self, event_name: str, state: str, video_time: float,
               severity: str = "INFO", confidence: float = 0.0):
        """
        state should be "START" or "END".
        Duration is only computed and stored on END.
        """
        duration = 0.0
        if state == "START":
            self._event_start_times[event_name] = video_time
        elif state == "END":
            start = self._event_start_times.get(event_name, video_time)
            duration = round(video_time - start, 2)

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "video_time": format_timestamp(video_time),
            "event": event_name,
            "state": state,
            "severity": severity,
            "confidence": round(float(confidence), 3),
            "duration": duration,
            "source": self.source_label,
        }
        self.events.append(record)
        self._append_csv(record)
        self._flush_json()
        return record

    def _append_csv(self, record):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(record)

    def _flush_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.events, f, indent=2)

    def summary_counts(self):
        counts = {}
        for e in self.events:
            if e["state"] == "START":
                counts[e["event"]] = counts.get(e["event"], 0) + 1
        return counts