"""
src/risk_engine.py
===================
Combines the current state of all detectors into an overall risk level.

This is a POC rule-based engine, NOT a scientifically validated safety
model. Rules are intentionally simple and fully configurable via
config.RISK_RULES so they can be tuned or replaced later (e.g. with a
learned model) without touching the rest of the pipeline.
"""

import config

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def compute_driver_risk(drowsy_active: bool, phone_active: bool,
                         distraction_active: bool, looking_away_active: bool) -> str:
    level = config.RISK_RULES["base"]

    if looking_away_active:
        level = _max_level(level, config.RISK_RULES["looking_away_sustained"])
    if distraction_active:
        level = _max_level(level, config.RISK_RULES["distraction"])
    if phone_active:
        level = _max_level(level, config.RISK_RULES["phone"])
    if drowsy_active:
        level = _max_level(level, config.RISK_RULES["drowsiness"])

    return level


def compute_combined_risk(drowsy_active: bool, phone_active: bool,
                           distraction_active: bool, looking_away_active: bool,
                           lane_departure_active: bool = False) -> str:
    level = compute_driver_risk(drowsy_active, phone_active, distraction_active, looking_away_active)

    if lane_departure_active:
        level = _max_level(level, config.RISK_RULES["lane_departure"])

    if drowsy_active and lane_departure_active:
        level = _max_level(level, config.RISK_RULES["drowsiness+lane_departure"])
    if phone_active and lane_departure_active:
        level = _max_level(level, config.RISK_RULES["phone+lane_departure"])

    return level


def _max_level(a: str, b: str) -> str:
    return a if RISK_ORDER[a] >= RISK_ORDER[b] else b
