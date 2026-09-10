"""
config.py
=========
Central configuration for the Truck Driver Safety Monitoring POC.

IMPORTANT: These are POC-level thresholds chosen for demonstration purposes.
They are NOT medically validated, NOT safety-certified, and MUST be tuned
against real fleet data before any production use.
"""

import os

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(ROOT_DIR, "models")
DROWSINESS_MODEL_DIR = os.path.join(MODELS_DIR, "drowsiness")
PHONE_MODEL_DIR = os.path.join(MODELS_DIR, "phone")
DISTRACTION_MODEL_DIR = os.path.join(MODELS_DIR, "distraction")
YOLOP_MODEL_DIR = os.path.join(MODELS_DIR, "yolop")

VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")
DRIVER_VIDEOS_DIR = os.path.join(VIDEOS_DIR, "driver")
ROAD_VIDEOS_DIR = os.path.join(VIDEOS_DIR, "road")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

EVENTS_JSON_PATH = os.path.join(LOGS_DIR, "events.json")
EVENTS_CSV_PATH = os.path.join(LOGS_DIR, "events.csv")

# --------------------------------------------------------------------------
# DEVICE
# --------------------------------------------------------------------------
DEFAULT_DEVICE = "cpu"  # overridden by --device cpu|cuda

# --------------------------------------------------------------------------
# DROWSINESS
# --------------------------------------------------------------------------
DROWSINESS_CONFIDENCE = 0.70          # min confidence to count a frame as "drowsy"
DROWSINESS_DURATION = 2.0             # seconds of sustained drowsy frames to trigger event
DROWSINESS_COOLDOWN = 5.0             # seconds before the same event can re-trigger

# Only let the drowsiness *classifier* vote "drowsy" on frames where a face
# was actually found this frame (by the geometric/MediaPipe stage that runs
# right before it). This is what stops a driverless / mispointed camera
# frame (e.g. dash camera briefly showing the ceiling/ door, nobody in
# seat) from being scored as "drowsy" by the .pt classifier, which -- unlike
# the EAR/PERCLOS path -- has no built-in concept of "is there even a face
# here". Also skips running the (comparatively heavy) classifier entirely
# on no-face frames, which is a latency win, not just an accuracy one.
DROWSINESS_REQUIRE_FACE = True

# GRACE PERIOD (real-time robustness): live/WebSocket frames are noisier
# than a clean prerecorded video -- a single dropped frame, a momentary
# blink misread, or one flaky classifier call can flip "drowsy" -> "not
# drowsy" for exactly one frame in the middle of a real, ongoing drowsy
# spell. Previously ANY single non-drowsy frame reset the sustained-timer
# to zero, so a real ~2s drowsy event spanning one bad frame could not
# reliably reach DROWSINESS_DURATION and would silently keep restarting
# instead of firing -- and conversely, once it does start rapid alternation
# reads as repeated START/END "multi-frame" spam. GRACE_PERIOD lets the
# sustained timer tolerate short gaps (<= this many seconds of consecutive
# "condition false" frames) without resetting, while a gap longer than this
# still means "the condition genuinely ended". This does NOT lower
# DROWSINESS_DURATION/CONFIDENCE -- those thresholds are unchanged; this
# only makes reaching them robust to normal frame-to-frame jitter.
DROWSINESS_GRACE_PERIOD = 0.5

# --------------------------------------------------------------------------
# PHONE USAGE
# --------------------------------------------------------------------------
PHONE_CONFIDENCE = 0.70
PHONE_DURATION = 2.0
PHONE_COOLDOWN = 5.0
PHONE_GRACE_PERIOD = 0.5              # same jitter-tolerance idea as above

# --------------------------------------------------------------------------
# DISTRACTION
# --------------------------------------------------------------------------
# Raised from 0.60 -> 0.78 and duration from 1.5s -> 2.0s after real testing
# showed the distraction model calling an ordinary driving frame "distracted"
# at 65.7% confidence (see test_models.py output) — the old threshold of
# 0.60 would have accepted that as a real event. Re-tune further once you
# have more labeled example frames from your own footage.
DISTRACTION_CONFIDENCE = 0.78
DISTRACTION_DURATION = 2.0
DISTRACTION_COOLDOWN = 5.0

# --------------------------------------------------------------------------
# HEAD POSE / LOOKING AWAY (MediaPipe fallback)
# --------------------------------------------------------------------------
# Camera is assumed dash-mounted, straight in front of the driver seat, so
# yaw=0 means facing the road and a single symmetric left/right threshold
# is used (no per-install calibration needed for that part).
#
# "Not focused" is now flagged two ways, per requirement:
#   1. CONTINUOUS: looking away >= HEAD_POSE_AWAY_DURATION seconds straight
#      (raised from 1.5s -> 10s).
#   2. FREQUENT: HEAD_POSE_FREQUENT_GLANCE_COUNT+ separate glances away
#      within a HEAD_POSE_FREQUENT_GLANCE_WINDOW-second rolling window,
#      even if none individually reaches 10s.
HEAD_POSE_AWAY_DURATION = 10.0        # seconds of CONTINUOUS look-away before flagged
HEAD_POSE_YAW_THRESHOLD_DEG = 25.0    # degrees left/right considered "away"
HEAD_POSE_PITCH_THRESHOLD_DEG = 20.0  # degrees up/down considered "away"
HEAD_POSE_FREQUENT_GLANCE_WINDOW = 30.0   # rolling window (s) for "looking away often"
HEAD_POSE_FREQUENT_GLANCE_COUNT = 3       # N+ separate glances within window -> not focused

# --------------------------------------------------------------------------
# DROWSINESS - EYE ASPECT RATIO (EAR) / MOUTH ASPECT RATIO (MAR)
# GEOMETRIC CROSS-CHECK (src/drowsiness_geometric.py)
# --------------------------------------------------------------------------
# Runs independently of the drowsiness .pt classifier. Combined per
# DROWSINESS_COMBINE_MODE: "OR" (either signal triggers, catches more real
# cases given the classifier was observed to be noisy/inconsistent on real
# test footage) or "AND" (both must agree, more conservative).
DROWSINESS_COMBINE_MODE = "OR"

# EAR_CLOSED_THRESHOLD_FLOOR is a fixed sanity floor used until enough
# samples exist to compute a reliable adaptive baseline (see
# EAR_BASELINE_* below) -- real testing against two different camera
# setups showed open-eye EAR baselines ranging ~0.20 (distant/lower-res
# camera) to ~0.28 (closer camera), so a single fixed threshold alone is
# not portable across setups.
EAR_CLOSED_THRESHOLD_FLOOR = 0.15
EAR_CLOSED_DURATION = 1.5             # seconds of sustained closed-eyes to flag drowsy
EAR_COOLDOWN = 5.0
EAR_GRACE_PERIOD = 0.4                # tolerate brief single-frame blink/tracking noise
EAR_BASELINE_WINDOW_SAMPLES = 450     # rolling window of recent "eyes open" EAR readings
EAR_BASELINE_MIN_SAMPLES = 60         # need this many samples before trusting the adaptive baseline
EAR_BASELINE_PERCENTILE = 90          # percentile of recent history used as "this person's open-eye EAR"
EAR_CLOSED_RATIO = 0.72               # flag closed when EAR drops below (baseline * this ratio)

MAR_YAWN_THRESHOLD = 0.55             # mouth-height/width ratio above this = wide open (yawn)
MAR_YAWN_DURATION = 1.5

# PERCLOS (PERcentage of eye CLOSure) -- the actual industry-standard
# drowsiness metric (Wierwille et al.), not a POC invention: percentage of
# time the eyes are closed over a rolling window, rather than a single
# "closed right now for N seconds" check. Catches heavy-lidded/frequent-
# long-blink patterns that never individually reach EAR_CLOSED_DURATION but
# still indicate drowsiness in aggregate.
PERCLOS_WINDOW_SECONDS = 60.0    # rolling window over which % closed is measured
PERCLOS_THRESHOLD = 0.15         # >=15% of the window spent closed -> drowsy (literature-cited value)
PERCLOS_MIN_WINDOW_COVERAGE = 0.5  # need at least half the window's worth of samples before trusting it
PERCLOS_COOLDOWN = 10.0

# --------------------------------------------------------------------------
# PHONE - GEOMETRIC GATE (hand-near-ear) + CASCADE
# (src/pose_utils.py, src/phone_detection.py)
# --------------------------------------------------------------------------
PHONE_HAND_EAR_DISTANCE_RATIO = 0.6   # wrist-to-ear distance, as fraction of shoulder width
PHONE_HAND_MOUTH_DISTANCE_RATIO = 0.7 # wrist-to-mouth distance (speakerphone/held-up-to-face posture)
PHONE_ELBOW_BEND_MAX_DEG = 140.0      # elbow angle below this = arm bent/raised
PHONE_YOLO_FALLBACK_MODEL = os.path.join(MODELS_DIR, "general", "yolov8m.pt")
PHONE_YOLO_CONF = 0.35                # confidence needed from the YOLOv8 'cell phone' fallback

# --------------------------------------------------------------------------
# LANE DEPARTURE
# --------------------------------------------------------------------------
# Normalized offset of vehicle/camera center from lane center, as a fraction
# of lane width. Must be tuned per-camera-mounting-geometry.
LANE_OFFSET_THRESHOLD = 0.35
LANE_DEPARTURE_DURATION = 1.0
LANE_DEPARTURE_COOLDOWN = 5.0

# --------------------------------------------------------------------------
# GENERAL TEMPORAL / EVENT SETTINGS
# --------------------------------------------------------------------------
DEFAULT_FRAME_SKIP = 0   # 0 = process every frame

# --------------------------------------------------------------------------
# RISK ENGINE RULES (see src/risk_engine.py for logic)
# --------------------------------------------------------------------------
# These are illustrative POC rules only, not scientifically validated.
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

RISK_RULES = {
    "base": "LOW",
    "looking_away_sustained": "MEDIUM",
    "distraction": "MEDIUM",
    "phone": "HIGH",
    "drowsiness": "HIGH",
    "lane_departure": "MEDIUM",
    # combinations (checked in risk_engine.py) escalate to CRITICAL
    "drowsiness+lane_departure": "CRITICAL",
    "phone+lane_departure": "CRITICAL",
}

# --------------------------------------------------------------------------
# DISPLAY
# --------------------------------------------------------------------------
OVERLAY_FONT_SCALE = 0.55
OVERLAY_THICKNESS = 1
OVERLAY_LINE_HEIGHT = 22