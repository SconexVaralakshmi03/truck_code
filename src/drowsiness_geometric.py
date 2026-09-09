"""
src/drowsiness_geometric.py
============================
Geometric, classifier-free drowsiness cross-check using MediaPipe Face Mesh
eye/mouth landmarks: Eye Aspect Ratio (EAR, Soukupova & Cech 2016) for
closed-eye detection, Mouth Aspect Ratio (MAR) for yawning, and PERCLOS
(percentage of eye closure over a rolling window) -- the actual industry-
standard drowsiness metric (Wierwille et al.), not a POC invention.

WHY NOT YOLOv8/YOLO26-POSE FOR EYES: pose models report a single (x, y)
point per eye (from COCO's 17 keypoints), with no shape/aperture
information -- there is no way to compute eye-openness from one point.
MediaPipe Face Mesh gives 468 (or 478 with iris refinement) points
including the full eye-contour outline, which is what makes EAR
computable at all. Face Mesh is the correct tool here, not a fallback.

THREE SIGNALS, COMBINED (config.DROWSINESS_COMBINE_MODE with the .pt
classifier via combine_drowsy() in the run_*.py scripts):
  1. EYE_EVENT: sustained closed-eyes for >= EAR_CLOSED_DURATION straight
     (catches a sudden microsleep).
  2. PERCLOS_EVENT: % of the last PERCLOS_WINDOW_SECONDS spent with eyes
     closed >= PERCLOS_THRESHOLD (catches frequent long blinks / heavy
     eyelids that never individually cross EAR_CLOSED_DURATION).
  3. YAWN_EVENT: sustained wide-open mouth (MAR) -- a secondary drowsiness
     indicator, not sufficient alone.

ADAPTIVE EAR THRESHOLD, NOT A FIXED GLOBAL ONE: real testing against two
different camera setups showed open-eye EAR baselines ranging from ~0.20
(distant windshield-mounted camera, lower-resolution face) to ~0.28
(closer dash camera) for the SAME "eyes open" state -- a single hardcoded
threshold would false-positive on one setup or miss on the other. Instead
this tracks a rolling baseline (a high percentile of recent EAR values,
robust to occasional blinks) and flags "closed" when EAR drops well below
that person's own recent baseline. A fixed floor is used until enough
samples exist, and as a sanity backstop.

Not a replacement for a clinically validated drowsiness system -- POC-grade.
"""

import numpy as np
import cv2
from collections import deque

from src.event_manager import TemporalFlag
import config

try:
    import mediapipe as mp
    _ = mp.solutions.face_mesh
    _MEDIAPIPE_AVAILABLE = True
except Exception:
    _MEDIAPIPE_AVAILABLE = False


# MediaPipe Face Mesh landmark indices for EAR (6-point eye contour each).
# These indices are stable whether or not refine_landmarks/iris is on --
# refinement adds MORE points (the iris ring, 468-477) and refines the
# precision of the existing contour points; it doesn't renumber them.
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # person's right eye (image-left)
_LEFT_EYE = [362, 385, 387, 263, 373, 380]   # person's left eye (image-right)
# Mouth landmarks for MAR
_MOUTH_CORNERS = (61, 291)
_MOUTH_VERTICAL = (13, 14)


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _ear(points_xy):
    p1, p2, p3, p4, p5, p6 = points_xy
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)


class GeometricDrowsinessDetector:
    def __init__(self):
        self.available = _MEDIAPIPE_AVAILABLE
        self.unavailable_reason = None
        if self.available:
            try:
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    # Iris refinement -> 478 landmarks instead of 468, with
                    # higher-precision eye-region tracking (this is the
                    # model MediaPipe itself recommends for anything
                    # eye-related, e.g. their own iris/gaze examples).
                    # Slightly more compute per frame in exchange for more
                    # accurate EAR, which is the "better/more accurate
                    # results" ask -- worth the trade here.
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                self.available = False
                self.unavailable_reason = str(e)
        else:
            self.unavailable_reason = (
                "mediapipe legacy 'solutions.face_mesh' API unavailable. "
                "Pin mediapipe<0.10.14 (see requirements.txt)."
            )

        self.eye_flag = TemporalFlag(
            name="EAR_EYES_CLOSED",
            duration_required=config.EAR_CLOSED_DURATION,
            cooldown=config.EAR_COOLDOWN,
        )
        self.yawn_flag = TemporalFlag(
            name="MAR_YAWN",
            duration_required=config.MAR_YAWN_DURATION,
            cooldown=config.EAR_COOLDOWN,
        )

        # Rolling baseline of recent EAR readings for the adaptive threshold.
        self._ear_history = deque(maxlen=config.EAR_BASELINE_WINDOW_SAMPLES)

        # PERCLOS: rolling (timestamp, eyes_closed) samples over the last
        # PERCLOS_WINDOW_SECONDS, used to compute % time closed.
        self._perclos_samples = deque()
        self._was_perclos_active = False
        self._perclos_last_end_time = None

    def _current_threshold(self) -> float:
        if len(self._ear_history) < config.EAR_BASELINE_MIN_SAMPLES:
            # Not enough history yet -- use the fixed floor.
            return config.EAR_CLOSED_THRESHOLD_FLOOR
        baseline = float(np.percentile(self._ear_history, config.EAR_BASELINE_PERCENTILE))
        adaptive = baseline * config.EAR_CLOSED_RATIO
        # Never trust an adaptive threshold below the sanity floor (e.g. if
        # the baseline itself got dragged down by a long eyes-closed spell).
        return max(adaptive, config.EAR_CLOSED_THRESHOLD_FLOOR)

    def _update_perclos(self, eyes_closed: bool, video_time: float):
        """
        Returns (perclos_ratio: float, perclos_active: bool, perclos_event: str|None).
        """
        self._perclos_samples.append((video_time, eyes_closed))
        cutoff = video_time - config.PERCLOS_WINDOW_SECONDS
        while self._perclos_samples and self._perclos_samples[0][0] < cutoff:
            self._perclos_samples.popleft()

        span = self._perclos_samples[-1][0] - self._perclos_samples[0][0] if len(self._perclos_samples) > 1 else 0.0
        enough_coverage = span >= (config.PERCLOS_WINDOW_SECONDS * config.PERCLOS_MIN_WINDOW_COVERAGE)

        closed_count = sum(1 for _, c in self._perclos_samples if c)
        ratio = closed_count / len(self._perclos_samples) if self._perclos_samples else 0.0

        active = enough_coverage and ratio >= config.PERCLOS_THRESHOLD

        event = None
        cooldown_ok = (
            self._perclos_last_end_time is None
            or (video_time - self._perclos_last_end_time) >= config.PERCLOS_COOLDOWN
        )
        if active and not self._was_perclos_active and cooldown_ok:
            event = "START"
        elif not active and self._was_perclos_active:
            event = "END"
            self._perclos_last_end_time = video_time
        self._was_perclos_active = active

        return round(ratio, 3), active, event

    def process_frame(self, frame_bgr, video_time: float) -> dict:
        if not self.available:
            return {
                "ear": None, "mar": None, "ear_threshold": None,
                "eyes_closed": False, "yawning": False, "sustained_active": False,
                "eye_event": None, "yawn_event": None,
                "perclos": None, "perclos_active": False, "perclos_event": None,
                "error": self.unavailable_reason,
            }

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            eye_event = self.eye_flag.update(False, video_time)
            yawn_event = self.yawn_flag.update(False, video_time)
            # No face -> don't feed a guessed "open" sample into PERCLOS;
            # a dropout shouldn't silently push the rolling ratio down.
            return {
                "ear": None, "mar": None, "ear_threshold": None,
                "eyes_closed": False, "yawning": False,
                "sustained_active": self.eye_flag.is_active or self.yawn_flag.is_active or self._was_perclos_active,
                "eye_event": eye_event, "yawn_event": yawn_event,
                "perclos": None, "perclos_active": self._was_perclos_active, "perclos_event": None,
                "error": "NO_FACE",
            }

        lm = results.multi_face_landmarks[0].landmark

        def pt(i):
            return (lm[i].x * w, lm[i].y * h)

        right_ear = _ear([pt(i) for i in _RIGHT_EYE])
        left_ear = _ear([pt(i) for i in _LEFT_EYE])
        avg_ear = (right_ear + left_ear) / 2.0

        mouth_w = _dist(pt(_MOUTH_CORNERS[0]), pt(_MOUTH_CORNERS[1]))
        mouth_h = _dist(pt(_MOUTH_VERTICAL[0]), pt(_MOUTH_VERTICAL[1]))
        mar = mouth_h / (mouth_w + 1e-6)

        threshold = self._current_threshold()
        eyes_closed = avg_ear < threshold
        yawning = mar > config.MAR_YAWN_THRESHOLD

        # Update the rolling EAR baseline AFTER computing this frame's
        # decision, and only with "probably open" readings so a long
        # closed-eye spell doesn't drag the baseline down and mask itself.
        if not eyes_closed:
            self._ear_history.append(avg_ear)

        eye_conf = float(np.clip(1.0 - (avg_ear / threshold), 0.0, 1.0)) if threshold > 0 else 0.0
        eye_event = self.eye_flag.update(eyes_closed, video_time, confidence=eye_conf)
        yawn_event = self.yawn_flag.update(yawning, video_time, confidence=min(mar, 1.0))

        perclos_ratio, perclos_active, perclos_event = self._update_perclos(eyes_closed, video_time)

        return {
            "ear": round(avg_ear, 3),
            "mar": round(mar, 3),
            "ear_threshold": round(threshold, 3),
            "eyes_closed": eyes_closed,
            "yawning": yawning,
            "sustained_active": self.eye_flag.is_active or self.yawn_flag.is_active or perclos_active,
            "eye_event": eye_event,
            "yawn_event": yawn_event,
            "perclos": perclos_ratio,
            "perclos_active": perclos_active,
            "perclos_event": perclos_event,
        }