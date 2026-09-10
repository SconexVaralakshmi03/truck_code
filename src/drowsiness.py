"""
src/drowsiness.py
==================
Wraps the local drowsiness model (whatever its actual classes turn out to
be — do NOT hardcode "Drowsy"/"Non-Drowsy" if the model reports something
else; this module reads config.DROWSY_LABELS but falls back to fuzzy
matching against the model's own reported class names).
"""

from src.detector_base import GenericFrameModel
from src.event_manager import TemporalFlag
import config


# Labels we treat as "drowsy" if present in the model's class list. If the
# model's real classes differ, edit this list to match — we do not invent
# classes that aren't actually in the model.
DROWSY_LABEL_HINTS = ["drowsy", "sleep", "yawn", "closed", "fatigue"]


class DrowsinessDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = GenericFrameModel(model_path, device=device)
        self.flag = TemporalFlag(
            name="DROWSINESS",
            duration_required=config.DROWSINESS_DURATION,
            cooldown=config.DROWSINESS_COOLDOWN,
            # Tolerates a short streak of non-drowsy-looking frames (one
            # noisy classifier call, a dropped WebSocket frame) without
            # resetting the sustained timer. Without this, a classifier
            # that flips its per-frame verdict even occasionally during a
            # real ~2s drowsy episode could keep restarting the timer and
            # never reach DROWSINESS_DURATION -- or, once it did, flap
            # START/END repeatedly for what should be one continuous event.
            grace_period=config.DROWSINESS_GRACE_PERIOD,
        )
        self.last_label = "UNKNOWN"
        self.last_conf = 0.0

    def _is_drowsy_label(self, label: str) -> bool:
        label_lower = label.lower()
        return any(hint in label_lower for hint in DROWSY_LABEL_HINTS)

    def process_frame(self, frame_bgr, video_time: float, face_present: bool = True):
        """
        Returns dict: {label, confidence, sustained_active, event (START/END/None)}

        face_present: whether a face/driver was found in this frame by the
        geometric (MediaPipe) stage, which runs first in the pipeline and
        is cheap relative to this classifier. When config.DROWSINESS_
        REQUIRE_FACE is on and face_present is False, this frame is NOT
        run through the (heavier) classifier at all -- both because a
        classifier trained on driver-facing crops has no reliable meaning
        on an empty/mispointed frame (ceiling, dashboard, door -- exactly
        the kind of frame a live dash camera produces between the driver
        sitting down and the camera settling), and because skipping the
        inference call on those frames reduces latency for the common
        real-time "nobody/nothing to classify yet" case.
        """
        if config.DROWSINESS_REQUIRE_FACE and not face_present:
            self.last_label = "NO_FACE"
            self.last_conf = 0.0
            event = self.flag.update(False, video_time, confidence=0.0)
            return {
                "label": "NO_FACE",
                "confidence": 0.0,
                "sustained_active": self.flag.is_active,
                "event": event,
            }

        try:
            label, conf, _extra = self.model.predict(frame_bgr)
        except RuntimeError as e:
            return {"label": "ERROR", "confidence": 0.0, "sustained_active": False,
                     "event": None, "error": str(e)}

        self.last_label = label
        self.last_conf = conf

        condition = self._is_drowsy_label(label) and conf >= config.DROWSINESS_CONFIDENCE
        event = self.flag.update(condition, video_time, confidence=conf)

        return {
            "label": label,
            "confidence": conf,
            "sustained_active": self.flag.is_active,
            "event": event,
        }
