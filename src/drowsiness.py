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
        )
        self.last_label = "UNKNOWN"
        self.last_conf = 0.0

    def _is_drowsy_label(self, label: str) -> bool:
        label_lower = label.lower()
        return any(hint in label_lower for hint in DROWSY_LABEL_HINTS)

    def process_frame(self, frame_bgr, video_time: float):
        """
        Returns dict: {label, confidence, sustained_active, event (START/END/None)}
        """
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
