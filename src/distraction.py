"""
src/distraction.py
===================
Wraps the local distraction model. Uses whatever classes the model
actually reports (e.g. "safe" / "phone" / "drink" / "distracted"). Any
label that is not recognized as "safe" is treated as a distraction
condition; edit SAFE_LABEL_HINTS below to match your model's real
"no distraction" class name(s).
"""

from src.detector_base import GenericFrameModel
from src.event_manager import TemporalFlag
import config

SAFE_LABEL_HINTS = ["safe", "normal", "attentive", "focused", "none"]


class DistractionDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = GenericFrameModel(model_path, device=device)
        self.flag = TemporalFlag(
            name="DISTRACTION",
            duration_required=config.DISTRACTION_DURATION,
            cooldown=config.DISTRACTION_COOLDOWN,
        )
        self.last_label = "UNKNOWN"
        self.last_conf = 0.0

    def _is_safe_label(self, label: str) -> bool:
        label_lower = label.lower()
        return any(hint in label_lower for hint in SAFE_LABEL_HINTS)

    def process_frame(self, frame_bgr, video_time: float):
        try:
            label, conf, _extra = self.model.predict(frame_bgr)
        except RuntimeError as e:
            return {"label": "ERROR", "confidence": 0.0, "sustained_active": False,
                     "event": None, "error": str(e)}

        self.last_label = label
        self.last_conf = conf

        condition = (not self._is_safe_label(label)) and conf >= config.DISTRACTION_CONFIDENCE
        event = self.flag.update(condition, video_time, confidence=conf)

        # Distraction uses explicit START/END naming per spec section 11
        event_name = None
        if event == "START":
            event_name = "DISTRACTION_START"
        elif event == "END":
            event_name = "DISTRACTION_END"

        return {
            "label": label,
            "confidence": conf,
            "sustained_active": self.flag.is_active,
            "event": event,
            "event_name": event_name,
        }
