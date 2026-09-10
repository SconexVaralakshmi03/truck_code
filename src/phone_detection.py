# """
# src/phone_detection.py
# =======================
# Wraps the local phone-usage model. Uses whatever classes the model
# actually reports (e.g. "phone" / "no_phone", or detection classes like
# "cell phone").
# """

# from src.detector_base import GenericFrameModel
# from src.event_manager import TemporalFlag
# import config

# PHONE_LABEL_HINTS = ["phone", "mobile", "cell"]
# NEGATIVE_HINTS = ["no_phone", "no-phone", "none", "background", "negative"]


# class PhoneDetector:
#     def __init__(self, model_path: str, device: str = "cpu"):
#         self.model = GenericFrameModel(model_path, device=device)
#         self.flag = TemporalFlag(
#             name="PHONE_USAGE",
#             duration_required=config.PHONE_DURATION,
#             cooldown=config.PHONE_COOLDOWN,
#         )
#         self.last_label = "UNKNOWN"
#         self.last_conf = 0.0

#     def _is_phone_label(self, label: str) -> bool:
#         label_lower = label.lower()
#         if any(neg in label_lower for neg in NEGATIVE_HINTS):
#             return False
#         return any(hint in label_lower for hint in PHONE_LABEL_HINTS)

#     def process_frame(self, frame_bgr, video_time: float):
#         try:
#             label, conf, _extra = self.model.predict(frame_bgr)
#         except RuntimeError as e:
#             return {"label": "ERROR", "confidence": 0.0, "sustained_active": False,
#                      "event": None, "error": str(e)}

#         self.last_label = label
#         self.last_conf = conf

#         condition = self._is_phone_label(label) and conf >= config.PHONE_CONFIDENCE
#         event = self.flag.update(condition, video_time, confidence=conf)

#         return {
#             "label": label,
#             "confidence": conf,
#             "sustained_active": self.flag.is_active,
#             "event": event,
#         }

"""
src/phone_detection.py
=======================
Direct-first phone-usage detection, with geometry as a fallback (not a
gate):

  1. CLASSIFIER DIRECT: run the local phone classifier on the frame. If it
     actually sees a phone with sufficient confidence, that's the strongest
     evidence available -- trust it immediately.
  2. YOLOV8 DIRECT: if the classifier didn't confirm (missing, broken, or
     says "no phone"), run a generic pretrained YOLOv8 COCO detector
     looking for a 'cell phone' box anywhere in frame. Also direct visual
     evidence, independent of the classifier.
  3. POSTURE FALLBACK, ONLY IF NEITHER SAW A PHONE: check MediaPipe Pose
     for a bent elbow with the wrist near the ear OR near the mouth
     (src/pose_utils.PoseGate — covers both the classic phone-to-ear call
     and the speakerphone/video-call-style held-up-to-face posture). This
     step only runs when steps 1-2 found nothing, because it's weaker,
     indirect evidence -- a bent arm near the face is consistent with a
     phone call, but also with scratching an ear, adjusting glasses, etc.
     It exists specifically to catch the case where a phone genuinely IS in
     use but isn't visually confirmable that frame (occluded by the hand/
     head, awkward angle, motion blur) -- not as the primary signal.

This is a deliberate inversion from gating classifier/YOLOv8 calls behind
the posture check: direct visual confirmation should never be skipped or
delayed by a geometric pre-filter, since geometry is the weaker signal of
the two. The trade-off is that the classifier and YOLOv8 fallback now run
on every frame rather than only when the gate is open -- more compute, but
correct evidence-ranking. The posture check still only runs on frames
where it's actually needed (i.e. it's the one thing still "gated").

Degrades gracefully: if MediaPipe Pose is unavailable, step 3 is simply
skipped (steps 1-2 alone still work). If the local classifier is missing/
broken, YOLOv8 alone can still confirm, with posture as the last resort.
"""

import os

from src.detector_base import GenericFrameModel
from src.event_manager import TemporalFlag
from src.pose_utils import PoseGate
import config

PHONE_LABEL_HINTS = ["phone", "mobile", "cell"]
NEGATIVE_HINTS = ["no_phone", "no-phone", "none", "background", "negative"]

_YOLO_COCO_PHONE_CLASS_NAME = "cell phone"


class PhoneDetector:
    def __init__(self, model_path, device: str = "cpu"):
        self.device = device

        self.classifier = None
        self.classifier_error = None
        if model_path:
            try:
                self.classifier = GenericFrameModel(model_path, device=device)
            except RuntimeError as e:
                self.classifier_error = str(e)
        else:
            self.classifier_error = "No local phone classifier model provided."

        self.pose_gate = PoseGate()

        self._yolo_fallback = None
        self._yolo_fallback_error = None
        self._yolo_fallback_attempted = False

        self.flag = TemporalFlag(
            name="PHONE_USAGE",
            duration_required=config.PHONE_DURATION,
            cooldown=config.PHONE_COOLDOWN,
            # Same real-time jitter tolerance as drowsiness -- one missed
            # detection in the middle of a real, ongoing phone-usage
            # episode (motion blur, brief occlusion, a dropped frame)
            # shouldn't restart the 2s timer from zero.
            grace_period=config.PHONE_GRACE_PERIOD,
        )
        self.last_label = "UNKNOWN"
        self.last_conf = 0.0
        self.last_mode = "NONE"

    def _is_phone_label(self, label: str) -> bool:
        label_lower = label.lower()
        if any(neg in label_lower for neg in NEGATIVE_HINTS):
            return False
        return any(hint in label_lower for hint in PHONE_LABEL_HINTS)

    def _load_yolo_fallback(self):
        if self._yolo_fallback_attempted:
            return
        self._yolo_fallback_attempted = True
        try:
            from ultralytics import YOLO
            path = config.PHONE_YOLO_FALLBACK_MODEL
            # Use a local copy if you've placed one (fully offline); otherwise
            # pass the bare model name so Ultralytics auto-downloads the
            # standard pretrained COCO weights on first use (needs internet
            # once, then it's cached locally -- same pattern you already rely
            # on for the rest of the Ultralytics ecosystem).
            source = path if os.path.exists(path) else "yolov8m.pt"
            self._yolo_fallback = YOLO(source)
        except Exception as e:
            self._yolo_fallback_error = str(e)

    def _check_yolo_fallback(self, frame_bgr):
        self._load_yolo_fallback()
        if self._yolo_fallback is None:
            return "no_phone (fallback unavailable)", 0.0
        try:
            results = self._yolo_fallback.predict(
                source=frame_bgr, device=self.device, verbose=False, conf=config.PHONE_YOLO_CONF
            )
            r = results[0]
            names = r.names
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    if label == _YOLO_COCO_PHONE_CLASS_NAME:
                        return "phone (yolov8-coco)", float(box.conf[0])
            return "no_phone (yolov8-coco)", 0.0
        except Exception as e:
            return f"yolov8_fallback_error: {e}", 0.0

    def process_frame(self, frame_bgr, video_time: float):
        confirmed = False
        label = "UNKNOWN"
        conf = 0.0
        mode = "NONE_DETECTED"
        gate = {
            "hand_near_ear": False, "side": None, "target": None, "landmarks_found": False,
            "wrist_target_distance_ratio": None, "elbow_angle_deg": None, "error": None,
        }

        # --- Step 1: classifier, direct ---
        classifier_label, classifier_conf = None, 0.0
        if self.classifier is not None:
            try:
                classifier_label, classifier_conf, _extra = self.classifier.predict(frame_bgr)
                if self._is_phone_label(classifier_label) and classifier_conf >= config.PHONE_CONFIDENCE:
                    confirmed = True
                    label, conf, mode = classifier_label, classifier_conf, "CLASSIFIER_DIRECT"
            except RuntimeError:
                classifier_label = None  # unusable this run -- fall through

        # --- Step 2: YOLOv8 COCO, direct (only if step 1 didn't confirm) ---
        fallback_label, fallback_conf = None, 0.0
        if not confirmed:
            fallback_label, fallback_conf = self._check_yolo_fallback(frame_bgr)
            if fallback_label.startswith("phone") and fallback_conf >= config.PHONE_YOLO_CONF:
                confirmed = True
                label, conf, mode = fallback_label, fallback_conf, "YOLO_DIRECT"

        # --- Step 3: posture fallback, ONLY if neither model saw a phone ---
        if not confirmed:
            gate = self.pose_gate.check_frame(frame_bgr)
            if gate.get("hand_near_ear", False):
                confirmed = True
                ratio = gate.get("wrist_target_distance_ratio")
                target = gate.get("target", "ear")
                # Whichever posture target matched (ear or mouth) has its own
                # distance-ratio threshold; use the matching one to normalize
                # the inferred confidence to roughly [0, 1].
                threshold = (
                    config.PHONE_HAND_MOUTH_DISTANCE_RATIO if target == "mouth"
                    else config.PHONE_HAND_EAR_DISTANCE_RATIO
                )
                pose_conf = (
                    round(1.0 - min(ratio / threshold, 1.0), 2)
                    if ratio is not None else 0.5
                )
                target_desc = "near ear" if target == "ear" else "held up near mouth (speakerphone-style)"
                label = f"phone (posture-inferred, {gate.get('side')} hand {target_desc} — not visually confirmed)"
                conf = pose_conf
                mode = "POSTURE_INFERRED"
            else:
                # Nothing found by any method this frame -- keep whatever
                # direct-check label we have for display/debugging,
                # preferring the classifier's own verdict over YOLOv8's.
                if classifier_label is not None:
                    label, conf = classifier_label, classifier_conf
                elif fallback_label is not None:
                    label, conf = fallback_label, fallback_conf
                mode = "NONE_DETECTED"

        self.last_label, self.last_conf, self.last_mode = label, conf, mode

        event = self.flag.update(confirmed, video_time, confidence=conf)

        return {
            "label": label,
            "confidence": conf,
            "sustained_active": self.flag.is_active,
            "event": event,
            "gate": gate,
            "mode": mode,
        }