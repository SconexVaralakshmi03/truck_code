"""
src/road_monitor.py
====================
Wraps the local road/lane model. Three cases are handled explicitly:

1. Ultralytics detection/segmentation checkpoint -> generic Ultralytics path.
2. ONNX export -> generic ONNX Runtime path (export-specific postprocessing
   still needs to be added per your exact export).
3. The official YOLOP checkpoint format (dict with a 'state_dict' key,
   542 params, 'model.N....' key naming) -> routed to src/yolop_model.py,
   which uses the REAL vendored YOLOP architecture (third_party/yolop/,
   MIT licensed) and the official repo's own pre/post-processing, so
   nothing here is guessed or fabricated.

This module NEVER fabricates lane or drivable-area output. Its capability
flags (`has_lane`, `has_drivable_area`, `has_detection`) are set based on
what actually loaded successfully.
"""

import numpy as np
import cv2

from src.model_loader import inspect_and_load
from src.event_manager import TemporalFlag
import config


class RoadMonitor:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.result = inspect_and_load(model_path, prefer_device=device)

        self.has_detection = False
        self.has_lane = False
        self.has_drivable_area = False
        self.usable = False
        self.load_error = self.result.load_error
        self._yolop = None

        # --- Case 3: official YOLOP checkpoint ---
        if self._is_yolop_checkpoint():
            try:
                from src.yolop_model import YOLOPModel
                self._yolop = YOLOPModel(model_path, device=device)
                self.usable = True
                self.has_detection = True
                self.has_lane = True
                self.has_drivable_area = True
                self.load_error = None
            except Exception as e:
                self.usable = False
                self.load_error = f"YOLOPModel failed to load real weights: {e}"

        # --- Case 1: Ultralytics ---
        elif self.result.load_status == "PASS" and self.result.framework == "Ultralytics":
            self.usable = True
            self.has_detection = self.result.task in ("detect", "segment")
            self.has_lane = False
            self.has_drivable_area = self.result.task == "segment"

        # --- Case 2: ONNX ---
        elif self.result.load_status == "PASS" and self.result.framework == "ONNX Runtime":
            self.usable = True
            self.has_detection = True
            self.has_drivable_area = True
            self.has_lane = True

        else:
            self.usable = False

        self.lane_flag = TemporalFlag(
            name="LANE_DEPARTURE",
            duration_required=config.LANE_DEPARTURE_DURATION,
            cooldown=config.LANE_DEPARTURE_COOLDOWN,
        )

    def _is_yolop_checkpoint(self) -> bool:
        r = self.result
        if r.framework != "UNKNOWN (.bin)" and "PyTorch" not in r.framework:
            return False
        if r.architecture and "YOLOP" in r.architecture:
            return True
        return False

    def process_frame(self, frame_bgr, video_time: float):
        """
        Returns a dict with whatever capabilities are actually supported.
        Unsupported capabilities are explicitly marked, never guessed.
        """
        out = {
            "detection": "NOT SUPPORTED",
            "lane": "NOT SUPPORTED",
            "drivable_area": "NOT SUPPORTED",
            "lane_offset": None,
            "lane_event": None,
            "road_departure": "NOT RELIABLY SUPPORTED",
            "objects": [],
            "drivable_area_mask": None,
            "lane_mask": None,
            "error": None,
        }

        if not self.usable:
            out["error"] = self.load_error or "Road model not usable — see inspect_models.py output."
            event = self.lane_flag.update(False, video_time)
            out["lane_event"] = event
            return out

        # --- YOLOP real inference ---
        if self._yolop is not None:
            result = self._yolop.process_frame(frame_bgr)
            out["objects"] = result["objects"]
            out["detection"] = "DETECTED" if result["objects"] else "NONE"
            out["drivable_area_mask"] = result["drivable_area_mask"]
            out["lane_mask"] = result["lane_mask"]
            out["drivable_area"] = "DETECTED" if result["drivable_area_mask"].any() else "NONE"
            out["lane"] = "DETECTED" if result["lane_mask"].any() else "NONE"

            offset = self.compute_lane_offset(result["lane_mask"], frame_bgr.shape[1])
            out["lane_offset"] = round(offset, 3)
            event = self.update_lane_departure(offset, video_time)
            out["lane_event"] = event
            return out

        if self.result.framework == "Ultralytics":
            model = self.result.handle
            results = model.predict(source=frame_bgr, device=self.device, verbose=False)
            r = results[0]
            objects = []
            if r.boxes is not None:
                names = self.result.classes or {}
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = names[cls_id] if names and cls_id < len(names) else str(cls_id)
                    xyxy = box.xyxy[0].tolist()
                    objects.append({"label": label, "confidence": conf, "box": xyxy})
            out["objects"] = objects
            out["detection"] = "DETECTED" if objects else "NONE"

            if self.has_drivable_area and r.masks is not None:
                out["drivable_area"] = "DETECTED"
            else:
                out["drivable_area"] = "NOT SUPPORTED" if not self.has_drivable_area else "NONE"

            out["lane"] = "NOT SUPPORTED"  # standard Ultralytics seg model has no lane-line head
            event = self.lane_flag.update(False, video_time)
            out["lane_event"] = event
            return out

        if self.result.framework == "ONNX Runtime":
            # A real YOLOP ONNX export needs its own specific pre/post
            # processing (letterbox resize, 3-way output split). We surface
            # the raw session so a project-specific post-processor can be
            # added; for the generic POC we report structural detection
            # of the three heads without fabricating pixel-level results
            # unless the exact export format has been confirmed.
            out["detection"] = "MODEL LOADED — implement export-specific postprocessing in road_monitor.py"
            out["lane"] = "MODEL LOADED — implement export-specific postprocessing in road_monitor.py"
            out["drivable_area"] = "MODEL LOADED — implement export-specific postprocessing in road_monitor.py"
            event = self.lane_flag.update(False, video_time)
            out["lane_event"] = event
            return out

        return out

    def compute_lane_offset(self, lane_mask: np.ndarray, frame_width: int) -> float:
        """
        Given a binary lane mask (left/right boundaries visible), estimate
        normalized offset of frame-center from lane-center. Returns a value
        in roughly [-1, 1]; 0 = centered. Caller feeds this into
        update_lane_departure(). Only meaningful if has_lane is True.
        """
        col_sums = lane_mask.sum(axis=0)
        nonzero = np.nonzero(col_sums)[0]
        if len(nonzero) < 2:
            return 0.0
        left_bound = nonzero[0]
        right_bound = nonzero[-1]
        lane_center = (left_bound + right_bound) / 2.0
        frame_center = frame_width / 2.0
        lane_width = max(right_bound - left_bound, 1)
        offset = (frame_center - lane_center) / (lane_width / 2.0)
        return float(np.clip(offset, -1.0, 1.0))

    def update_lane_departure(self, offset: float, video_time: float):
        condition = abs(offset) >= config.LANE_OFFSET_THRESHOLD
        return self.lane_flag.update(condition, video_time, confidence=abs(offset))