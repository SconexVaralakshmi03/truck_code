"""
src/pose_utils.py
==================
MediaPipe Pose-based geometric fallback check for phone-usage detection:
is a wrist bent up near the ear (classic phone call) OR near the mouth
(speakerphone / video-call-style holding the phone up in front of the
face)? Either posture is treated the same way -- as indirect evidence
used ONLY when the phone itself isn't directly visible to the classifier
or YOLOv8 (see src/phone_detection.py for the full cascade and why this
check runs last, not first).

Assumes a dash-mounted camera facing the driver roughly head-on (as
described: straight in front of the driver seat), so a single symmetric
left/right threshold is used without extra per-install calibration.

NOTE: On first use, MediaPipe downloads its pose_landmark model file from
Google's servers (same one-time-download pattern Ultralytics already uses
for its own weights) -- this requires internet access on first run only.
"""

import numpy as np
import cv2

import config

try:
    import mediapipe as mp
    # Same legacy-API guard as src/head_pose.py -- see requirements.txt pin.
    _ = mp.solutions.pose
    _MEDIAPIPE_AVAILABLE = True
except Exception:
    _MEDIAPIPE_AVAILABLE = False

# MediaPipe Pose (BlazePose, 33 landmarks) indices
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_ELBOW, _RIGHT_ELBOW = 13, 14
_LEFT_WRIST, _RIGHT_WRIST = 15, 16
_LEFT_EAR, _RIGHT_EAR = 7, 8
_MOUTH_LEFT, _MOUTH_RIGHT = 9, 10


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _angle_deg(a, b, c):
    """Angle at point b, formed by points a-b-c, in degrees.
    ~180 deg = straight arm, small angle = sharply bent elbow."""
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


class PoseGate:
    def __init__(self):
        self.available = _MEDIAPIPE_AVAILABLE
        self.unavailable_reason = None
        if self.available:
            try:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,  # lightest model -- this is only a fallback check
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                self.available = False
                self.unavailable_reason = str(e)
        else:
            self.unavailable_reason = (
                "mediapipe legacy 'solutions.pose' API unavailable. "
                "Pin mediapipe<0.10.14 (see requirements.txt)."
            )

    def check_frame(self, frame_bgr) -> dict:
        """
        Returns:
            {
              "hand_near_ear": bool,      # True for EITHER the ear or mouth match
                                           # (name kept for backward compatibility
                                           # with callers/logs -- see "target" for which)
              "side": "left" | "right" | None,
              "target": "ear" | "mouth" | None,
              "landmarks_found": bool,
              "wrist_target_distance_ratio": float | None,
              "elbow_angle_deg": float | None,
              "error": str | None,
            }
        """
        out = {
            "hand_near_ear": False, "side": None, "target": None, "landmarks_found": False,
            "wrist_target_distance_ratio": None, "elbow_angle_deg": None, "error": None,
        }
        if not self.available:
            out["error"] = self.unavailable_reason
            return out

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return out

        out["landmarks_found"] = True
        lm = results.pose_landmarks.landmark

        def pt(i):
            return (lm[i].x * w, lm[i].y * h)

        def vis(i):
            return lm[i].visibility

        shoulder_width = _dist(pt(_LEFT_SHOULDER), pt(_RIGHT_SHOULDER))
        if shoulder_width < 1e-3:
            return out

        mouth_center = None
        if vis(_MOUTH_LEFT) >= 0.3 and vis(_MOUTH_RIGHT) >= 0.3:
            ml, mr = pt(_MOUTH_LEFT), pt(_MOUTH_RIGHT)
            mouth_center = ((ml[0] + mr[0]) / 2.0, (ml[1] + mr[1]) / 2.0)

        best = None  # (side, target, distance_ratio, elbow_angle) -- keep the closest match

        for side, wrist_i, elbow_i, shoulder_i, ear_i in [
            ("left", _LEFT_WRIST, _LEFT_ELBOW, _LEFT_SHOULDER, _LEFT_EAR),
            ("right", _RIGHT_WRIST, _RIGHT_ELBOW, _RIGHT_SHOULDER, _RIGHT_EAR),
        ]:
            if vis(wrist_i) < 0.4 or vis(elbow_i) < 0.4:
                continue  # this arm isn't reliably visible this frame -- skip, don't guess

            elbow_angle = _angle_deg(pt(shoulder_i), pt(elbow_i), pt(wrist_i))
            if elbow_angle >= config.PHONE_ELBOW_BEND_MAX_DEG:
                continue  # arm not bent -- neither ear-call nor speakerphone posture

            wrist = pt(wrist_i)

            # Target 1: ear (classic phone-to-ear call posture)
            if vis(ear_i) >= 0.3:
                ear_dist = _dist(wrist, pt(ear_i)) / shoulder_width
                if ear_dist < config.PHONE_HAND_EAR_DISTANCE_RATIO:
                    if best is None or ear_dist < best[2]:
                        best = (side, "ear", ear_dist, elbow_angle)

            # Target 2: mouth center (speakerphone / video-call-style, phone
            # held up in front of the face rather than against the ear)
            if mouth_center is not None:
                mouth_dist = _dist(wrist, mouth_center) / shoulder_width
                if mouth_dist < config.PHONE_HAND_MOUTH_DISTANCE_RATIO:
                    if best is None or mouth_dist < best[2]:
                        best = (side, "mouth", mouth_dist, elbow_angle)

        if best:
            side, target, dist_ratio, elbow_angle = best
            out.update({
                "hand_near_ear": True,  # kept as the general "posture matched" flag
                "side": side,
                "target": target,
                "wrist_target_distance_ratio": round(dist_ratio, 3),
                "elbow_angle_deg": round(elbow_angle, 1),
            })

        return out