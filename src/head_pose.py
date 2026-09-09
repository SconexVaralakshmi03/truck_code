# """
# src/head_pose.py
# =================
# Lightweight head-orientation estimation used ONLY as a fallback when the
# provided drowsiness/phone/distraction models don't reliably expose head
# orientation. Uses MediaPipe Face Mesh (no training, no new model) to solve
# a PnP problem and classify the head as FORWARD / LEFT / RIGHT / UP / DOWN,
# then applies temporal smoothing before declaring DRIVER_LOOKING_AWAY.

# If MediaPipe is not installed or no face is detected, this module reports
# that clearly rather than fabricating a pose.
# """

# import numpy as np
# import cv2

# from src.event_manager import TemporalFlag
# import config

# try:
#     import mediapipe as mp
#     # Some recent MediaPipe releases (0.10.14+) dropped the legacy
#     # `mp.solutions.face_mesh` Python API in favor of the newer Tasks API.
#     # We require the legacy solutions API here; if it isn't present we
#     # treat MediaPipe as unavailable rather than crashing at import time.
#     _ = mp.solutions.face_mesh  # noqa: B018 - existence check only
#     _MEDIAPIPE_AVAILABLE = True
# except Exception:
#     _MEDIAPIPE_AVAILABLE = False


# # 3D model points for a generic face (nose tip, chin, eye corners, mouth
# # corners) used for solvePnP. Approximate, in arbitrary units.
# _MODEL_POINTS = np.array([
#     (0.0, 0.0, 0.0),          # Nose tip
#     (0.0, -330.0, -65.0),     # Chin
#     (-225.0, 170.0, -135.0),  # Left eye left corner
#     (225.0, 170.0, -135.0),   # Right eye right corner
#     (-150.0, -150.0, -125.0), # Left mouth corner
#     (150.0, -150.0, -125.0),  # Right mouth corner
# ], dtype=np.float64)

# # Corresponding MediaPipe Face Mesh landmark indices
# _LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


# class HeadPoseEstimator:
#     def __init__(self):
#         self.available = _MEDIAPIPE_AVAILABLE
#         self.unavailable_reason = None
#         if self.available:
#             try:
#                 self._mesh = mp.solutions.face_mesh.FaceMesh(
#                     static_image_mode=False,
#                     max_num_faces=1,
#                     refine_landmarks=False,
#                     min_detection_confidence=0.5,
#                     min_tracking_confidence=0.5,
#                 )
#             except Exception as e:
#                 self.available = False
#                 self.unavailable_reason = str(e)
#         else:
#             self.unavailable_reason = (
#                 "mediapipe is installed but its legacy 'solutions.face_mesh' API is not "
#                 "available in this version. Either pin mediapipe<0.10.14 "
#                 "(pip install 'mediapipe<0.10.14'), or port this module to the newer "
#                 "Tasks API (mediapipe.tasks.python.vision.FaceLandmarker) — see "
#                 "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
#             )
#         self.flag = TemporalFlag(
#             name="DRIVER_LOOKING_AWAY",
#             duration_required=config.HEAD_POSE_AWAY_DURATION,
#             cooldown=5.0,
#         )

#     def _classify(self, yaw, pitch):
#         if abs(yaw) < config.HEAD_POSE_YAW_THRESHOLD_DEG and abs(pitch) < config.HEAD_POSE_PITCH_THRESHOLD_DEG:
#             return "FORWARD"
#         if yaw >= config.HEAD_POSE_YAW_THRESHOLD_DEG:
#             return "RIGHT"
#         if yaw <= -config.HEAD_POSE_YAW_THRESHOLD_DEG:
#             return "LEFT"
#         if pitch >= config.HEAD_POSE_PITCH_THRESHOLD_DEG:
#             return "DOWN"
#         if pitch <= -config.HEAD_POSE_PITCH_THRESHOLD_DEG:
#             return "UP"
#         return "FORWARD"

#     def process_frame(self, frame_bgr, video_time: float):
#         if not self.available:
#             return {"pose": "UNAVAILABLE", "event": None, "sustained_active": False,
#                      "error": self.unavailable_reason or "mediapipe not installed"}

#         h, w = frame_bgr.shape[:2]
#         rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#         results = self._mesh.process(rgb)

#         if not results.multi_face_landmarks:
#             # No face -> cannot classify pose this frame; do not fabricate.
#             event = self.flag.update(False, video_time)
#             return {"pose": "NO_FACE", "event": event, "sustained_active": self.flag.is_active}

#         landmarks = results.multi_face_landmarks[0].landmark
#         image_points = np.array([
#             (landmarks[i].x * w, landmarks[i].y * h) for i in _LANDMARK_IDS
#         ], dtype=np.float64)

#         focal_length = w
#         center = (w / 2, h / 2)
#         camera_matrix = np.array([
#             [focal_length, 0, center[0]],
#             [0, focal_length, center[1]],
#             [0, 0, 1],
#         ], dtype=np.float64)
#         dist_coeffs = np.zeros((4, 1))

#         success, rotation_vec, _translation_vec = cv2.solvePnP(
#             _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
#             flags=cv2.SOLVEPNP_ITERATIVE,
#         )

#         if not success:
#             event = self.flag.update(False, video_time)
#             return {"pose": "UNKNOWN", "event": event, "sustained_active": self.flag.is_active}

#         rotation_mat, _ = cv2.Rodrigues(rotation_vec)
#         pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
#         _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
#         euler_angles = np.asarray(euler_angles).flatten()  # (3,1) -> (3,) so each entry is a scalar
#         pitch, yaw, _roll = [float(a) for a in euler_angles]

#         pose = self._classify(yaw, pitch)
#         away = pose != "FORWARD"
#         event = self.flag.update(away, video_time, confidence=1.0)

#         return {
#             "pose": pose,
#             "yaw": yaw,
#             "pitch": pitch,
#             "event": event,
#             "sustained_active": self.flag.is_active,
#         }

"""
src/head_pose.py
=================
Lightweight head-orientation estimation used ONLY as a fallback when the
provided drowsiness/phone/distraction models don't reliably expose head
orientation. Uses MediaPipe Face Mesh (no training, no new model) to solve
a PnP problem and classify the head as FORWARD / LEFT / RIGHT / UP / DOWN,
then applies temporal smoothing before declaring DRIVER_LOOKING_AWAY.

Two ways "not focused on driving" is detected, per spec — LEFT/RIGHT ONLY
(UP/DOWN are still reported in the "pose" field for visibility, but do NOT
count toward "not focused", per your requirement):
  1. CONTINUOUS: looking left or right for >= HEAD_POSE_AWAY_DURATION
     seconds straight (default 10s).
  2. FREQUENT: several separate left/right glances within a rolling window
     (e.g. 3+ glances in 30s), even if none individually reaches 10s --
     repeated short glances are also a "not focused" pattern, just a
     different one.

Assumes the camera is mounted roughly straight in front of the driver seat
(as described), so yaw=0 means facing the road and a single symmetric
left/right threshold applies without extra calibration.

If MediaPipe is not installed or no face is detected, this module reports
that clearly rather than fabricating a pose.
"""

import numpy as np
import cv2
from collections import deque

from src.event_manager import TemporalFlag
import config

try:
    import mediapipe as mp
    # Some recent MediaPipe releases (0.10.14+) dropped the legacy
    # `mp.solutions.face_mesh` Python API in favor of the newer Tasks API.
    # We require the legacy solutions API here; if it isn't present we
    # treat MediaPipe as unavailable rather than crashing at import time.
    _ = mp.solutions.face_mesh  # noqa: B018 - existence check only
    _MEDIAPIPE_AVAILABLE = True
except Exception:
    _MEDIAPIPE_AVAILABLE = False


# 3D model points for a generic face (nose tip, chin, eye corners, mouth
# corners) used for solvePnP. Approximate, in arbitrary units.
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices
_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


class HeadPoseEstimator:
    def __init__(self):
        self.available = _MEDIAPIPE_AVAILABLE
        self.unavailable_reason = None
        if self.available:
            try:
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                self.available = False
                self.unavailable_reason = str(e)
        else:
            self.unavailable_reason = (
                "mediapipe is installed but its legacy 'solutions.face_mesh' API is not "
                "available in this version. Either pin mediapipe<0.10.14 "
                "(pip install 'mediapipe<0.10.14'), or port this module to the newer "
                "Tasks API (mediapipe.tasks.python.vision.FaceLandmarker) — see "
                "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
            )

        # Continuous-look-away flag (default now 10s per user requirement).
        self.flag = TemporalFlag(
            name="DRIVER_LOOKING_AWAY",
            duration_required=config.HEAD_POSE_AWAY_DURATION,
            cooldown=5.0,
        )

        # Frequency-based tracking: timestamps of each separate glance-away
        # START within a rolling window.
        self._glance_start_times = deque()
        self._was_away = False
        self._was_frequent = False

    def _classify(self, yaw, pitch):
        if abs(yaw) < config.HEAD_POSE_YAW_THRESHOLD_DEG and abs(pitch) < config.HEAD_POSE_PITCH_THRESHOLD_DEG:
            return "FORWARD"
        if yaw >= config.HEAD_POSE_YAW_THRESHOLD_DEG:
            return "RIGHT"
        if yaw <= -config.HEAD_POSE_YAW_THRESHOLD_DEG:
            return "LEFT"
        if pitch >= config.HEAD_POSE_PITCH_THRESHOLD_DEG:
            return "DOWN"
        if pitch <= -config.HEAD_POSE_PITCH_THRESHOLD_DEG:
            return "UP"
        return "FORWARD"

    def _register_glance(self, away: bool, video_time: float):
        """Tracks distinct look-away glances (any length) for the
        frequency-based 'looking away often' signal, independent of
        whether any single glance reaches the 10s continuous threshold."""
        if away and not self._was_away:
            self._glance_start_times.append(video_time)
        self._was_away = away
        cutoff = video_time - config.HEAD_POSE_FREQUENT_GLANCE_WINDOW
        while self._glance_start_times and self._glance_start_times[0] < cutoff:
            self._glance_start_times.popleft()

    @property
    def frequent_glances_count(self) -> int:
        return len(self._glance_start_times)

    def _frequent_event(self, video_time: float):
        frequent = self.frequent_glances_count >= config.HEAD_POSE_FREQUENT_GLANCE_COUNT
        event = None
        if frequent and not self._was_frequent:
            event = "START"
        elif not frequent and self._was_frequent:
            event = "END"
        self._was_frequent = frequent
        return frequent, event

    def process_frame(self, frame_bgr, video_time: float):
        if not self.available:
            return {
                "pose": "UNAVAILABLE", "event": None, "frequent_event": None,
                "sustained_active": False, "frequent_glances": False,
                "error": self.unavailable_reason,
            }

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            # No face -> cannot classify pose this frame; do not fabricate.
            # Conservatively treat as "not away" for the frequency counter
            # so face-detection dropouts don't inflate the glance count.
            self._register_glance(False, video_time)
            frequent, frequent_event = self._frequent_event(video_time)
            event = self.flag.update(False, video_time)
            return {
                "pose": "NO_FACE", "event": event, "frequent_event": frequent_event,
                "sustained_active": self.flag.is_active or frequent,
                "frequent_glances": frequent,
            }

        landmarks = results.multi_face_landmarks[0].landmark
        image_points = np.array([
            (landmarks[i].x * w, landmarks[i].y * h) for i in _LANDMARK_IDS
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _translation_vec = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            self._register_glance(False, video_time)
            frequent, frequent_event = self._frequent_event(video_time)
            event = self.flag.update(False, video_time)
            return {
                "pose": "UNKNOWN", "event": event, "frequent_event": frequent_event,
                "sustained_active": self.flag.is_active or frequent,
                "frequent_glances": frequent,
            }

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        euler_angles = np.asarray(euler_angles).flatten()  # (3,1) -> (3,) so each entry is a scalar
        pitch, yaw, _roll = [float(a) for a in euler_angles]

        pose = self._classify(yaw, pitch)
        # "Not focused" is deliberately LEFT/RIGHT only, per requirement --
        # UP/DOWN are reported in `pose` (useful context, e.g. looking at a
        # dashboard/phone in lap) but do not drive this signal, since a
        # brief glance down at instruments is a different failure mode than
        # turning to talk to a passenger or check a blind spot.
        away = pose in ("LEFT", "RIGHT")

        self._register_glance(away, video_time)
        frequent, frequent_event = self._frequent_event(video_time)
        event = self.flag.update(away, video_time, confidence=1.0)

        return {
            "pose": pose,
            "yaw": yaw,
            "pitch": pitch,
            "event": event,                    # START/END for the 10s-continuous case
            "frequent_event": frequent_event,   # START/END for the "looking away often" case
            "frequent_glances": frequent,
            "glance_count": self.frequent_glances_count,
            "sustained_active": self.flag.is_active or frequent,  # combined "not focused" signal
        }