"""
src/video_utils.py
===================
Thin wrappers around OpenCV VideoCapture/VideoWriter plus small helpers
for timestamps and overlay text. Windows-compatible (uses 'mp4v' fourcc
and normal os.path handling — no POSIX-only assumptions).
"""

import os
import time
import cv2


class VideoSource:
    """Wraps an MP4 file. Never opens a webcam (index-based source)."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        if not path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            raise ValueError(f"Only prerecorded video files are supported. Got: {path}")

        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video file: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.fps <= 0:
            # Some containers report 0 fps; fall back to a sane default
            # rather than dividing by zero later.
            self.fps = 25.0

        self.duration_sec = self.frame_count / self.fps if self.fps else 0.0

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoSink:
    """Wraps an OpenCV VideoWriter for saving annotated MP4 output."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps if fps > 0 else 25.0, (width, height))
        self.path = path
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"[:-4]  # trim to HH:MM:SS.mm


def format_timestamp_short(seconds: float) -> str:
    """Format seconds as HH:MM:SS (no milliseconds), for on-video overlay."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class FPSMeter:
    """Tracks wall-clock processing FPS (distinct from the source video's FPS)."""

    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self._last_tick = self.start_time

    def tick(self):
        self.frame_count += 1
        self._last_tick = time.time()

    @property
    def elapsed(self):
        return self._last_tick - self.start_time

    @property
    def average_fps(self):
        return self.frame_count / self.elapsed if self.elapsed > 0 else 0.0
