"""
main.py
=======
Live, multi-camera backend for the Truck Driver Safety POC.

WHAT THIS FILE IS
------------------
This is the ONLY new/changed file. It does NOT modify:
    - src/drowsiness.py, src/drowsiness_geometric.py, src/phone_detection.py
    - src/detector_base.py, src/event_manager.py, src/pose_utils.py, ...
    - config.py
    - drowsiness_mobile_poc.py   (its TemporalViolation class is imported
                                   and reused as-is, not rewritten)

ARCHITECTURE (v5 -- one dedicated GPU worker process per camera)
------------------------------------------------------------------
Earlier versions of this file used exactly 2 worker processes total (one
for drowsiness, one for phone), each serving every connected camera in
turn. That meant 3 cameras streaming at once would queue up behind each
other inside a single process.

This version flips that around, per your request:

    1 camera stream == 1 dedicated OS process (a "camera worker"),
    started the moment that camera connects and torn down the moment
    it disconnects.

    Up to MAX_CONCURRENT_CAMERAS such processes run at once (default 3,
    set the env var to 4 or more if your GPU has room -- see below).
    Because each is a real OS process, Driver A's stream is never stuck
    waiting behind Driver B's or Driver C's frames -- all cameras are
    processed truly in parallel, each getting alerts the instant its own
    worker produces them.

    INSIDE each camera worker, the two detection pipelines also run in
    parallel with each other, via a 2-thread pool:
        thread 1: DrowsinessDetector + GeometricDrowsinessDetector
                  (combined exactly per config.DROWSINESS_COMBINE_MODE)
        thread 2: PhoneDetector (classifier -> YOLOv8 -> posture cascade)
    Both threads work on the SAME frame concurrently. torch/opencv/
    mediapipe/ultralytics all release the GIL during their heavy
    C/CUDA compute, so these two threads genuinely overlap instead of
    serializing -- especially important on a GPU worker, where each
    thread's "work" is mostly waiting on a CUDA kernel/queue rather than
    holding the GIL. Whichever pipeline finishes first (e.g. drowsiness)
    has its alert pushed out immediately -- it does not wait for the
    slower pipeline to also finish.

So for 3 (or 4) simultaneous drivers, you get 3 (or 4) processes x 2
threads each = true stream-level AND detector-level parallelism, and a
drowsiness alert for Driver B can never be delayed by a phone-detection
computation for Driver A or C.

HOW TO RUN
----------
    pip install -r requirements.txt
    python main.py
        (or: uvicorn main:app --host 0.0.0.0 --port 8000)

Set MAX_CONCURRENT_CAMERAS=4 (or higher) as an env var to raise the cap,
and DRIVER_SAFETY_DEVICE=cuda to run each worker's models on GPU.

WebSocket protocol (same style you've been using):
    ws(s)://<host>:8000/video
    -> {"type": "START_STREAM", "user_id": "...", "user_name": "...", "camera_id": "..."}
    <- {"type": "STREAM_STARTED", ...}
    -> binary JPEG frames, one per message, forever
    <- JSON status/alert messages as they occur
"""

from __future__ import annotations

import os
import sys
import time
import queue
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


# =============================================================================
# PROJECT PATH SETUP (same pattern as drowsiness_mobile_poc.py)
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# =============================================================================
# CONFIGURATION (server/API layer only -- detection thresholds themselves
# still live in config.py and are read from there, untouched)
# =============================================================================

DEVICE = os.environ.get("DRIVER_SAFETY_DEVICE", "cpu")  # "cpu" or "cuda"

DROWSINESS_MODEL_PATH = os.environ.get(
    "DROWSINESS_MODEL_PATH", str(SCRIPT_DIR / "drowsiness.pt")
)
PHONE_MODEL_PATH = os.environ.get(
    "PHONE_MODEL_PATH", str(SCRIPT_DIR / "phone.pt")
)
PHONE_YOLO_FALLBACK_MODEL_PATH = os.environ.get(
    "PHONE_YOLO_FALLBACK_MODEL_PATH", str(SCRIPT_DIR / "yolov8m.pt")
)

# How many camera workers (== how many simultaneous drivers) are allowed
# at once. Start at 3, raise to 4+ once you've confirmed your GPU has
# memory headroom for another full set of models.
MAX_CONCURRENT_CAMERAS = int(os.environ.get("MAX_CONCURRENT_CAMERAS", "3"))

# How frequently a heartbeat/normal status is sent per camera per domain.
NORMAL_STATUS_INTERVAL_SECONDS = 0.5

# Bounded per-camera task queue so a stalled worker can't grow memory
# without limit or stall the WebSocket receive loop.
TASK_QUEUE_MAXSIZE = 12


# =============================================================================
# CAMERA WORKER -- one dedicated process per connected camera
# =============================================================================

def camera_worker_process(
    task_q,
    result_q,
    camera_id: str,
    user_name: str,
    drowsy_model_path: str,
    phone_model_path: str,
    phone_yolo_path: str,
    device: str,
):
    """
    Runs for the entire lifetime of ONE camera connection. Loads its own
    copy of every detector once, then for every incoming frame runs the
    drowsiness pipeline and the phone pipeline in parallel threads,
    pushing each domain's result to result_q the instant it's ready.

    Reuses, unmodified:
        src.drowsiness.DrowsinessDetector
        src.drowsiness_geometric.GeometricDrowsinessDetector
        src.phone_detection.PhoneDetector
        drowsiness_mobile_poc.TemporalViolation
        config.DROWSINESS_COMBINE_MODE / DROWSINESS_COOLDOWN / PHONE_COOLDOWN
    """
    print(f"[camera-worker '{camera_id}' pid={os.getpid()}] starting", flush=True)

    from src.drowsiness import DrowsinessDetector
    from src.drowsiness_geometric import GeometricDrowsinessDetector
    from src.phone_detection import PhoneDetector
    from drowsiness_mobile_poc import TemporalViolation
    import config as cfg

    if phone_yolo_path:
        cfg.PHONE_YOLO_FALLBACK_MODEL = str(phone_yolo_path)

    # ---- load models once for this camera -------------------------------
    classifier = None
    if drowsy_model_path and os.path.exists(drowsy_model_path):
        try:
            classifier = DrowsinessDetector(drowsy_model_path, device=device)
        except Exception as exc:
            print(f"[camera-worker '{camera_id}'] drowsiness classifier disabled: {exc!r}", flush=True)

    geometric = GeometricDrowsinessDetector()

    phone_model = phone_model_path if phone_model_path and os.path.exists(phone_model_path) else None
    phone_detector = PhoneDetector(phone_model, device=device)

    drowsy_temporal = TemporalViolation("DROWSINESS", 0.0, cfg.DROWSINESS_COOLDOWN)
    phone_temporal = TemporalViolation("PHONE_USAGE", 0.0, cfg.PHONE_COOLDOWN)

    last_status_at = {"drowsiness": 0.0, "phone": 0.0}

    # ---- per-domain pipelines, each run on its own thread ----------------
    def run_drowsiness(frame, video_time):
        if classifier is not None:
            d = classifier.process_frame(frame, video_time)
        else:
            d = {"label": "DISABLED", "confidence": 0.0, "sustained_active": False}

        geo = geometric.process_frame(frame, video_time)

        classifier_active = bool(d.get("sustained_active", False))
        geometric_active = bool(geo.get("sustained_active", False))

        if cfg.DROWSINESS_COMBINE_MODE.upper() == "AND":
            condition = classifier_active and geometric_active
        else:
            condition = classifier_active or geometric_active

        confidence = float(d.get("confidence", 0.0) or 0.0)
        event = drowsy_temporal.update(condition, video_time, confidence)

        now = time.monotonic()
        due = now - last_status_at["drowsiness"] >= NORMAL_STATUS_INTERVAL_SECONDS
        if event is None and not due:
            return None
        last_status_at["drowsiness"] = now

        if event == "START":
            print(
                f"[camera-worker '{camera_id}'] \U0001f6a8 DROWSINESS START "
                f"conf={confidence:.3f}",
                flush=True,
            )

        return {
            "domain": "drowsiness",
            "camera_id": camera_id,
            "user_name": user_name,
            "event": event,
            "active": drowsy_temporal.active,
            "label": d.get("label"),
            "confidence": round(confidence, 4),
            "peak_confidence": round(drowsy_temporal.peak_confidence, 4),
            "ear": geo.get("ear"),
            "perclos": geo.get("perclos"),
        }

    def run_phone(frame, video_time):
        p = phone_detector.process_frame(frame, video_time)

        condition = bool(p.get("sustained_active", False))
        confidence = float(p.get("confidence", 0.0) or 0.0)
        event = phone_temporal.update(condition, video_time, confidence)

        now = time.monotonic()
        due = now - last_status_at["phone"] >= NORMAL_STATUS_INTERVAL_SECONDS
        if event is None and not due:
            return None
        last_status_at["phone"] = now

        if event == "START":
            print(
                f"[camera-worker '{camera_id}'] \U0001f6a8 PHONE_USAGE START "
                f"mode={p.get('mode')} conf={confidence:.3f}",
                flush=True,
            )

        return {
            "domain": "phone",
            "camera_id": camera_id,
            "user_name": user_name,
            "event": event,
            "active": phone_temporal.active,
            "label": p.get("label"),
            "mode": p.get("mode"),
            "confidence": round(confidence, 4),
            "peak_confidence": round(phone_temporal.peak_confidence, 4),
        }

    # ---- main loop: 1 frame in, both pipelines run concurrently ---------
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"det-{camera_id}") as pool:
        while True:
            task = task_q.get()  # blocks until a frame or shutdown sentinel

            if task is None:
                print(f"[camera-worker '{camera_id}'] shutdown signal received", flush=True)
                break

            try:
                frame = cv2.imdecode(
                    np.frombuffer(task["frame_bytes"], dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    continue

                video_time = task["video_time"]

                futures = {
                    pool.submit(run_drowsiness, frame, video_time): "drowsiness",
                    pool.submit(run_phone, frame, video_time): "phone",
                }

                # Push each domain's result the instant IT finishes -- the
                # faster pipeline never waits on the slower one.
                for future in as_completed(futures):
                    try:
                        item = future.result()
                    except Exception as exc:
                        print(
                            f"[camera-worker '{camera_id}'] "
                            f"{futures[future]} pipeline error: {exc!r}",
                            flush=True,
                        )
                        continue
                    if item is not None:
                        result_q.put(item)

            except Exception as exc:
                print(f"[camera-worker '{camera_id}'] frame error: {exc!r}", flush=True)

    print(f"[camera-worker '{camera_id}' pid={os.getpid()}] exiting", flush=True)


# =============================================================================
# RESULT -> WEBSOCKET MESSAGE FORMATTING
# =============================================================================

def get_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_message(item: dict) -> dict:
    domain = item["domain"]  # "drowsiness" or "phone"
    event = item.get("event")

    if domain == "drowsiness":
        if event == "START":
            return {
                "type": "DROWSINESS_ALERT",
                "state": "drowsy",
                "user_name": item["user_name"],
                "camera_id": item["camera_id"],
                "timestamp": get_timestamp(),
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                "peak_confidence": item.get("peak_confidence"),
                "ear": item.get("ear"),
                "perclos": item.get("perclos"),
            }
        if event == "END":
            return {
                "type": "DROWSINESS_CLEARED",
                "state": "normal",
                "user_name": item["user_name"],
                "camera_id": item["camera_id"],
                "timestamp": get_timestamp(),
            }
        return {
            "type": "DETECTION_STATUS",
            "domain": "drowsiness",
            "state": "drowsy" if item.get("active") else "normal",
            "user_name": item["user_name"],
            "camera_id": item["camera_id"],
        }

    # domain == "phone"
    if event == "START":
        return {
            "type": "PHONE_ALERT",
            "state": "phone_detected",
            "user_name": item["user_name"],
            "camera_id": item["camera_id"],
            "timestamp": get_timestamp(),
            "label": item.get("label"),
            "mode": item.get("mode"),
            "confidence": item.get("confidence"),
            "peak_confidence": item.get("peak_confidence"),
        }
    if event == "END":
        return {
            "type": "PHONE_CLEARED",
            "state": "normal",
            "user_name": item["user_name"],
            "camera_id": item["camera_id"],
            "timestamp": get_timestamp(),
        }
    return {
        "type": "DETECTION_STATUS",
        "domain": "phone",
        "state": "phone_detected" if item.get("active") else "normal",
        "user_name": item["user_name"],
        "camera_id": item["camera_id"],
    }


# =============================================================================
# GLOBAL SERVER STATE (main process)
# =============================================================================

state_lock = threading.Lock()

active_clients: dict[int, dict] = {}     # connection_id -> info
active_camera_workers: dict[str, dict] = {}  # camera_id -> {process, task_q, result_q}

total_connections = 0
active_connections = 0
total_frames_received = 0
total_frames_processed = 0
total_alerts_sent = 0
total_frames_dropped = 0


# =============================================================================
# PER-CAMERA DISPATCHER THREAD
# One of these is started per camera connection, reading ONLY that camera's
# dedicated result queue and forwarding straight into that connection's own
# asyncio outgoing queue -- no shared routing table needed, because the
# 1-worker-per-camera design already gives 1:1 isolation.
# =============================================================================

def start_camera_dispatcher(
    result_q, out_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, camera_id: str
) -> threading.Thread:
    def _run():
        global total_alerts_sent
        while True:
            item = result_q.get()
            if item is None:
                break
            if item.get("event") == "START":
                with state_lock:
                    total_alerts_sent += 1
            message = build_message(item)
            asyncio.run_coroutine_threadsafe(out_queue.put(message), loop)

    t = threading.Thread(target=_run, daemon=True, name=f"dispatcher-{camera_id}")
    t.start()
    return t


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[main] ready -- max concurrent camera workers = {MAX_CONCURRENT_CAMERAS}", flush=True)
    yield

    print("[main] shutting down -- stopping all camera workers...", flush=True)
    with state_lock:
        workers = list(active_camera_workers.items())

    for camera_id, info in workers:
        try:
            info["task_q"].put_nowait(None)
        except Exception:
            pass

    for camera_id, info in workers:
        info["process"].join(timeout=5)
        if info["process"].is_alive():
            info["process"].terminate()
        try:
            info["result_q"].put(None)
        except Exception:
            pass


app = FastAPI(
    title="Multi-Camera Driver Drowsiness + Phone-Usage Backend",
    version="5.0.0",
    lifespan=lifespan,
)


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@app.get("/")
async def home():
    return {
        "status": "success",
        "message": "Multi-camera drowsiness + phone-usage backend is running",
        "architecture": "1 dedicated GPU worker process per camera; "
        "drowsiness + phone detectors run in parallel threads inside each worker",
        "max_concurrent_cameras": MAX_CONCURRENT_CAMERAS,
        "websocket_endpoint": "/video",
    }


@app.get("/health")
async def health():
    with state_lock:
        clients = dict(active_clients)
        workers = {
            camera_id: {
                "pid": info["process"].pid,
                "alive": info["process"].is_alive(),
            }
            for camera_id, info in active_camera_workers.items()
        }
        alerts_sent = total_alerts_sent

    return {
        "status": "ok",
        "active_connections": active_connections,
        "total_connections": total_connections,
        "frames_received": total_frames_received,
        "frames_processed": total_frames_processed,
        "alerts_sent": alerts_sent,
        "frames_dropped": total_frames_dropped,
        "active_camera_workers": workers,
        "clients": clients,
        "slots_used": f"{len(workers)}/{MAX_CONCURRENT_CAMERAS}",
    }


# =============================================================================
# WEBSOCKET VIDEO ENDPOINT (per-camera connection)
# =============================================================================

@app.websocket("/video")
async def video_receiver(websocket: WebSocket):
    global total_connections, active_connections
    global total_frames_received, total_frames_processed, total_frames_dropped

    await websocket.accept()

    connection_id = id(websocket)
    total_connections += 1
    active_connections += 1

    print()
    print("=" * 80)
    print("NEW FRONTEND CAMERA CONNECTION")
    print(f"Connection ID: {connection_id}")
    print("=" * 80)

    user_id: Optional[str] = None
    user_name: Optional[str] = None
    camera_id: Optional[str] = None

    sender_task: Optional[asyncio.Task] = None
    dispatcher_thread: Optional[threading.Thread] = None
    task_q = None
    result_q = None
    worker_process = None

    connection_start = time.monotonic()

    try:
        # ---------------------------------------------------------------
        # 1. REGISTRATION
        # ---------------------------------------------------------------
        registration = await websocket.receive_json()

        if registration.get("type") != "START_STREAM":
            await websocket.send_json(
                {"type": "ERROR", "message": "First message must be START_STREAM"}
            )
            await websocket.close()
            return

        user_id = registration.get("user_id")
        user_name = registration.get("user_name")
        camera_id = registration.get("camera_id")

        if not user_id:
            await websocket.send_json({"type": "ERROR", "message": "user_id is required"})
            await websocket.close()
            return
        if not user_name:
            await websocket.send_json({"type": "ERROR", "message": "user_name is required"})
            await websocket.close()
            return
        if not camera_id:
            await websocket.send_json({"type": "ERROR", "message": "camera_id is required"})
            await websocket.close()
            return

        # ---------------------------------------------------------------
        # 2. ADMIT / REJECT (duplicate camera_id or all worker slots full)
        # ---------------------------------------------------------------
        registration_error = None
        with state_lock:
            if camera_id in active_camera_workers:
                registration_error = f"camera_id '{camera_id}' is already streaming"
            elif len(active_camera_workers) >= MAX_CONCURRENT_CAMERAS:
                registration_error = (
                    f"All {MAX_CONCURRENT_CAMERAS} camera worker slots are in use"
                )

        if registration_error:
            await websocket.send_json({"type": "ERROR", "message": registration_error})
            await websocket.close()
            return

        # ---------------------------------------------------------------
        # 3. SPIN UP A DEDICATED WORKER PROCESS FOR THIS CAMERA
        # ---------------------------------------------------------------
        task_q = mp.Queue(maxsize=TASK_QUEUE_MAXSIZE)
        result_q = mp.Queue()

        worker_process = mp.Process(
            target=camera_worker_process,
            args=(
                task_q,
                result_q,
                camera_id,
                user_name,
                DROWSINESS_MODEL_PATH,
                PHONE_MODEL_PATH,
                PHONE_YOLO_FALLBACK_MODEL_PATH,
                DEVICE,
            ),
            name=f"camera-worker-{camera_id}",
            daemon=True,
        )
        worker_process.start()

        with state_lock:
            active_camera_workers[camera_id] = {
                "process": worker_process,
                "task_q": task_q,
                "result_q": result_q,
            }
            active_clients[connection_id] = {
                "user_id": user_id,
                "user_name": user_name,
                "camera_id": camera_id,
                "connected_at": get_timestamp(),
                "worker_pid": worker_process.pid,
            }

        print("-" * 80)
        print("STREAM REGISTERED")
        print(f"Connection : {connection_id}")
        print(f"User ID    : {user_id}")
        print(f"User Name  : {user_name}")
        print(f"Camera ID  : {camera_id}")
        print(f"Worker PID : {worker_process.pid}")
        print(f"Slots used : {len(active_camera_workers)}/{MAX_CONCURRENT_CAMERAS}")
        print("-" * 80)

        await websocket.send_json(
            {
                "type": "STREAM_STARTED",
                "state": "normal",
                "user_id": user_id,
                "user_name": user_name,
                "camera_id": camera_id,
            }
        )

        # ---------------------------------------------------------------
        # 4. SENDER TASK + DISPATCHER
        # ---------------------------------------------------------------
        out_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        dispatcher_thread = start_camera_dispatcher(result_q, out_queue, loop, camera_id)

        async def sender_loop():
            while True:
                message = await out_queue.get()
                await websocket.send_json(message)

        sender_task = asyncio.create_task(sender_loop())

        # ---------------------------------------------------------------
        # 5. RECEIVE CONTINUOUS JPEG FRAMES -> this camera's own worker
        # ---------------------------------------------------------------
        while True:
            data = await websocket.receive_bytes()
            total_frames_received += 1

            if not data:
                continue

            total_frames_processed += 1
            video_time = time.monotonic() - connection_start

            task = {"frame_bytes": data, "video_time": video_time}

            try:
                task_q.put_nowait(task)
            except queue.Full:
                total_frames_dropped += 1

    except WebSocketDisconnect:
        print()
        print("=" * 80)
        print("FRONTEND CAMERA DISCONNECTED")
        print(f"Connection : {connection_id}")
        print(f"User       : {user_name}")
        print(f"Camera     : {camera_id}")
        print("=" * 80)

    except Exception as exc:
        print()
        print("=" * 80)
        print("VIDEO RECEIVER ERROR")
        print("=" * 80)
        print(f"Connection : {connection_id}")
        print(f"User       : {user_name}")
        print(f"Camera     : {camera_id}")
        print(f"Error      : {exc!r}")
        print("=" * 80)

    finally:
        active_connections -= 1

        if sender_task is not None:
            sender_task.cancel()

        with state_lock:
            active_clients.pop(connection_id, None)
            active_camera_workers.pop(camera_id, None)

        # Stop this camera's dedicated worker process.
        if task_q is not None:
            try:
                task_q.put_nowait(None)
            except Exception:
                pass

        if worker_process is not None:
            worker_process.join(timeout=5)
            if worker_process.is_alive():
                worker_process.terminate()

        # Stop this camera's dispatcher thread.
        if result_q is not None:
            try:
                result_q.put(None)
            except Exception:
                pass

        print(f"Active connections: {active_connections}")
        print(
            f"Camera worker slots free: "
            f"{MAX_CONCURRENT_CAMERAS - len(active_camera_workers)}/{MAX_CONCURRENT_CAMERAS}"
        )


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    # "spawn" is the safe cross-platform choice (required on Windows, and
    # avoids fork-related surprises with OpenCV/PyTorch/MediaPipe/CUDA on
    # Linux too).
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set (e.g. re-imported)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)