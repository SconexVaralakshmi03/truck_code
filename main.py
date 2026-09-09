"""
main.py
=======
Live, multi-camera backend for the Truck Driver Safety POC.

WHAT THIS FILE IS
------------------
This is the ONLY new/changed file. It does NOT modify:
    - src/drowsiness.py
    - src/drowsiness_geometric.py
    - src/phone_detection.py
    - src/detector_base.py, src/event_manager.py, src/pose_utils.py, ...
    - config.py
    - drowsiness_mobile_poc.py   (its TemporalViolation class is imported
                                   and reused as-is, not rewritten)

All detection logic (classifier + MediaPipe EAR/MAR/PERCLOS geometric
cross-check for drowsiness, classifier + YOLOv8 + posture cascade for
phone usage, and the START/sustained/END/cooldown temporal state machine)
is untouched. This file only adds the *live* API/serving layer on top of
that existing logic, using the same WebSocket protocol style as the
reference implementation you supplied (START_STREAM registration,
STREAM_STARTED ack, DETECTION_STATUS heartbeats, *_ALERT events, /health).

WHAT'S NEW HERE
----------------
1. TRUE PARALLELISM, 2 DEDICATED WORKERS
   Two separate OS processes are started at server startup:
     - one process runs ONLY drowsiness detection (classifier + geometric)
     - one process runs ONLY phone-usage detection (classifier + YOLO +
       posture)
   Because they are separate processes (not threads), they run on
   separate CPU cores in true parallel -- the phone detector is never
   blocked waiting on the drowsiness detector or vice versa, so alerts
   for each are produced as fast as each pipeline allows.

2. MULTI-CAMERA (up to MAX_CONCURRENT_CAMERAS, default 3)
   Every camera connection gets its own fully independent detector state
   (its own EAR baseline, its own temporal START/END timers, its own
   cooldowns) inside each worker process -- exactly the same isolation
   the original single-video script had per video, just keyed by
   camera_id instead of by file. Camera A's drowsy timer can never bleed
   into camera B's.

3. PER-CAMERA ROUTING
   Every frame carries its connection_id/camera_id all the way through
   both worker processes and back. A background dispatcher thread reads
   each worker's result queue and pushes the result straight into that
   specific camera's own outgoing asyncio.Queue, which a per-connection
   sender task immediately flushes to that camera's own WebSocket. There
   is no polling delay and no cross-camera mixing -- each camera gets its
   alert the moment its own detection result is ready.

HOW TO RUN
----------
    pip install -r requirements.txt
    python main.py
        (starts uvicorn on 0.0.0.0:8000, same as: uvicorn main:app --host 0.0.0.0 --port 8000)

WebSocket clients connect exactly like the reference implementation:
    ws://<host>:8000/video
    -> send JSON  {"type": "START_STREAM", "user_id": "...", "user_name": "...", "camera_id": "..."}
    <- receive JSON {"type": "STREAM_STARTED", ...}
    -> send binary JPEG frames, one per message, forever
    <- receive JSON status/alert messages as they occur
"""

from __future__ import annotations

import os
import sys
import time
import queue
import asyncio
import threading
import multiprocessing as mp
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


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

# How many camera connections are allowed at the same time.
MAX_CONCURRENT_CAMERAS = int(os.environ.get("MAX_CONCURRENT_CAMERAS", "3"))

# How frequently a heartbeat/normal status is sent per camera per domain.
NORMAL_STATUS_INTERVAL_SECONDS = 0.5

# Bounded queues so a slow/backed-up worker can never grow memory without
# limit or stall the event loop. If a queue is full we drop that single
# frame for that domain only (the other domain still gets it) rather than
# blocking the WebSocket receive loop.
TASK_QUEUE_MAXSIZE = 12


# =============================================================================
# MULTIPROCESSING PRIMITIVES (created at import time; workers are started
# in the FastAPI lifespan handler below)
# =============================================================================

task_queue_drowsy: mp.Queue = mp.Queue(maxsize=TASK_QUEUE_MAXSIZE)
task_queue_phone: mp.Queue = mp.Queue(maxsize=TASK_QUEUE_MAXSIZE)

result_queue_drowsy: mp.Queue = mp.Queue()
result_queue_phone: mp.Queue = mp.Queue()

drowsy_process: Optional[mp.Process] = None
phone_process: Optional[mp.Process] = None

drowsy_dispatcher_thread: Optional[threading.Thread] = None
phone_dispatcher_thread: Optional[threading.Thread] = None


# =============================================================================
# WORKER 1 -- DROWSINESS ONLY (runs in its own process)
# =============================================================================

def drowsiness_worker_process(task_q, result_q, drowsy_model_path, device):
    """
    Dedicated worker process for DROWSINESS detection only.

    Reuses, unmodified:
        src.drowsiness.DrowsinessDetector          (classifier)
        src.drowsiness_geometric.GeometricDrowsinessDetector (EAR/MAR/PERCLOS)
        drowsiness_mobile_poc.TemporalViolation      (combine + START/END)
        config.DROWSINESS_COMBINE_MODE / DROWSINESS_COOLDOWN

    Maintains one full detector set PER camera_id, created lazily on that
    camera's first frame, so every camera keeps a completely independent
    EAR baseline and temporal timer -- identical isolation to the original
    single-video POC script, just keyed by camera instead of by file.
    """
    print(f"[drowsiness-worker pid={os.getpid()}] starting", flush=True)

    from src.drowsiness import DrowsinessDetector
    from src.drowsiness_geometric import GeometricDrowsinessDetector
    from drowsiness_mobile_poc import TemporalViolation
    import config as cfg

    camera_state = {}

    def get_state(camera_id):
        if camera_id not in camera_state:
            classifier = None
            if drowsy_model_path and os.path.exists(drowsy_model_path):
                try:
                    classifier = DrowsinessDetector(drowsy_model_path, device=device)
                    print(
                        f"[drowsiness-worker] classifier loaded for camera '{camera_id}'",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[drowsiness-worker] classifier disabled for "
                        f"'{camera_id}': {exc!r}",
                        flush=True,
                    )
            else:
                print(
                    f"[drowsiness-worker] no drowsiness model found, "
                    f"'{camera_id}' will use geometric detector only",
                    flush=True,
                )

            camera_state[camera_id] = {
                "classifier": classifier,
                "geometric": GeometricDrowsinessDetector(),
                "temporal": TemporalViolation(
                    "DROWSINESS", 0.0, cfg.DROWSINESS_COOLDOWN
                ),
                "last_status_at": 0.0,
            }
        return camera_state[camera_id]

    while True:
        task = task_q.get()  # blocks until a frame (or control/shutdown) arrives

        if task is None:
            print("[drowsiness-worker] shutdown signal received", flush=True)
            break

        if task.get("control") == "REMOVE_CAMERA":
            camera_state.pop(task["camera_id"], None)
            print(
                f"[drowsiness-worker] released state for camera "
                f"'{task['camera_id']}'",
                flush=True,
            )
            continue

        camera_id = task["camera_id"]

        try:
            frame = cv2.imdecode(
                np.frombuffer(task["frame_bytes"], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                continue

            video_time = task["video_time"]
            state = get_state(camera_id)

            if state["classifier"] is not None:
                d = state["classifier"].process_frame(frame, video_time)
            else:
                d = {
                    "label": "DISABLED",
                    "confidence": 0.0,
                    "sustained_active": False,
                    "event": None,
                }

            geo = state["geometric"].process_frame(frame, video_time)

            classifier_active = bool(d.get("sustained_active", False))
            geometric_active = bool(geo.get("sustained_active", False))

            if cfg.DROWSINESS_COMBINE_MODE.upper() == "AND":
                condition = classifier_active and geometric_active
            else:
                # Default POC behavior: either signal triggers.
                condition = classifier_active or geometric_active

            confidence = float(d.get("confidence", 0.0) or 0.0)
            event = state["temporal"].update(condition, video_time, confidence)

            now = time.monotonic()
            is_heartbeat_due = (
                now - state["last_status_at"] >= NORMAL_STATUS_INTERVAL_SECONDS
            )

            if event is None and not is_heartbeat_due:
                continue  # nothing worth sending yet

            state["last_status_at"] = now

            if event == "START":
                print(
                    f"[drowsiness-worker] 🚨 DROWSINESS START camera="
                    f"{camera_id} conf={confidence:.3f}",
                    flush=True,
                )

            result_q.put(
                {
                    "domain": "drowsiness",
                    "connection_id": task["connection_id"],
                    "camera_id": camera_id,
                    "user_name": task["user_name"],
                    "event": event,
                    "active": state["temporal"].active,
                    "label": d.get("label"),
                    "confidence": round(confidence, 4),
                    "peak_confidence": round(state["temporal"].peak_confidence, 4),
                    "ear": geo.get("ear"),
                    "perclos": geo.get("perclos"),
                }
            )

        except Exception as exc:
            print(
                f"[drowsiness-worker] ERROR on camera '{camera_id}': {exc!r}",
                flush=True,
            )


# =============================================================================
# WORKER 2 -- PHONE USAGE ONLY (runs in its own process)
# =============================================================================

def phone_worker_process(task_q, result_q, phone_model_path, phone_yolo_path, device):
    """
    Dedicated worker process for PHONE-USAGE detection only.

    Reuses, unmodified:
        src.phone_detection.PhoneDetector    (classifier -> YOLOv8 -> posture cascade)
        drowsiness_mobile_poc.TemporalViolation
        config.PHONE_COOLDOWN

    Maintains one PhoneDetector PER camera_id, same isolation reasoning as
    the drowsiness worker above.
    """
    print(f"[phone-worker pid={os.getpid()}] starting", flush=True)

    from src.phone_detection import PhoneDetector
    from drowsiness_mobile_poc import TemporalViolation
    import config as cfg

    if phone_yolo_path:
        cfg.PHONE_YOLO_FALLBACK_MODEL = str(phone_yolo_path)

    camera_state = {}

    def get_state(camera_id):
        if camera_id not in camera_state:
            model_path = (
                phone_model_path
                if phone_model_path and os.path.exists(phone_model_path)
                else None
            )
            detector = PhoneDetector(model_path, device=device)
            camera_state[camera_id] = {
                "detector": detector,
                "temporal": TemporalViolation("PHONE_USAGE", 0.0, cfg.PHONE_COOLDOWN),
                "last_status_at": 0.0,
            }
            print(
                f"[phone-worker] detector initialized for camera '{camera_id}'",
                flush=True,
            )
        return camera_state[camera_id]

    while True:
        task = task_q.get()

        if task is None:
            print("[phone-worker] shutdown signal received", flush=True)
            break

        if task.get("control") == "REMOVE_CAMERA":
            camera_state.pop(task["camera_id"], None)
            print(
                f"[phone-worker] released state for camera '{task['camera_id']}'",
                flush=True,
            )
            continue

        camera_id = task["camera_id"]

        try:
            frame = cv2.imdecode(
                np.frombuffer(task["frame_bytes"], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                continue

            video_time = task["video_time"]
            state = get_state(camera_id)

            p = state["detector"].process_frame(frame, video_time)

            condition = bool(p.get("sustained_active", False))
            confidence = float(p.get("confidence", 0.0) or 0.0)
            event = state["temporal"].update(condition, video_time, confidence)

            now = time.monotonic()
            is_heartbeat_due = (
                now - state["last_status_at"] >= NORMAL_STATUS_INTERVAL_SECONDS
            )

            if event is None and not is_heartbeat_due:
                continue

            state["last_status_at"] = now

            if event == "START":
                print(
                    f"[phone-worker] 🚨 PHONE_USAGE START camera={camera_id} "
                    f"mode={p.get('mode')} conf={confidence:.3f}",
                    flush=True,
                )

            result_q.put(
                {
                    "domain": "phone",
                    "connection_id": task["connection_id"],
                    "camera_id": camera_id,
                    "user_name": task["user_name"],
                    "event": event,
                    "active": state["temporal"].active,
                    "label": p.get("label"),
                    "mode": p.get("mode"),
                    "confidence": round(confidence, 4),
                    "peak_confidence": round(state["temporal"].peak_confidence, 4),
                }
            )

        except Exception as exc:
            print(
                f"[phone-worker] ERROR on camera '{camera_id}': {exc!r}",
                flush=True,
            )


# =============================================================================
# RESULT -> WEBSOCKET MESSAGE FORMATTING
# (message "shape" follows the same style as the reference implementation:
#  type / state / user_name / camera_id / timestamp)
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

active_clients: dict[int, dict] = {}          # connection_id -> info
connection_out_queues: dict[int, "asyncio.Queue"] = {}  # connection_id -> per-camera outbox
active_camera_ids: set[str] = set()

total_connections = 0
active_connections = 0
total_frames_received = 0
total_frames_processed = 0
total_alerts_sent = 0
total_frames_dropped_drowsy = 0
total_frames_dropped_phone = 0


# =============================================================================
# DISPATCHER THREADS
# Bridge a (blocking) multiprocessing result queue into the specific
# camera's asyncio.Queue on the event loop thread. This is what makes each
# camera's alert appear "at that camera's exact point in time" instead of
# on some shared polling interval: the moment a worker process finishes a
# frame for camera X, the result is routed straight to camera X's own
# outgoing queue and flushed to its own WebSocket.
# =============================================================================

def start_dispatcher(result_q, loop: asyncio.AbstractEventLoop, label: str) -> threading.Thread:
    def _run():
        global total_alerts_sent
        while True:
            item = result_q.get()
            if item is None:
                print(f"[dispatcher-{label}] shutdown signal received", flush=True)
                break

            connection_id = item.get("connection_id")

            with state_lock:
                out_q = connection_out_queues.get(connection_id)

            if out_q is None:
                # Camera already disconnected; drop the stale result.
                continue

            if item.get("event") == "START":
                with state_lock:
                    total_alerts_sent += 1

            message = build_message(item)
            asyncio.run_coroutine_threadsafe(out_q.put(message), loop)

    t = threading.Thread(target=_run, daemon=True, name=f"dispatcher-{label}")
    t.start()
    return t


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global drowsy_process, phone_process
    global drowsy_dispatcher_thread, phone_dispatcher_thread

    loop = asyncio.get_running_loop()

    drowsy_process = mp.Process(
        target=drowsiness_worker_process,
        args=(task_queue_drowsy, result_queue_drowsy, DROWSINESS_MODEL_PATH, DEVICE),
        name="drowsiness-worker",
        daemon=True,
    )
    phone_process = mp.Process(
        target=phone_worker_process,
        args=(
            task_queue_phone,
            result_queue_phone,
            PHONE_MODEL_PATH,
            PHONE_YOLO_FALLBACK_MODEL_PATH,
            DEVICE,
        ),
        name="phone-worker",
        daemon=True,
    )

    drowsy_process.start()
    phone_process.start()

    drowsy_dispatcher_thread = start_dispatcher(result_queue_drowsy, loop, "drowsiness")
    phone_dispatcher_thread = start_dispatcher(result_queue_phone, loop, "phone")

    print(f"[main] drowsiness worker PID={drowsy_process.pid}", flush=True)
    print(f"[main] phone worker PID={phone_process.pid}", flush=True)
    print(f"[main] max concurrent cameras = {MAX_CONCURRENT_CAMERAS}", flush=True)

    yield

    print("[main] shutting down...", flush=True)

    for q in (task_queue_drowsy, task_queue_phone):
        try:
            q.put_nowait(None)
        except Exception:
            pass

    for p in (drowsy_process, phone_process):
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    for q in (result_queue_drowsy, result_queue_phone):
        try:
            q.put(None)
        except Exception:
            pass


app = FastAPI(
    title="Multi-Camera Driver Drowsiness + Phone-Usage Backend",
    version="4.0.0",
    lifespan=lifespan,
)

# Allow the frontend to connect through a tunnel/public URL such as
# wss://<ngrok-domain>.ngrok-free.dev and still hit the same FastAPI
# websocket endpoint. The backend must not lock itself to localhost-only
# browser origin policy while running behind a public proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@app.get("/")
async def home():
    public_ws_base = os.environ.get("PUBLIC_WS_BASE", "ws://localhost:8000")
    return {
        "status": "success",
        "message": "Multi-camera drowsiness + phone-usage backend is running",
        "workers": {
            "drowsiness": "dedicated process (parallel)",
            "phone": "dedicated process (parallel)",
        },
        "max_concurrent_cameras": MAX_CONCURRENT_CAMERAS,
        "websocket_endpoint": "/video",
        "example_frontend_url": f"{public_ws_base}/video",
    }


@app.get("/health")
async def health():
    with state_lock:
        cameras = sorted(active_camera_ids)
        clients = dict(active_clients)
        alerts_sent = total_alerts_sent

    def _qsize(q):
        try:
            return q.qsize()
        except NotImplementedError:
            return None

    public_ws_base = os.environ.get("PUBLIC_WS_BASE", "ws://localhost:8000")

    return {
        "status": "ok",
        "active_connections": active_connections,
        "total_connections": total_connections,
        "frames_received": total_frames_received,
        "frames_processed": total_frames_processed,
        "alerts_sent": alerts_sent,
        "frames_dropped": {
            "drowsiness_queue_full": total_frames_dropped_drowsy,
            "phone_queue_full": total_frames_dropped_phone,
        },
        "active_cameras": cameras,
        "clients": clients,
        "queue_depth": {
            "drowsiness_tasks": _qsize(task_queue_drowsy),
            "phone_tasks": _qsize(task_queue_phone),
        },
        "worker_pids": {
            "drowsiness": drowsy_process.pid if drowsy_process else None,
            "phone": phone_process.pid if phone_process else None,
        },
        "public_stream_hint": f"{public_ws_base}/video",
    }


# =============================================================================
# WEBSOCKET VIDEO ENDPOINT (per-camera connection)
# =============================================================================

@app.websocket("/video")
async def video_receiver(websocket: WebSocket):
    global total_connections, active_connections
    global total_frames_received, total_frames_processed
    global total_frames_dropped_drowsy, total_frames_dropped_phone

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
        # 2. ADMIT / REJECT (duplicate camera_id or too many cameras)
        # ---------------------------------------------------------------
        registration_error = None
        with state_lock:
            if camera_id in active_camera_ids:
                registration_error = f"camera_id '{camera_id}' is already streaming"
            elif len(active_camera_ids) >= MAX_CONCURRENT_CAMERAS:
                registration_error = (
                    f"Maximum of {MAX_CONCURRENT_CAMERAS} concurrent cameras reached"
                )
            else:
                active_camera_ids.add(camera_id)

        if registration_error:
            await websocket.send_json({"type": "ERROR", "message": registration_error})
            await websocket.close()
            return

        out_queue: asyncio.Queue = asyncio.Queue()
        with state_lock:
            connection_out_queues[connection_id] = out_queue
            active_clients[connection_id] = {
                "user_id": user_id,
                "user_name": user_name,
                "camera_id": camera_id,
                "connected_at": get_timestamp(),
            }

        print("-" * 80)
        print("STREAM REGISTERED")
        print(f"Connection : {connection_id}")
        print(f"User ID    : {user_id}")
        print(f"User Name  : {user_name}")
        print(f"Camera ID  : {camera_id}")
        print(f"Active cameras now: {sorted(active_camera_ids)}")
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
        # 3. SENDER TASK
        # Delivers this camera's own alerts/status the instant a worker
        # process produces them, independent of the receive loop below.
        # ---------------------------------------------------------------
        async def sender_loop():
            while True:
                message = await out_queue.get()
                await websocket.send_json(message)

        sender_task = asyncio.create_task(sender_loop())

        # ---------------------------------------------------------------
        # 4. RECEIVE CONTINUOUS JPEG FRAMES
        # Each frame is fanned out to BOTH worker queues so drowsiness and
        # phone detection run in parallel for the same frame.
        # ---------------------------------------------------------------
        while True:
            data = await websocket.receive_bytes()
            total_frames_received += 1

            # Cheap sanity check without a full decode (decode happens
            # inside each worker process so it also benefits from the
            # two-process parallelism instead of serializing on the
            # single asyncio event loop).
            if not data:
                continue

            total_frames_processed += 1
            video_time = time.monotonic() - connection_start

            task = {
                "connection_id": connection_id,
                "camera_id": camera_id,
                "user_name": user_name,
                "frame_bytes": data,
                "video_time": video_time,
            }

            try:
                task_queue_drowsy.put_nowait(task)
            except queue.Full:
                total_frames_dropped_drowsy += 1

            try:
                task_queue_phone.put_nowait(task)
            except queue.Full:
                total_frames_dropped_phone += 1

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
            connection_out_queues.pop(connection_id, None)
            active_camera_ids.discard(camera_id)

        # Free that camera's model/timer state inside both worker
        # processes so it doesn't sit in memory forever.
        if camera_id:
            control = {"control": "REMOVE_CAMERA", "camera_id": camera_id}
            for q in (task_queue_drowsy, task_queue_phone):
                try:
                    q.put_nowait(control)
                except queue.Full:
                    pass

        print(f"Active connections: {active_connections}")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    # "spawn" is the safe cross-platform choice (required on Windows, and
    # avoids fork-related surprises with OpenCV/PyTorch/MediaPipe threads
    # on Linux/macOS too).
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set (e.g. re-imported)

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )