"""
Standalone Driver Violation Detector
====================================

Detects ONLY:
    1. Drowsiness
    2. Mobile/phone-in-hand usage

The detection logic is intentionally taken from the supplied POC:
    DROWSINESS
      - local drowsiness classifier
      - MediaPipe EAR/MAR/PERCLOS geometric detector
      - classifier + geometric result combined using the POC's OR/AND mode

    PHONE
      - local phone classifier (direct)
      - YOLOv8 COCO "cell phone" detector (direct fallback)
      - MediaPipe Pose hand-near-ear/mouth posture fallback

Temporal logic:
    - A violation must remain above the configured confidence/condition
      for the configured duration before START is logged.
    - START is logged once.
    - END is logged once when the violation clears.
    - Cooldown prevents immediate duplicate events.

Outputs are intentionally simple:
    <VIDEO_DIRECTORY>/logs.txt
    <VIDEO_DIRECTORY>/violation_frame_*.jpg
    <OUTPUT_VIDEO_PATH>  (optional annotated video)

IMPORTANT:
    This is POC logic, not a safety-certified/medically validated system.
    Tune thresholds against your actual camera and fleet footage.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2

# ============================================================================
# EDIT THESE PATHS
# ============================================================================

# Input prerecorded driver-facing video.
INPUT_VIDEO_PATH = r"C:\Users\kshar\OneDrive\Desktop\truck_safety_poc\videos\driver\mb_2.mp4"

# Annotated output video. Set to None to disable output-video writing.
OUTPUT_VIDEO_PATH = r"C:\Users\kshar\OneDrive\Desktop\truck_drowsiness.poc.mp4"

# Local model weights from the supplied codebase.
# Leave as None if you want to rely on the YOLOv8 phone fallback.
DROWSINESS_MODEL_PATH = r"C:\Users\kshar\OneDrive\Desktop\truck_safety_poc\models\drowsiness\drowsiness.pt"
PHONE_MODEL_PATH = r"C:\Users\kshar\OneDrive\Desktop\truck_safety_poc\models\phone\phone.pt"

# Optional explicit YOLOv8 fallback model.
# If this file does not exist, "yolov8m.pt" is passed to Ultralytics,
# which may download it automatically the first time.
PHONE_YOLO_FALLBACK_MODEL_PATH = r"models/general/yolov8m.pt"

# The logs.txt and evidence JPG files are written into this same directory.
# None means: same folder as INPUT_VIDEO_PATH.
LOG_DIR = None

# ============================================================================
# DETECTION / TEMPORAL SETTINGS
# These are the values from the supplied POC config.py.
# ============================================================================

DEVICE = "cpu"                       # "cpu" or "cuda"

DROWSINESS_CONFIDENCE = 0.70
DROWSINESS_DURATION = 2.0
DROWSINESS_COOLDOWN = 5.0

PHONE_CONFIDENCE = 0.70
PHONE_DURATION = 2.0
PHONE_COOLDOWN = 5.0

# Same combination rule used by the supplied POC.
# "OR" catches more cases; "AND" is more conservative.
DROWSINESS_COMBINE_MODE = "OR"

# MediaPipe EAR/MAR/PERCLOS
EAR_CLOSED_THRESHOLD_FLOOR = 0.15
EAR_CLOSED_DURATION = 1.5
EAR_COOLDOWN = 5.0
EAR_BASELINE_WINDOW_SAMPLES = 450
EAR_BASELINE_MIN_SAMPLES = 60
EAR_BASELINE_PERCENTILE = 90
EAR_CLOSED_RATIO = 0.72

MAR_YAWN_THRESHOLD = 0.55
MAR_YAWN_DURATION = 1.5

PERCLOS_WINDOW_SECONDS = 60.0
PERCLOS_THRESHOLD = 0.15
PERCLOS_MIN_WINDOW_COVERAGE = 0.5
PERCLOS_COOLDOWN = 10.0

# Phone geometry
PHONE_HAND_EAR_DISTANCE_RATIO = 0.6
PHONE_HAND_MOUTH_DISTANCE_RATIO = 0.7
PHONE_ELBOW_BEND_MAX_DEG = 140.0
PHONE_YOLO_CONF = 0.35

# Evidence frame saving:
# 0 = save only START and END evidence.
# 1.0 = save START/END plus at most one active-violation frame per second.
EVIDENCE_FRAME_INTERVAL_SECONDS = 1.0

# Process every frame by default.
FRAME_SKIP = 0

# Show live OpenCV window while processing.
DISPLAY = False


# ============================================================================
# Use the exact detector logic from the supplied codebase.
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from src.drowsiness import DrowsinessDetector
    from src.drowsiness_geometric import GeometricDrowsinessDetector
    from src.phone_detection import PhoneDetector
except Exception as exc:
    raise RuntimeError(
        "Could not import the supplied POC detectors. "
        "Place this file in the truck_safety_poc project root so that "
        "the existing 'src/' directory is beside this file."
    ) from exc


# ============================================================================
# Helpers
# ============================================================================

def resolve_path(path_value, base_dir=SCRIPT_DIR):
    if path_value is None:
        return None
    p = Path(path_value)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def timestamp(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def find_first_model(folder):
    folder = Path(folder)
    if not folder.exists():
        return None
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".pt", ".pth", ".onnx", ".bin"}:
            return p.resolve()
    return None


class TemporalViolation:
    """
    Same temporal idea as the supplied POC's TemporalFlag:
      condition -> sustained duration -> START -> ACTIVE -> END -> cooldown.

    It also records the peak confidence observed during the active event.
    """

    def __init__(self, name, duration_required, cooldown):
        self.name = name
        self.duration_required = float(duration_required)
        self.cooldown = float(cooldown)

        self.condition_since = None
        self.active = False
        self.last_end_time = None
        self.last_confidence = 0.0
        self.peak_confidence = 0.0

    def update(self, condition, video_time, confidence):
        confidence = float(confidence or 0.0)
        self.last_confidence = confidence

        if condition:
            self.peak_confidence = max(self.peak_confidence, confidence)

            if self.condition_since is None:
                self.condition_since = video_time

            sustained = video_time - self.condition_since
            cooldown_ok = (
                self.last_end_time is None
                or video_time - self.last_end_time >= self.cooldown
            )

            if not self.active and sustained >= self.duration_required and cooldown_ok:
                self.active = True
                return "START"

            return None

        self.condition_since = None

        if self.active:
            self.active = False
            self.last_end_time = video_time
            return "END"

        return None

    def reset_peak(self):
        self.peak_confidence = 0.0


def write_log(log_file, text):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def save_evidence(frame, frame_number, video_time, violation, suffix, output_dir):
    filename = (
        f"violation_{violation.lower()}_{suffix}_"
        f"frame_{frame_number:08d}_"
        f"time_{video_time:.3f}s.jpg"
    )
    path = output_dir / filename
    ok = cv2.imwrite(str(path), frame)
    return path if ok else None


def draw_overlay(frame, video_time, drowsy, phone, drowsy_geo):
    out = frame.copy()

    d_active = bool(drowsy.get("active", False))
    p_active = bool(phone.get("sustained_active", False))

    lines = [
        f"TIME: {timestamp(video_time)}",
        f"DROWSINESS: {'VIOLATION' if d_active else 'OK'}",
        f"  classifier: {drowsy.get('label', 'N/A')} "
        f"{float(drowsy.get('confidence', 0.0)) * 100:.1f}%",
        f"  EAR: {drowsy_geo.get('ear', 'N/A')} "
        f"closed={drowsy_geo.get('eyes_closed', False)}",
        f"  PERCLOS: {drowsy_geo.get('perclos', 'N/A')}",
        f"PHONE: {'VIOLATION' if p_active else 'OK'}",
        f"  mode: {phone.get('mode', 'N/A')}",
        f"  confidence: {float(phone.get('confidence', 0.0)) * 100:.1f}%",
    ]

    panel_h = 30 + 25 * len(lines)
    cv2.rectangle(out, (0, 0), (510, panel_h), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (10, 28 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


# ============================================================================
# Main
# ============================================================================

def main():
    global INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH
    global DROWSINESS_MODEL_PATH, PHONE_MODEL_PATH
    global PHONE_YOLO_FALLBACK_MODEL_PATH
    global DEVICE, DISPLAY, FRAME_SKIP

    parser = argparse.ArgumentParser(
        description="Standalone drowsiness + mobile-in-hand detector"
    )

    parser.add_argument("--source", default=INPUT_VIDEO_PATH)
    parser.add_argument("--output", default=OUTPUT_VIDEO_PATH)
    parser.add_argument("--drowsiness-model", default=DROWSINESS_MODEL_PATH)
    parser.add_argument("--phone-model", default=PHONE_MODEL_PATH)
    parser.add_argument(
        "--phone-yolo",
        default=PHONE_YOLO_FALLBACK_MODEL_PATH,
        help="Local YOLOv8 model path. If missing, detector may use yolov8m.pt.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEVICE)
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--display", action="store_true", default=DISPLAY)
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override classifier confidence threshold for drowsiness and phone.",
    )

    args = parser.parse_args()

    source_path = resolve_path(args.source)
    output_path = resolve_path(args.output) if args.output else None
    drowsy_model_path = resolve_path(args.drowsiness_model)
    phone_model_path = resolve_path(args.phone_model)
    phone_yolo_path = resolve_path(args.phone_yolo)

    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    if args.confidence is not None:
        # Match the POC's --confidence behavior for these two classifiers.
        import config
        config.DROWSINESS_CONFIDENCE = args.confidence
        config.PHONE_CONFIDENCE = args.confidence

    video_dir = source_path.parent
    evidence_dir = video_dir
    log_dir = resolve_path(LOG_DIR) if LOG_DIR else video_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "logs.txt"

    # Start a fresh log for this video.
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("DRIVER SAFETY VIOLATION LOG\n")
        f.write("=" * 78 + "\n")
        f.write(f"Input video : {source_path}\n")
        f.write(f"Output video: {output_path if output_path else 'DISABLED'}\n")
        f.write(f"Device      : {args.device}\n")
        f.write(f"Drowsy model: {drowsy_model_path}\n")
        f.write(f"Phone model : {phone_model_path}\n")
        f.write(f"Phone YOLO  : {phone_yolo_path}\n")
        f.write(
            f"Thresholds  : drowsy={DROWSINESS_CONFIDENCE:.2f}, "
            f"phone={PHONE_CONFIDENCE:.2f}\n"
        )
        f.write(
            f"Temporal    : drowsy={DROWSINESS_DURATION:.2f}s, "
            f"phone={PHONE_DURATION:.2f}s\n"
        )
        f.write("=" * 78 + "\n\n")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.resolve() == source_path.resolve():
            raise ValueError("Output video must not overwrite the input video.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video: {output_path}")

    # ----------------------------------------------------------------------
    # Load the EXACT supplied POC detectors.
    # ----------------------------------------------------------------------

    drowsiness_classifier = None

    if drowsy_model_path and drowsy_model_path.exists():
        try:
            drowsiness_classifier = DrowsinessDetector(
                str(drowsy_model_path),
                device=args.device,
            )
            print(f"[OK] Drowsiness classifier: {drowsy_model_path}")
        except Exception as exc:
            print(f"[WARNING] Drowsiness classifier disabled: {exc}")
            write_log(log_file, f"[WARNING] Drowsiness classifier disabled: {exc}")
    else:
        print("[WARNING] Drowsiness model not found. Using geometric detector only.")
        write_log(
            log_file,
            "[WARNING] Drowsiness model not found. "
            "Using geometric detector only.",
        )

    geometric_drowsiness = GeometricDrowsinessDetector()

    if geometric_drowsiness.available:
        print("[OK] MediaPipe EAR/MAR/PERCLOS enabled")
    else:
        print(
            "[WARNING] MediaPipe EAR/MAR/PERCLOS unavailable: "
            f"{geometric_drowsiness.unavailable_reason}"
        )
        write_log(
            log_file,
            "[WARNING] MediaPipe EAR/MAR/PERCLOS unavailable: "
            f"{geometric_drowsiness.unavailable_reason}",
        )

    # PhoneDetector itself implements:
    # classifier -> YOLOv8 direct -> MediaPipe posture fallback.
    phone_detector = PhoneDetector(
        str(phone_model_path) if phone_model_path and phone_model_path.exists() else None,
        device=args.device,
    )

    # The supplied phone detector reads the YOLO fallback location from config.
    # Override that config value with the path specified at the top/CLI.
    import config
    if phone_yolo_path:
        config.PHONE_YOLO_FALLBACK_MODEL = str(phone_yolo_path)
    config.PHONE_YOLO_CONF = PHONE_YOLO_CONF
    config.PHONE_HAND_EAR_DISTANCE_RATIO = PHONE_HAND_EAR_DISTANCE_RATIO
    config.PHONE_HAND_MOUTH_DISTANCE_RATIO = PHONE_HAND_MOUTH_DISTANCE_RATIO
    config.PHONE_ELBOW_BEND_MAX_DEG = PHONE_ELBOW_BEND_MAX_DEG

    if phone_model_path and phone_model_path.exists():
        print(f"[OK] Phone classifier: {phone_model_path}")
    else:
        print("[INFO] Phone classifier not found; using YOLO + posture fallback.")

    if phone_detector.pose_gate.available:
        print("[OK] MediaPipe phone posture fallback enabled")
    else:
        print(
            "[WARNING] MediaPipe phone posture fallback unavailable: "
            f"{phone_detector.pose_gate.unavailable_reason}"
        )

    # ----------------------------------------------------------------------
    # Temporal state for the two final violations.
    # ----------------------------------------------------------------------

    # IMPORTANT:
    # The supplied DrowsinessDetector and PhoneDetector already contain the
    # POC's TemporalFlag duration/cooldown logic. Therefore these final
    # aggregators do NOT add another duration on top (that would accidentally
    # turn 2s into ~4s). They only combine/track the already-temporal results
    # and handle evidence/logging.
    drowsiness_temporal = TemporalViolation(
        "DROWSINESS",
        0.0,
        DROWSINESS_COOLDOWN,
    )

    phone_temporal = TemporalViolation(
        "PHONE_USAGE",
        0.0,
        PHONE_COOLDOWN,
    )

    frame_number = 0
    processed_frames = 0

    active_start_time = {
        "DROWSINESS": None,
        "PHONE_USAGE": None,
    }

    last_evidence_time = {
        "DROWSINESS": None,
        "PHONE_USAGE": None,
    }

    event_counts = {
        "DROWSINESS": 0,
        "PHONE_USAGE": 0,
    }

    wall_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1

            if args.frame_skip > 0:
                if frame_number % (args.frame_skip + 1) != 0:
                    if writer:
                        writer.write(frame)
                    continue

            processed_frames += 1
            video_time = (frame_number - 1) / fps

            # ==============================================================
            # DROWSINESS
            # ==============================================================

            if drowsiness_classifier:
                d = drowsiness_classifier.process_frame(frame, video_time)
            else:
                d = {
                    "label": "DISABLED",
                    "confidence": 0.0,
                    "sustained_active": False,
                    "event": None,
                }

            geo = geometric_drowsiness.process_frame(frame, video_time)

            classifier_active = bool(d.get("sustained_active", False))
            geometric_active = bool(geo.get("sustained_active", False))

            if DROWSINESS_COMBINE_MODE.upper() == "AND":
                drowsy_condition = classifier_active and geometric_active
            else:
                # Exact default behavior of supplied POC.
                drowsy_condition = classifier_active or geometric_active

            # Confidence for temporal/evidence reporting:
            # use the strongest available drowsiness signal.
            drowsy_conf = float(d.get("confidence", 0.0) or 0.0)

            if geo.get("ear") is not None and geo.get("eyes_closed"):
                # EAR itself is not a probability, so do not pretend it is.
                # Keep classifier confidence as the actual confidence field.
                pass

            d_event = drowsiness_temporal.update(
                drowsy_condition,
                video_time,
                drowsy_conf,
            )

            # ==============================================================
            # PHONE
            # ==============================================================

            p = phone_detector.process_frame(frame, video_time)

            phone_condition = bool(p.get("sustained_active", False))

            phone_conf = float(p.get("confidence", 0.0) or 0.0)

            p_event = phone_temporal.update(
                phone_condition,
                video_time,
                phone_conf,
            )

            # ==============================================================
            # Log and save evidence
            # ==============================================================

            events_this_frame = []

            for violation, event, temporal, confidence in [
                ("DROWSINESS", d_event, drowsiness_temporal, drowsy_conf),
                ("PHONE_USAGE", p_event, phone_temporal, phone_conf),
            ]:
                if event == "START":
                    event_counts[violation] += 1
                    active_start_time[violation] = video_time
                    last_evidence_time[violation] = video_time

                    evidence = save_evidence(
                        frame,
                        frame_number,
                        video_time,
                        violation,
                        "START",
                        evidence_dir,
                    )

                    events_this_frame.append(
                        f"START {violation}: frame={frame_number}, "
                        f"time={timestamp(video_time)}, "
                        f"confidence={confidence:.3f}, "
                        f"peak_confidence={temporal.peak_confidence:.3f}, "
                        f"evidence={evidence}"
                    )

                    write_log(log_file, events_this_frame[-1])

                elif event == "END":
                    start_time = active_start_time[violation]
                    duration = (
                        video_time - start_time
                        if start_time is not None
                        else 0.0
                    )

                    evidence = save_evidence(
                        frame,
                        frame_number,
                        video_time,
                        violation,
                        "END",
                        evidence_dir,
                    )

                    write_log(
                        log_file,
                        f"END   {violation}: frame={frame_number}, "
                        f"time={timestamp(video_time)}, "
                        f"duration={duration:.3f}s, "
                        f"last_confidence={confidence:.3f}, "
                        f"peak_confidence={temporal.peak_confidence:.3f}, "
                        f"evidence={evidence}",
                    )

                    active_start_time[violation] = None
                    last_evidence_time[violation] = None
                    temporal.reset_peak()

            # Save periodic evidence while the violation remains active.
            for violation, active, confidence in [
                (
                    "DROWSINESS",
                    drowsiness_temporal.active,
                    drowsy_conf,
                ),
                (
                    "PHONE_USAGE",
                    phone_temporal.active,
                    phone_conf,
                ),
            ]:
                if not active or EVIDENCE_FRAME_INTERVAL_SECONDS <= 0:
                    continue

                last = last_evidence_time[violation]
                if last is None or (
                    video_time - last >= EVIDENCE_FRAME_INTERVAL_SECONDS
                ):
                    evidence = save_evidence(
                        frame,
                        frame_number,
                        video_time,
                        violation,
                        "ACTIVE",
                        evidence_dir,
                    )

                    write_log(
                        log_file,
                        f"ACTIVE {violation}: frame={frame_number}, "
                        f"time={timestamp(video_time)}, "
                        f"confidence={confidence:.3f}, "
                        f"evidence={evidence}",
                    )

                    last_evidence_time[violation] = video_time

            # --------------------------------------------------------------
            # Annotated video
            # --------------------------------------------------------------

            if writer:
                annotated = draw_overlay(
                    frame,
                    video_time,
                    {
                        "active": drowsiness_temporal.active,
                        "label": d.get("label"),
                        "confidence": d.get("confidence", 0.0),
                    },
                    p,
                    geo,
                )
                writer.write(annotated)

            if args.display:
                display_frame = draw_overlay(
                    frame,
                    video_time,
                    {
                        "active": drowsiness_temporal.active,
                        "label": d.get("label"),
                        "confidence": d.get("confidence", 0.0),
                    },
                    p,
                    geo,
                )
                cv2.imshow("Drowsiness + Mobile Detection", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

            if frame_number % max(int(fps), 1) == 0:
                print(
                    f"frame={frame_number}/{total_frames} "
                    f"time={timestamp(video_time)} "
                    f"drowsy={'YES' if drowsiness_temporal.active else 'NO'} "
                    f"phone={'YES' if phone_temporal.active else 'NO'}"
                )

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.time() - wall_start

    # If the video ends during an active violation, close it explicitly in
    # logs rather than silently losing the occurrence.
    final_video_time = max(0.0, (frame_number - 1) / fps)

    for violation, temporal in [
        ("DROWSINESS", drowsiness_temporal),
        ("PHONE_USAGE", phone_temporal),
    ]:
        if temporal.active:
            start_time = active_start_time[violation]
            duration = (
                final_video_time - start_time
                if start_time is not None
                else 0.0
            )

            write_log(
                log_file,
                f"END   {violation}: video_end_frame={frame_number}, "
                f"time={timestamp(final_video_time)}, "
                f"duration={duration:.3f}s, "
                f"peak_confidence={temporal.peak_confidence:.3f}, "
                f"reason=VIDEO_END",
            )

    write_log(log_file, "")
    write_log(log_file, "=" * 78)
    write_log(log_file, "SUMMARY")
    write_log(log_file, "=" * 78)
    write_log(log_file, f"Frames read       : {frame_number}")
    write_log(log_file, f"Frames processed  : {processed_frames}")
    write_log(log_file, f"Video FPS         : {fps:.3f}")
    write_log(
        log_file,
        f"Video duration    : {frame_number / fps:.3f}s",
    )
    write_log(log_file, f"Processing time    : {elapsed:.3f}s")
    write_log(
        log_file,
        f"Processing speed  : "
        f"{processed_frames / elapsed if elapsed > 0 else 0.0:.2f} FPS",
    )
    write_log(log_file, f"Drowsiness events  : {event_counts['DROWSINESS']}")
    write_log(log_file, f"Phone events       : {event_counts['PHONE_USAGE']}")
    write_log(log_file, "")
    write_log(log_file, "Evidence JPG files are stored beside the input video.")
    write_log(log_file, f"Log file: {log_file}")
    if output_path:
        write_log(log_file, f"Annotated video: {output_path}")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Drowsiness events : {event_counts['DROWSINESS']}")
    print(f"Phone events      : {event_counts['PHONE_USAGE']}")
    print(f"logs.txt          : {log_file}")
    print(f"Evidence frames   : {evidence_dir}")
    if output_path:
        print(f"Output video      : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
