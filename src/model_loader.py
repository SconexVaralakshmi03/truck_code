"""
src/model_loader.py
====================
Generic, cautious model loader/inspector.

The whole point of this module is to NOT assume every ".pt" file is an
Ultralytics YOLO model. Instead, for each weight file we:

  1. Look at the file extension.
  2. If it's a torch-ish file (.pt/.pth), try `torch.load` with
     `weights_only=False` in a controlled way and inspect what comes back:
       - An Ultralytics-style checkpoint (dict with 'model' key whose model
         has a YOLO-like structure, or a file that also has an adjacent
         .yaml, or that ultralytics.YOLO() can open) -> treat as Ultralytics.
       - A raw state_dict (dict of tensors) -> generic PyTorch checkpoint;
         we cannot know the architecture without the user's model class, so
         we report this clearly instead of guessing.
       - A pickled nn.Module -> generic PyTorch model, usable directly with
         model.eval() / forward().
  3. If it's a `.onnx` file -> ONNX Runtime.
  4. If it's a `.bin` file (e.g. YOLOP-style `End-to-end.bin`) -> this is
     almost always NOT a standalone-loadable format; it typically requires
     the original model-definition code (e.g. the YOLOP repo's `lib/models`)
     to build the architecture and then load the state dict into it. We
     detect this and report it rather than silently failing or faking it.

Every inspection function returns an `InspectionResult` describing exactly
what was found, what inference API applies, and whether loading actually
succeeded — never a fabricated success.
"""

import os
import json
import traceback
from dataclasses import dataclass, field
from typing import Optional, Any, List


@dataclass
class InspectionResult:
    file: str
    exists: bool = False
    framework: str = "UNKNOWN"
    architecture: str = "UNKNOWN"
    task: str = "UNKNOWN"
    input_size: str = "UNKNOWN"
    classes: Optional[List[str]] = None
    num_classes: Optional[int] = None
    load_status: str = "NOT ATTEMPTED"
    load_error: Optional[str] = None
    inference_api: str = "UNKNOWN"
    can_cpu: Optional[bool] = None
    can_cuda: Optional[bool] = None
    notes: List[str] = field(default_factory=list)
    # the actual loaded object, if successful (not serialized/printed)
    handle: Any = None


def _try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


def inspect_and_load(file_path: str, prefer_device: str = "cpu") -> InspectionResult:
    """
    Inspect a single model weight file and attempt to load it with the
    correct API. Never assumes; reports exactly what it finds.
    """
    result = InspectionResult(file=file_path)

    if not os.path.exists(file_path):
        result.load_status = "FAIL"
        result.load_error = f"File not found: {file_path}"
        return result

    result.exists = True
    ext = os.path.splitext(file_path)[1].lower()

    torch = _try_import("torch")

    if not torch and ext in (".pt", ".pth"):
        result.framework = "PyTorch"
        result.load_status = "FAIL"
        result.load_error = "PyTorch is not installed. Run check_environment.py and install requirements.txt."
        return result

    # ------------------------------------------------------------------
    # .pt / .pth — could be Ultralytics YOLO OR a raw PyTorch checkpoint
    # ------------------------------------------------------------------
    if ext in (".pt", ".pth"):
        return _inspect_torch_checkpoint(file_path, torch, prefer_device, result)

    # ------------------------------------------------------------------
    # .onnx
    # ------------------------------------------------------------------
    if ext == ".onnx":
        return _inspect_onnx(file_path, result, prefer_device)

    # ------------------------------------------------------------------
    # .bin — e.g. YOLOP "End-to-end.bin" style checkpoints. These are
    # usually a raw state_dict saved with a non-standard extension and
    # REQUIRE the original architecture code to instantiate the model.
    # ------------------------------------------------------------------
    if ext == ".bin":
        return _inspect_bin(file_path, torch, result)

    result.framework = "UNKNOWN"
    result.load_status = "FAIL"
    result.load_error = f"Unrecognized extension '{ext}'. Inspect manually."
    return result


def _inspect_torch_checkpoint(file_path, torch, prefer_device, result: InspectionResult) -> InspectionResult:
    # Step A: peek at the raw pickle content WITHOUT assuming Ultralytics.
    raw = None
    try:
        raw = torch.load(file_path, map_location="cpu", weights_only=False)
        result.notes.append("Loaded raw checkpoint object with torch.load().")
    except Exception as e:
        # Some Ultralytics checkpoints only load cleanly through the
        # ultralytics.YOLO() wrapper (custom pickled classes). Fall through
        # to attempt that before giving up.
        result.notes.append(f"torch.load() raw peek failed: {e}")

    is_dict = isinstance(raw, dict)
    looks_like_ultralytics = False
    if is_dict:
        keys = set(raw.keys())
        # Ultralytics train checkpoints typically contain these keys.
        if {"model"}.issubset(keys) and (
            "train_args" in keys or "ema" in keys or "epoch" in keys or "date" in keys
        ):
            looks_like_ultralytics = True
        result.notes.append(f"Top-level checkpoint keys: {sorted(keys)}")

    if looks_like_ultralytics:
        return _load_as_ultralytics(file_path, prefer_device, result)

    if is_dict and not looks_like_ultralytics:
        # Could still be a bare state_dict (all values are tensors), or a
        # custom checkpoint dict with e.g. {'state_dict':..., 'classes':...}
        all_tensors = all(hasattr(v, "shape") for v in raw.values()) if raw else False
        if all_tensors:
            result.framework = "PyTorch"
            result.architecture = "Raw state_dict (architecture unknown)"
            result.task = "UNKNOWN — cannot determine without model class definition"
            result.load_status = "PARTIAL"
            result.load_error = (
                "This file is a raw state_dict, not a full model. The original "
                "model class (Python code that defines the architecture) is "
                "required to instantiate it before these weights can be loaded. "
                "Provide the model definition (e.g. a models.py from the training "
                "repo) or export the model with the architecture bundled "
                "(e.g. torch.save(model) instead of torch.save(model.state_dict()))."
            )
            result.inference_api = "N/A until architecture is supplied"
            result.handle = raw
            return result
        else:
            # Custom dict — try to detect useful metadata (classes, names, etc.)
            result.framework = "PyTorch"
            result.architecture = "Custom checkpoint dict"
            possible_class_keys = [k for k in raw.keys() if "class" in k.lower() or "name" in k.lower()]
            if possible_class_keys:
                result.notes.append(f"Possible class-name fields: {possible_class_keys}")
                for k in possible_class_keys:
                    val = raw.get(k)
                    if isinstance(val, (list, dict)):
                        result.classes = list(val.values()) if isinstance(val, dict) else list(val)
                        result.num_classes = len(result.classes)
            result.load_status = "PARTIAL"
            result.load_error = (
                "Custom checkpoint dict detected. Could not confirm architecture "
                "automatically — inspect result.notes for the raw keys and "
                "cross-reference with any README/config from the training repo."
            )
            result.handle = raw
            return result

    # Not a dict at all -> likely a pickled nn.Module saved directly
    # (torch.save(model)). Usable directly.
    nn = _try_import("torch")
    try:
        if raw is not None and hasattr(raw, "eval") and hasattr(raw, "forward"):
            result.framework = "PyTorch"
            result.architecture = type(raw).__name__
            result.task = "UNKNOWN — inspect model.__class__ and output shape to confirm"
            result.load_status = "PASS"
            result.inference_api = "model.eval(); model(input_tensor)"
            result.can_cpu = True
            try:
                result.can_cuda = torch.cuda.is_available()
            except Exception:
                result.can_cuda = False
            result.handle = raw
            return result
    except Exception as e:
        result.notes.append(f"Post-load inspection error: {e}")

    # Last resort: try the Ultralytics wrapper anyway, since some exports
    # only open correctly through it even without the tell-tale dict keys.
    return _load_as_ultralytics(file_path, prefer_device, result, is_fallback=True)


def _load_as_ultralytics(file_path, prefer_device, result: InspectionResult, is_fallback=False) -> InspectionResult:
    ultra = _try_import("ultralytics")
    if not ultra:
        result.load_status = "FAIL"
        result.load_error = (
            "File appears to be an Ultralytics YOLO checkpoint but the "
            "'ultralytics' package is not installed. Run: pip install ultralytics"
        )
        return result
    try:
        from ultralytics import YOLO
        model = YOLO(file_path)
        result.framework = "Ultralytics"
        # model.task is one of: detect, classify, segment, pose, obb
        result.task = getattr(model, "task", "UNKNOWN")
        result.architecture = type(model.model).__name__ if hasattr(model, "model") else "YOLO"
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            result.classes = [names[k] for k in sorted(names.keys())]
        elif isinstance(names, list):
            result.classes = names
        result.num_classes = len(result.classes) if result.classes else None
        result.input_size = "Typically 640x640 (Ultralytics default; confirm via model.overrides.get('imgsz'))"
        result.load_status = "PASS"
        result.inference_api = "ultralytics.YOLO(path); model.predict(frame, conf=..., device=...)"
        result.can_cpu = True
        torch = _try_import("torch")
        result.can_cuda = bool(torch and torch.cuda.is_available())
        result.handle = model
        if is_fallback:
            result.notes.append("Loaded via Ultralytics fallback path (raw pickle peek was inconclusive).")
        return result
    except Exception as e:
        result.load_status = "FAIL"
        result.load_error = f"Ultralytics YOLO() load failed: {e}\n{traceback.format_exc(limit=2)}"
        return result


def _inspect_onnx(file_path, result: InspectionResult, prefer_device) -> InspectionResult:
    ort = _try_import("onnxruntime")
    if not ort:
        result.framework = "ONNX"
        result.load_status = "FAIL"
        result.load_error = "onnxruntime is not installed. Run: pip install onnxruntime (or onnxruntime-gpu)."
        return result
    try:
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        result.can_cpu = "CPUExecutionProvider" in available
        result.can_cuda = "CUDAExecutionProvider" in available
        if prefer_device == "cuda" and result.can_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(file_path, providers=providers)
        result.framework = "ONNX Runtime"
        result.architecture = "ONNX graph (see inputs/outputs below)"
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        result.input_size = str([i.shape for i in inputs])
        result.notes.append(f"Input names/shapes: {[(i.name, i.shape) for i in inputs]}")
        result.notes.append(f"Output names/shapes: {[(o.name, o.shape) for o in outputs]}")
        result.task = "UNKNOWN — infer from output shape count/dims (see notes)"
        result.load_status = "PASS"
        result.inference_api = "onnxruntime.InferenceSession; session.run(None, {input_name: array})"
        result.handle = session
        return result
    except Exception as e:
        result.framework = "ONNX Runtime"
        result.load_status = "FAIL"
        result.load_error = f"ONNX Runtime failed to open file: {e}"
        return result


def _inspect_bin(file_path, torch, result: InspectionResult) -> InspectionResult:
    result.framework = "UNKNOWN (.bin)"
    result.notes.append(
        ".bin is not a self-describing PyTorch/ONNX extension. Common cases: "
        "(a) YOLOP-style repos that ship 'End-to-end.pth'/'.bin' checkpoints as "
        "a dict with a 'state_dict' key, requiring the repo's own "
        "lib/models/YOLOP class to instantiate; (b) a raw state_dict with no "
        "wrapper dict, same requirement; (c) a fully pickled nn.Module "
        "(torch.save(model) style), which CAN be used directly; or "
        "(d) a HuggingFace-style state_dict requiring its config.json + "
        "modeling code."
    )

    if not torch:
        result.load_status = "FAIL"
        result.load_error = "PyTorch is not installed. Run check_environment.py and install requirements.txt."
        return result

    try:
        raw = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as e:
        result.load_status = "FAIL"
        result.load_error = f"torch.load() on .bin file failed: {e}"
        return result

    # Case 1: a fully pickled nn.Module (torch.save(model)) -- usable directly.
    if hasattr(raw, "eval") and hasattr(raw, "forward"):
        result.framework = "PyTorch"
        result.architecture = type(raw).__name__
        result.task = (
            "UNKNOWN — if this is YOLOP, expect a multi-task output: "
            "object detection + drivable-area segmentation + lane-line segmentation"
        )
        result.load_status = "PASS"
        result.inference_api = (
            "model.eval(); model(input_tensor) — NOTE: verify the exact input "
            "size/normalization and output unpacking (YOLOP returns a tuple of "
            "3 outputs) against the original repo before trusting results."
        )
        result.can_cpu = True
        result.can_cuda = bool(torch.cuda.is_available())
        result.handle = raw
        result.notes.append("Loaded as a fully pickled nn.Module — usable directly, no architecture code needed.")
        return result

    # Case 2: a dict of some shape.
    if isinstance(raw, dict):
        keys = list(raw.keys())
        result.notes.append(f"Top-level checkpoint keys: {keys}")
        all_tensors = all(hasattr(v, "shape") for v in raw.values()) if raw else False

        if all_tensors:
            result.architecture = "Raw state_dict (.bin, likely YOLOP or similar multi-task model)"
            result.task = "Likely multi-task: object detection + drivable-area + lane-line segmentation"
            result.load_status = "PARTIAL"
            result.load_error = (
                "This is a raw state_dict. It CANNOT be loaded standalone — it "
                "requires the original model class (e.g. the YOLOP repository's "
                "lib/models/YOLOP.py model definition) to build the network graph, "
                "after which this can be loaded via model.load_state_dict(raw). "
                "Clone/vendor the matching model-definition code, or request an "
                "ONNX/TorchScript export of the same weights instead."
            )
            result.notes.append(f"state_dict has {len(raw)} tensor entries.")
            result.handle = raw
            return result

        # Custom wrapper dict, e.g. {'state_dict': ..., 'epoch': ...} -- this is
        # the shape of the official YOLOP 'End-to-end.pth' checkpoint.
        # Prefer an exact 'state_dict' match over a fuzzy 'model' match --
        # some checkpoints (including the official YOLOP one) have BOTH keys,
        # where 'model' is just a config/name string, not weights.
        state_dict_key = None
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state_dict_key = "state_dict"
        else:
            for k in keys:
                if k.lower() == "model" and isinstance(raw[k], dict):
                    state_dict_key = k
                    break
            if state_dict_key is None:
                state_dict_key = next(
                    (k for k in keys if "state_dict" in k.lower()), None
                )
        result.architecture = "Custom checkpoint dict (.bin)"
        result.task = "Likely multi-task: object detection + drivable-area + lane-line segmentation (if YOLOP)"
        result.load_status = "PARTIAL"

        if state_dict_key:
            inner = raw[state_dict_key]
            n_tensors = len(inner) if isinstance(inner, dict) else "unknown"
            result.notes.append(
                f"Found '{state_dict_key}' key containing what looks like a state_dict "
                f"({n_tensors} entries)."
            )
            looks_like_yolop = (
                isinstance(inner, dict)
                and len(inner) == 542
                and all(k.startswith("model.") for k in list(inner.keys())[:5])
            )
            if looks_like_yolop:
                result.architecture = "Official YOLOP checkpoint (hustvl/YOLOP, 542 params)"
                result.load_error = (
                    "This matches the official YOLOP checkpoint format exactly "
                    "(542 params, 'model.N....' key naming). Use src/yolop_model.py "
                    "(vendored from https://github.com/hustvl/YOLOP, MIT licensed) "
                    "to load and run this — it is already wired into "
                    "src/road_monitor.py."
                )
                result.inference_api = (
                    "src.yolop_model.YOLOPModel(weight_path).process_frame(frame_bgr)"
                )
            else:
                result.load_error = (
                    f"This checkpoint stores its weights under raw['{state_dict_key}']. "
                    "It still requires the original model class to instantiate before "
                    f"model.load_state_dict(raw['{state_dict_key}']) can be called. If "
                    "this is the official YOLOP 'End-to-end.pth', clone "
                    "https://github.com/hustvl/YOLOP, import its model class, build the "
                    "model, then load this state_dict. Once loaded, wire the resulting "
                    "model object into RoadMonitor (src/road_monitor.py) as a custom "
                    "framework path — the generic Ultralytics/ONNX paths do not apply."
                )
        else:
            result.load_error = (
                "Custom checkpoint dict detected, no obvious 'state_dict' key found. "
                "Inspect result.notes for the raw keys above and cross-reference with "
                "your training repo to find where the weights live."
            )
        result.handle = raw
        return result

    # Anything else (bare tensor, numpy array, etc.) -- report exactly what we found.
    result.load_status = "FAIL"
    result.load_error = (
        f"torch.load() succeeded but returned an unrecognized object of type "
        f"{type(raw).__name__} (not a dict, not an nn.Module). Inspect it manually."
    )
    return result


def result_to_dict(r: InspectionResult) -> dict:
    d = {
        "file": r.file,
        "framework": r.framework,
        "architecture": r.architecture,
        "task": r.task,
        "input_size": r.input_size,
        "classes": r.classes,
        "num_classes": r.num_classes,
        "load_status": r.load_status,
        "load_error": r.load_error,
        "inference_api": r.inference_api,
        "can_cpu": r.can_cpu,
        "can_cuda": r.can_cuda,
        "notes": r.notes,
    }
    return d