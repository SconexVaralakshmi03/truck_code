"""
src/detector_base.py
=====================
A generic single-frame classifier/detector wrapper built on top of
src/model_loader.py's InspectionResult. It exposes one method,
`predict(frame_bgr) -> (label, confidence)`, that works across the
frameworks we might encounter (Ultralytics classify/detect, raw PyTorch
module, ONNX Runtime) WITHOUT assuming which one a given file is.

If a model could not be loaded (result.load_status == "FAIL"), predict()
raises RuntimeError instead of returning fabricated results — callers must
handle this and report it, per the "do not fake results" requirement.
"""

import numpy as np
import cv2

from src.model_loader import inspect_and_load


class GenericFrameModel:
    def __init__(self, model_path: str, device: str = "cpu", input_size: int = 224):
        self.model_path = model_path
        self.device = device
        self.input_size = input_size
        self.result = inspect_and_load(model_path, prefer_device=device)

        if self.result.load_status == "FAIL":
            raise RuntimeError(
                f"Cannot use model at '{model_path}': {self.result.load_error}"
            )

        self.framework = self.result.framework
        self.task = self.result.task
        self.classes = self.result.classes

    @property
    def usable(self):
        return self.result.load_status in ("PASS", "PARTIAL") and self.result.handle is not None

    def predict(self, frame_bgr):
        """
        Returns (label: str, confidence: float, extra: dict)
        `extra` may contain raw boxes/probs for downstream overlay use.
        """
        if self.result.load_status == "PARTIAL":
            raise RuntimeError(
                f"Model at '{self.model_path}' loaded only PARTIALLY "
                f"(architecture/classes could not be fully confirmed): "
                f"{self.result.load_error}"
            )

        if self.framework == "Ultralytics":
            return self._predict_ultralytics(frame_bgr)
        elif self.framework == "PyTorch":
            return self._predict_pytorch(frame_bgr)
        elif self.framework == "ONNX Runtime":
            return self._predict_onnx(frame_bgr)
        else:
            raise RuntimeError(f"No inference path implemented for framework '{self.framework}'")

    # ------------------------------------------------------------------
    def _predict_ultralytics(self, frame_bgr):
        model = self.result.handle
        results = model.predict(source=frame_bgr, device=self.device, verbose=False)
        r = results[0]

        if self.task == "classify" and r.probs is not None:
            top1 = int(r.probs.top1)
            conf = float(r.probs.top1conf)
            label = self.classes[top1] if self.classes and top1 < len(self.classes) else str(top1)
            return label, conf, {"probs": r.probs}

        if r.boxes is not None and len(r.boxes) > 0:
            confs = r.boxes.conf.tolist()
            clss = r.boxes.cls.tolist()
            best_idx = int(np.argmax(confs))
            cls_id = int(clss[best_idx])
            conf = float(confs[best_idx])
            label = self.classes[cls_id] if self.classes and cls_id < len(self.classes) else str(cls_id)
            return label, conf, {"boxes": r.boxes}

        # Nothing detected this frame
        return "none", 0.0, {"boxes": r.boxes}

    # ------------------------------------------------------------------
    def _predict_pytorch(self, frame_bgr):
        import torch
        model = self.result.handle
        model.eval()

        img = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        if self.device == "cuda" and torch.cuda.is_available():
            tensor = tensor.cuda()
            model = model.cuda()

        with torch.no_grad():
            out = model(tensor)

        if isinstance(out, (list, tuple)):
            out = out[0]

        probs = torch.softmax(out, dim=1) if out.dim() == 2 else out
        conf, idx = torch.max(probs, dim=1)
        idx = int(idx.item())
        conf = float(conf.item())
        label = self.classes[idx] if self.classes and idx < len(self.classes) else str(idx)
        return label, conf, {"raw_output": out}

    # ------------------------------------------------------------------
    def _predict_onnx(self, frame_bgr):
        session = self.result.handle
        input_meta = session.get_inputs()[0]
        shape = input_meta.shape
        h = self.input_size
        w = self.input_size
        img = cv2.resize(frame_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]

        outputs = session.run(None, {input_meta.name: img})
        out = outputs[0]
        if out.ndim == 2:
            exp = np.exp(out - np.max(out))
            probs = exp / np.sum(exp)
            idx = int(np.argmax(probs))
            conf = float(probs[0, idx]) if probs.ndim == 2 else float(probs[idx])
            label = self.classes[idx] if self.classes and idx < len(self.classes) else str(idx)
            return label, conf, {"raw_output": out}

        return "unknown", 0.0, {"raw_output": out}
