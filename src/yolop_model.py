"""
src/yolop_model.py
===================
Loads the REAL YOLOP architecture (vendored from the official hustvl/YOLOP
repo under third_party/yolop/, MIT licensed) and your local weight file's
`state_dict`, then runs inference exactly the way the official
`tools/demo.py` does — same letterbox preprocessing, same normalization
(including the repo's own BGR-not-converted quirk, preserved for fidelity
since the model was trained on exactly that pipeline), same NMS, same
segmentation-mask unletterboxing.

This is NOT a generic path — it exists specifically because a raw
state_dict / custom checkpoint dict cannot be loaded without the original
model class (see src/model_loader.py's _inspect_bin). If your checkpoint
ever changes shape (different key names, different param count), this
module will fail loudly with the real PyTorch error rather than silently
producing wrong output.
"""

import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

_THIRD_PARTY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "yolop"
)
if _THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_DIR)

from lib.config import cfg as _yolop_cfg          # noqa: E402
from lib.models import get_net                     # noqa: E402
from lib.core.general import non_max_suppression, scale_coords  # noqa: E402
from lib.utils import letterbox_for_img             # noqa: E402

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_TRANSFORM = transforms.Compose([transforms.ToTensor(), _NORMALIZE])

# Official YOLOP names for the detection head (BDD100K vehicle class only)
DEFAULT_DET_NAMES = ["vehicle"]


class YOLOPModel:
    def __init__(self, weight_path: str, device: str = "cpu",
                 img_size: int = 640, conf_thres: float = 0.25, iou_thres: float = 0.45):
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model = get_net(_yolop_cfg)
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)

        if not (isinstance(checkpoint, dict) and "state_dict" in checkpoint):
            raise RuntimeError(
                f"Expected a checkpoint dict with a 'state_dict' key (the official "
                f"YOLOP format), but got keys: "
                f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
            )

        # Will raise a clear PyTorch error (missing/unexpected keys, shape
        # mismatch) if this weight file doesn't actually match the vendored
        # architecture — we do not catch/hide that, per "do not fake results".
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.names = DEFAULT_DET_NAMES

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns:
            {
              "objects": [{"label": str, "confidence": float, "box": [x1,y1,x2,y2]}, ...],
              "drivable_area_mask": np.ndarray (H,W) uint8, 1 = drivable, same size as input frame,
              "lane_mask": np.ndarray (H,W) uint8, 1 = lane line, same size as input frame,
              "inference_time_sec": float,
            }
        """
        img0 = frame_bgr
        h0, w0 = img0.shape[:2]

        img, ratio, pad = letterbox_for_img(img0, new_shape=self.img_size, auto=True)
        img = np.ascontiguousarray(img)

        tensor = _TRANSFORM(img).to(self.device).float().unsqueeze(0)

        t0 = time.time()
        det_out, da_seg_out, ll_seg_out = self.model(tensor)
        inf_time = time.time() - t0

        inf_out, _ = det_out
        det_pred = non_max_suppression(
            inf_out, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=None, agnostic=False
        )
        det = det_pred[0]

        objects = []
        if det is not None and len(det):
            det = det.clone()
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det.tolist():
                cls_id = int(cls)
                label = self.names[cls_id] if cls_id < len(self.names) else str(cls_id)
                objects.append({"label": label, "confidence": float(conf), "box": [float(v) for v in xyxy]})

        _, _, height, width = tensor.shape
        pad_w, pad_h = pad
        pad_w, pad_h = int(pad_w), int(pad_h)

        # Guard against zero-size crops on extreme aspect ratios
        crop_h_end = max(height - pad_h, pad_h + 1)
        crop_w_end = max(width - pad_w, pad_w + 1)

        da_predict = da_seg_out[:, :, pad_h:crop_h_end, pad_w:crop_w_end]
        da_mask = torch.nn.functional.interpolate(da_predict, size=(h0, w0), mode="bilinear", align_corners=False)
        _, da_mask = torch.max(da_mask, 1)
        da_mask = da_mask.int().squeeze().cpu().numpy().astype(np.uint8)

        ll_predict = ll_seg_out[:, :, pad_h:crop_h_end, pad_w:crop_w_end]
        ll_mask = torch.nn.functional.interpolate(ll_predict, size=(h0, w0), mode="bilinear", align_corners=False)
        _, ll_mask = torch.max(ll_mask, 1)
        ll_mask = ll_mask.int().squeeze().cpu().numpy().astype(np.uint8)

        return {
            "objects": objects,
            "drivable_area_mask": da_mask,
            "lane_mask": ll_mask,
            "inference_time_sec": inf_time,
        }