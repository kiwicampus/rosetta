"""nn.Module wrapper for ONNX models. Use when onnx2torch fails (e.g. Loop op)."""
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision.io import read_image


class OnnxRuntimeWrapper(nn.Module):
    def __init__(self, onnx_path: str):
        super().__init__()
        self.onnx_path = onnx_path
        self._session = None
        self._session_cpu = None

    @property
    def session(self):
        if self._session is None:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # Errors only (suppress Memcpy warnings)
            self._session = ort.InferenceSession(
                self.onnx_path,
                opts,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            provider = self._session.get_providers()[0]
            device = "GPU" if "CUDA" in provider else "CPU"
            print(f"\nONNX Runtime: using {device} ({provider})\n")
        return self._session

    @property
    def session_cpu(self):
        if self._session_cpu is None:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._session_cpu = ort.InferenceSession(
                self.onnx_path, opts, providers=['CPUExecutionProvider']
            )
        return self._session_cpu

    def forward(self, *args, **kwargs):
        sess = self.session
        input_names = [inp.name for inp in sess.get_inputs()]
        feed = {}
        for i, name in enumerate(input_names):
            if i < len(args):
                feed[name] = args[i]
            elif name in kwargs:
                feed[name] = kwargs[name]
            else:
                raise ValueError(f"Missing input '{name}'")
        for k, v in feed.items():
            if torch.is_tensor(v):
                feed[k] = v.detach().cpu().numpy()
        try:
            outputs = sess.run(None, feed)
        except Exception as e:
            if "allocate" in str(e).lower():
                outputs = self.session_cpu.run(None, feed)
            else:
                raise
        out = [torch.from_numpy(o) for o in outputs]
        return out[0] if len(out) == 1 else tuple(out)


def _load_image_rgb(image_path: str) -> "np.ndarray":
    """Load image as [H,W,3] RGB uint8 numpy."""
    img = read_image(image_path)  # [C,H,W] uint8
    return img.permute(1, 2, 0).numpy()


_MAX_INFER_DIM = 800  # Cap to avoid NMS OOM; 1200px images bypassed 1280, causing fragmentation on re-runs


def _run_tf_od(
    model: nn.Module, img: "np.ndarray", score_thresh: float, device: torch.device
) -> dict:
    """Run TF OD model on [H,W,3] RGB uint8. Returns {bboxes, labels}."""
    h, w = img.shape[0], img.shape[1]
    # Resize large images to avoid NonMaxSuppression OOM (scales with spatial size)
    scale = min(1.0, _MAX_INFER_DIM / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)  # [1,H,W,3]
    if hasattr(model, "session"):
        # OnnxRuntimeWrapper converts to numpy; ORT handles device
        inp_name = model.session.get_inputs()[0].name
        with torch.inference_mode():
            num_det, scores, boxes = model(**{inp_name: x})
    else:
        model = model.to(device)
        x = x.to(device)
        with torch.inference_mode():
            num_det, scores, boxes = model(x)
    num_det = int(num_det[0].item())
    scores = scores[0][:num_det]
    boxes = boxes[0][:num_det]  # ymin,xmin,ymax,xmax norm [0,1]
    keep = scores >= score_thresh
    boxes = boxes[keep]
    # Denorm to original image coords (h,w from outer scope; norm boxes map to any size)
    boxes_xyxy = torch.stack([
        boxes[:, 1] * w, boxes[:, 0] * h,
        boxes[:, 3] * w, boxes[:, 2] * h
    ], dim=1)
    return {"bboxes": boxes_xyxy, "labels": torch.zeros(len(boxes_xyxy), dtype=torch.long)}


def _resolve_device(device: str | None) -> torch.device:
    """Resolve 'auto'|'cuda'|'cpu' to torch.device."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TfOdPredictor(nn.Module):
    """Base for TF OD format models (plate, face, etc)."""

    def __init__(
        self,
        model_path: str,
        classes: list[str],
        score_thresh: float = 0.5,
        device: str | None = "auto",
    ):
        super().__init__()
        self.device = _resolve_device(device)
        self._model = torch.load(
            model_path, weights_only=False, map_location=self.device
        )
        self.score_thresh = score_thresh
        self.classes = classes

    def predict(self, image_path: str):
        return self.predict_from_array(_load_image_rgb(image_path))

    def predict_from_array(self, img: "np.ndarray"):
        """img: [H,W,3] RGB uint8 numpy. Avoids reload when blurring in place."""
        return _run_tf_od(self._model, img, self.score_thresh, self.device)


class PlatePredictor(TfOdPredictor):
    """Wraps plate ONNX (TF OD format) with predict() and .classes for anon.py."""

    def __init__(
        self,
        model_path: str,
        score_thresh: float = 0.5,
        device: str | None = "auto",
    ):
        super().__init__(model_path, ["plate"], score_thresh, device)


class FacePredictor(TfOdPredictor):
    """Wraps face ONNX (TF OD format) with predict() and .classes for anon.py."""

    def __init__(
        self,
        model_path: str,
        score_thresh: float = 0.5,
        device: str | None = "auto",
    ):
        super().__init__(model_path, ["face"], score_thresh, device)
