"""nn.Module wrapper for ONNX models. Use when onnx2torch fails (e.g. Loop op)."""
import torch
import torch.nn as nn


class OnnxRuntimeWrapper(nn.Module):
    def __init__(self, onnx_path: str):
        super().__init__()
        self.onnx_path = onnx_path
        self._session = None

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self.onnx_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            provider = self._session.get_providers()[0]
            device = "GPU" if "CUDA" in provider else "CPU"
            print(f"\nONNX Runtime: using {device} ({provider})\n")
        return self._session

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
        outputs = sess.run(None, feed)
        out = [torch.from_numpy(o) for o in outputs]
        return out[0] if len(out) == 1 else tuple(out)


def _load_image_rgb(image_path: str) -> "np.ndarray":
    """Load image as [H,W,3] RGB uint8 numpy."""
    from torchvision.io import read_image
    img = read_image(image_path)  # [C,H,W] uint8
    return img.permute(1, 2, 0).numpy()


def _run_tf_od(model: nn.Module, img: "np.ndarray", score_thresh: float) -> dict:
    """Run TF OD model on [H,W,3] RGB uint8. Returns {bboxes, labels}."""
    import numpy as np
    h, w = img.shape[0], img.shape[1]
    x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)  # [1,H,W,3]
    inp_name = model.session.get_inputs()[0].name
    with torch.inference_mode():
        num_det, scores, boxes = model(**{inp_name: x})
    num_det = int(num_det[0].item())
    scores = scores[0][:num_det]
    boxes = boxes[0][:num_det]  # ymin,xmin,ymax,xmax norm [0,1]
    keep = scores >= score_thresh
    boxes = boxes[keep]
    boxes_xyxy = torch.stack([
        boxes[:, 1] * w, boxes[:, 0] * h,
        boxes[:, 3] * w, boxes[:, 2] * h
    ], dim=1)
    return {"bboxes": boxes_xyxy, "labels": torch.zeros(len(boxes_xyxy), dtype=torch.long)}


class TfOdPredictor(nn.Module):
    """Base for TF OD format models (plate, face, etc)."""

    def __init__(self, model_path: str, classes: list[str], score_thresh: float = 0.5):
        super().__init__()
        self._model = torch.load(model_path, weights_only=False, map_location="cpu")
        self.score_thresh = score_thresh
        self.classes = classes

    def predict(self, image_path: str):
        return self.predict_from_array(_load_image_rgb(image_path))

    def predict_from_array(self, img: "np.ndarray"):
        """img: [H,W,3] RGB uint8 numpy. Avoids reload when blurring in place."""
        return _run_tf_od(self._model, img, self.score_thresh)


class PlatePredictor(TfOdPredictor):
    """Wraps plate ONNX (TF OD format) with predict() and .classes for anon.py."""

    def __init__(self, model_path: str, score_thresh: float = 0.5):
        super().__init__(model_path, ["plate"], score_thresh)


class FacePredictor(TfOdPredictor):
    """Wraps face ONNX (TF OD format) with predict() and .classes for anon.py."""

    def __init__(self, model_path: str, score_thresh: float = 0.5):
        super().__init__(model_path, ["face"], score_thresh)
