#!/usr/bin/env python3
"""Blur license plates and faces in images. Input: dir (recursive), multiple files, or single file."""
import argparse
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import tqdm

from onnx_wrapper import PlatePredictor, FacePredictor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
PLATE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "plate.pt"
FACE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face.pt"


class _SequentialAnonymizer:
    """Holds paths and device for sequential model loading. Only one model in VRAM at a time."""

    def __init__(self, plate_path: str, face_path: str, device: str):
        self.plate_path = plate_path
        self.face_path = face_path
        self.device = device


def _blur_roi(img_np, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_np.shape[1], int(x2)), min(img_np.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img_np[y1:y2, x1:x2]
    kh = max(3, ((y2 - y1) // 2) | 1)
    kw = max(3, ((x2 - x1) // 2) | 1)
    img_np[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)


def _iter_bboxes(results):
    """Extract bbox list from model results, handling torch/numpy."""
    b = results.get("bboxes", []) if results else []
    return b.int() if hasattr(b, "int") else b


def blur_image_in_place(img: np.ndarray, plate_model, face_model) -> None:
    """Blur plates and faces in img (HWC RGB uint8) in-place. Models must be loaded."""
    plate_results = plate_model.predict_from_array(img)
    face_results = face_model.predict_from_array(img)
    plate_results = plate_results or {"bboxes": []}
    face_results = face_results or {"bboxes": []}
    for bbox in _iter_bboxes(plate_results):
        _blur_roi(img, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))
    for bbox in _iter_bboxes(face_results):
        _blur_roi(img, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))


def anonymize_frames_batch(
    frames: list,
    image_keys: list,
    plate_model_or_anon,
    face_model=None,
) -> None:
    """Anonymize all image arrays in frames in-place.
    Frames are list of dicts; image_keys are keys whose values are HWC RGB uint8 numpy arrays.
    Skips depth/non-RGB; overwrites arrays in place (no suffix, no copy).

    Called as anonymize_frames_batch(frames, keys, *anonymizer). anonymizer is either
    (plate_model, face_model) or (_SequentialAnonymizer,) - single-element tuple for sequential."""
    # Collect all (frame, key) pairs with valid image arrays
    to_process: list[tuple[dict, str, np.ndarray]] = []
    for frame in frames:
        for key in image_keys:
            arr = frame.get(key)
            if arr is None or not isinstance(arr, np.ndarray):
                continue
            if arr.ndim != 3 or arr.shape[-1] != 3:
                continue  # Skip depth / non-RGB
            to_process.append((frame, key, arr))

    if not to_process:
        return

    sequential = isinstance(plate_model_or_anon, _SequentialAnonymizer)
    if sequential:
        _anonymize_frames_sequential(to_process, plate_model_or_anon)
    else:
        plate_model, face_model = plate_model_or_anon, face_model
        _anonymize_frames_dual_model(to_process, plate_model, face_model)


def _anonymize_frames_sequential(to_process: list, sa: "_SequentialAnonymizer") -> None:
    """Sequential: plate on all, unload, face on all. Half the VRAM."""
    sub_batch = 16
    plate_results = [None] * len(to_process)

    plate_model = PlatePredictor(sa.plate_path, 0.5, sa.device)
    plate_model.predict_from_array(np.zeros((100, 100, 3), dtype=np.uint8))  # warmup
    for start in range(0, len(to_process), sub_batch):
        end = min(start + sub_batch, len(to_process))
        for j in range(start, end):
            _, _, arr = to_process[j]
            plate_results[j] = plate_model.predict_from_array(arr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del plate_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    face_model = FacePredictor(sa.face_path, 0.5, sa.device)
    face_model.predict_from_array(np.zeros((100, 100, 3), dtype=np.uint8))
    face_results = [None] * len(to_process)
    for start in range(0, len(to_process), sub_batch):
        end = min(start + sub_batch, len(to_process))
        for j in range(start, end):
            _, _, arr = to_process[j]
            face_results[j] = face_model.predict_from_array(arr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for (_, _, arr), pr, fr in zip(to_process, plate_results, face_results):
        pr = pr or {"bboxes": []}
        fr = fr or {"bboxes": []}
        for bbox in _iter_bboxes(pr):
            _blur_roi(arr, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))
        for bbox in _iter_bboxes(fr):
            _blur_roi(arr, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))


def _anonymize_frames_dual_model(
    to_process: list, plate_model, face_model
) -> None:
    """Original dual-model flow (both in VRAM)."""
    sub_batch = 8
    plate_results = [None] * len(to_process)
    face_results = [None] * len(to_process)

    for start in range(0, len(to_process), sub_batch):
        end = min(start + sub_batch, len(to_process))
        chunk = to_process[start:end]
        for j, (_, _, arr) in enumerate(chunk):
            plate_results[start + j] = plate_model.predict_from_array(arr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for j, (_, _, arr) in enumerate(chunk):
            face_results[start + j] = face_model.predict_from_array(arr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for (_, _, arr), pr, fr in zip(to_process, plate_results, face_results):
        pr = pr or {"bboxes": []}
        fr = fr or {"bboxes": []}
        for bbox in _iter_bboxes(pr):
            _blur_roi(arr, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))
        for bbox in _iter_bboxes(fr):
            _blur_roi(arr, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))


def create_anonymizer(
    device: str = "auto",
    plate_path: Path | None = None,
    face_path: Path | None = None,
    sequential: bool = True,
):
    """Create anonymizer for reuse. Returns (plate_model, face_model) or (SequentialAnonymizer,).

    sequential=True: returns (_SequentialAnonymizer,) - loads one model at a time, halves VRAM.
    sequential=False: loads both models, returns (plate_model, face_model)."""
    plate_path = plate_path or PLATE_MODEL_PATH
    face_path = face_path or FACE_MODEL_PATH
    if sequential:
        return (_SequentialAnonymizer(str(plate_path), str(face_path), device),)
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as ex:
        plate_fut = ex.submit(PlatePredictor, str(plate_path), 0.5, device)
        face_fut = ex.submit(FacePredictor, str(face_path), 0.5, device)
        plate_model = plate_fut.result()
        face_model = face_fut.result()
    plate_model.predict_from_array(dummy)
    face_model.predict_from_array(dummy)
    return plate_model, face_model


def process_one(
    plate_model: PlatePredictor,
    face_model: FacePredictor,
    in_path: Path,
    out_path: Path,
) -> str | None:
    """Load, detect plates+faces, blur all, save. Returns error msg or None on success."""
    try:
        img = cv2.imread(str(in_path))
        if img is None:
            return f"Failed to read {in_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Run sequentially to avoid GPU OOM (both models compete for ~640MB each in NMS)
        plate_results = plate_model.predict_from_array(img)
        face_results = face_model.predict_from_array(img)
        plate_results = plate_results or {"bboxes": []}
        face_results = face_results or {"bboxes": []}

        for bbox in _iter_bboxes(plate_results):
            _blur_roi(img, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))
        for bbox in _iter_bboxes(face_results):
            _blur_roi(img, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return None
    except Exception as e:
        return f"{in_path}: {e}"


def _out_path(in_path: Path, override: bool) -> Path:
    if override:
        return in_path.resolve()
    return (in_path.parent / f"{in_path.stem}_blurred{in_path.suffix}").resolve()


def collect_paths(args) -> list[tuple[Path, Path]]:
    """Return list of (input_path, output_path)."""
    pairs: list[tuple[Path, Path]] = []

    for raw in args.input:
        p = Path(raw)
        if not p.exists():
            tqdm.write(f"Skip (not found): {p}")
            continue
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    pairs.append((f.resolve(), _out_path(f, args.override)))
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            pairs.append((p.resolve(), _out_path(p, args.override)))
        else:
            tqdm.write(f"Skip (not file/dir): {p}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Blur license plates and faces in images")
    parser.add_argument("input", nargs="+", help="File(s) or dir(s). Dir = recursive.")
    parser.add_argument("--override", action="store_true", help="Override input image instead of writing to _blurred file")
    parser.add_argument("-j", "--jobs", type=int, default=None, required=False, help="Thread count (default: min(32, cpu_count+4))")
    parser.add_argument("--plate-model", default=str(PLATE_MODEL_PATH), required=False, help="Plate model path (default: scripts/models/plate.pt)")
    parser.add_argument("--face-model", default=str(FACE_MODEL_PATH), required=False, help="Face model path (default: scripts/models/face.pt)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for inference (default: auto)")
    parser.add_argument("--no-sequential", action="store_true", help="Load both models (higher VRAM, faster if no OOM)")
    args = parser.parse_args()

    tic = time.perf_counter()
    pairs = collect_paths(args)
    if not pairs:
        print("No images to process.")
        return

    use_gpu = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    sequential = not args.no_sequential and use_gpu

    if sequential:
        _run_sequential_file(pairs, args.plate_model, args.face_model, args.device)
    else:
        _run_dual_model_file(pairs, args.plate_model, args.face_model, args.device, args.jobs)

    toc = time.perf_counter()
    print(f"Elapsed: {toc - tic:.2f}s")


def _run_sequential_file(pairs, plate_path, face_path, device):
    """Sequential: plate on all, unload, face on all. Halves VRAM."""
    plate_results = {}
    plate_model = PlatePredictor(plate_path, 0.5, device)
    plate_model.predict_from_array(np.zeros((100, 100, 3), dtype=np.uint8))
    errors = []
    for inp, out in tqdm(pairs, desc="Plate", unit="img"):
        try:
            img = cv2.imread(str(inp))
            if img is None:
                errors.append(f"Failed to read {inp}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plate_results[inp] = plate_model.predict_from_array(img)
        except Exception as e:
            errors.append(f"{inp}: {e}")
    del plate_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    face_model = FacePredictor(face_path, 0.5, device)
    face_model.predict_from_array(np.zeros((100, 100, 3), dtype=np.uint8))
    for inp, out in tqdm(pairs, desc="Face + blur", unit="img"):
        try:
            img = cv2.imread(str(inp))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_results = face_model.predict_from_array(img)
            pr = plate_results.get(inp) or {"bboxes": []}
            fr = face_results or {"bboxes": []}
            for bbox in _iter_bboxes(pr):
                _blur_roi(img, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))
            for bbox in _iter_bboxes(fr):
                _blur_roi(img, bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            errors.append(f"{inp}: {e}")
    for e in errors:
        tqdm.write(e)
    success = len(pairs) - len(errors)
    print(f"Done. {success} image(s) saved." if not errors else f"\n{len(errors)} error(s)")


def _run_dual_model_file(pairs, plate_path, face_path, device, jobs):
    """Original dual-model flow with threadpool."""
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as ex:
        plate_fut = ex.submit(PlatePredictor, plate_path, 0.5, device)
        face_fut = ex.submit(FacePredictor, face_path, 0.5, device)
        plate_model = plate_fut.result()
        face_model = face_fut.result()
    plate_model.predict_from_array(dummy)
    face_model.predict_from_array(dummy)

    if jobs is None:
        use_gpu = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        jobs = 1 if use_gpu else min(32, (os.cpu_count() or 4) + 4)

    errors = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {
            ex.submit(process_one, plate_model, face_model, inp, out): inp
            for inp, out in pairs
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Blurring", unit="img"):
            err = f.result()
            if err:
                errors.append(err)
                tqdm.write(err)
    if errors:
        tqdm.write(f"\n{len(errors)} error(s)")
    else:
        print(f"Done. {len(pairs)} image(s) saved.")


if __name__ == "__main__":
    main()
