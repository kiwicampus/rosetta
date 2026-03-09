#!/usr/bin/env python3
"""Blur license plates and faces in images. Input: dir (recursive), multiple files, or single file."""
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

from onnx_wrapper import PlatePredictor, FacePredictor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
PLATE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "plate.pt"
FACE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face.pt"


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
        # Run both models in parallel (inference releases GIL; threading avoids executor overhead)
        plate_results, face_results = [None], [None]

        def run_plate():
            plate_results[0] = plate_model.predict_from_array(img)

        def run_face():
            face_results[0] = face_model.predict_from_array(img)

        t1 = threading.Thread(target=run_plate)
        t2 = threading.Thread(target=run_face)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        plate_results, face_results = plate_results[0], face_results[0]
        for bbox in plate_results["bboxes"].int():
            _blur_roi(img, bbox.tolist())
        for bbox in face_results["bboxes"].int():
            _blur_roi(img, bbox.tolist())
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
    args = parser.parse_args()

    tic = time.perf_counter()
    # Load both models in parallel
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as ex:
        plate_fut = ex.submit(PlatePredictor, args.plate_model)
        face_fut = ex.submit(FacePredictor, args.face_model)
        plate_model = plate_fut.result()
        face_model = face_fut.result()
    # Eager warmup (avoids per-thread session init races)
    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(plate_model.predict_from_array, dummy).result()
        ex.submit(face_model.predict_from_array, dummy).result()
    pairs = collect_paths(args)
    if not pairs:
        print("No images to process.")
        return

    errors = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {
            ex.submit(process_one, plate_model, face_model, inp, out): inp
            for inp, out in pairs
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Blurring", unit="img"):
            err = f.result()
            if err:
                errors.append(err)
                tqdm.write(err)

    toc = time.perf_counter()
    if errors:
        tqdm.write(f"\n{len(errors)} error(s)")
    else:
        print(f"Done. {len(pairs)} image(s) saved.")
    print(f"Elapsed: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()
