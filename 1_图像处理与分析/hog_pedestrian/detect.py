"""Run the trained HOG + linear SVM detector on a full scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image


WINDOW_W, WINDOW_H = 64, 128


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def nms(boxes: list[tuple[int, int, int, int]], scores: list[float], overlap: float) -> list[int]:
    if not boxes:
        return []
    data = np.asarray(boxes, dtype=np.float32)
    x1, y1 = data[:, 0], data[:, 1]
    x2, y2 = data[:, 0] + data[:, 2], data[:, 1] + data[:, 3]
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(np.asarray(scores))[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = area[i] + area[order[1:]] - intersection
        overlap_values = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = order[1:][overlap_values <= overlap]
    return keep


def candidate_detections(image: np.ndarray, model: dict, stride: int, scale_step: float):
    classifier = model["classifier"]
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    original_h, original_w = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    scale = 1.0
    while True:
        level_w = max(WINDOW_W, int(round(original_w * scale)))
        level_h = max(WINDOW_H, int(round(original_h * scale)))
        if level_w > original_w or level_h > original_h:
            break
        level = image if scale == 1.0 else cv2.resize(image, (level_w, level_h), interpolation=cv2.INTER_AREA)
        windows: list[np.ndarray] = []
        locations: list[tuple[int, int]] = []
        for y in range(0, level_h - WINDOW_H + 1, stride):
            for x in range(0, level_w - WINDOW_W + 1, stride):
                patch = cv2.cvtColor(level[y : y + WINDOW_H, x : x + WINDOW_W], cv2.COLOR_BGR2GRAY)
                windows.append(hog.compute(patch).reshape(-1))
                locations.append((x, y))
        if windows:
            decisions = classifier.decision_function(np.asarray(windows, dtype=np.float32))
            for (x, y), score in zip(locations, decisions):
                boxes.append((round(x / scale), round(y / scale), round(WINDOW_W / scale), round(WINDOW_H / scale)))
                scores.append(float(score))
        if level_w == WINDOW_W and level_h == WINDOW_H:
            break
        scale *= scale_step
    return boxes, scores


def detect(image: np.ndarray, model: dict, threshold: float, overlap: float, stride: int, scale_step: float):
    boxes, scores = candidate_detections(image, model, stride, scale_step)
    selected = [i for i, score in enumerate(scores) if score >= threshold]
    boxes = [boxes[i] for i in selected]
    scores = [scores[i] for i in selected]
    keep = nms(boxes, scores, overlap=overlap)
    return [(boxes[i], scores[i]) for i in keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/hog_svm.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pedestrian_detection.png"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--nms-overlap", type=float, default=None)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--scale-step", type=float, default=0.8)
    args = parser.parse_args()

    image = read_image(args.image)
    model = joblib.load(args.model)
    threshold = args.threshold if args.threshold is not None else model.get("detection_threshold", 0.8)
    overlap = args.nms_overlap if args.nms_overlap is not None else model.get("nms_overlap", 0.35)
    detections = detect(image, model, threshold, overlap, args.stride, args.scale_step)
    result = image.copy()
    for (x, y, w, h), score in detections:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(result, f"person {score:.2f}", (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV's imwrite can fail silently for non-ASCII Windows paths.
    ok, encoded = cv2.imencode(args.output.suffix or ".png", result)
    if not ok:
        raise RuntimeError(f"Could not encode output image: {args.output}")
    encoded.tofile(str(args.output))
    print(f"image={args.image}")
    print(f"detections={len(detections)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
