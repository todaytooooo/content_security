"""Prepare INRIA samples and train a HOG + linear SVM detector."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

from detect import candidate_detections, nms


WINDOW = (64, 128)  # width, height


def read_image(path: Path) -> np.ndarray:
    """Read an image reliably, including INRIA PNGs rejected by OpenCV."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        (64, 128), (16, 16), (8, 8), (8, 8), 9
    )


def feature(hog: cv2.HOGDescriptor, image: np.ndarray) -> np.ndarray:
    if image.shape[1] != 64 or image.shape[0] != 128:
        image = cv2.resize(image, WINDOW, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).reshape(-1).astype(np.float32)


def read_boxes(annotation_path: Path) -> list[tuple[int, int, int, int]]:
    """Read INRIA PASCAL-style (xmin, ymin) - (xmax, ymax) annotations."""
    text = annotation_path.read_text(encoding="latin1")
    pattern = re.compile(
        r'Bounding box for object .*? : \((\d+), (\d+)\) - \((\d+), (\d+)\)'
    )
    return [tuple(map(int, match.groups())) for match in pattern.finditer(text)]


def full_person_crop(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Create a 1:2 window that contains the annotated person plus small context."""
    x1, y1, x2, y2 = box
    image_h, image_w = image.shape[:2]
    person_w = x2 - x1 + 1
    person_h = y2 - y1 + 1
    # The frame must cover the person horizontally and vertically before resize.
    frame_h = max(person_h * 1.10, person_w * 2.10) * scale
    frame_w = frame_h / 2.0
    frame_h = min(frame_h, float(image_h))
    frame_w = min(frame_w, float(image_w))
    center_x = (x1 + x2) / 2.0 + shift_x * person_w
    center_y = (y1 + y2) / 2.0 + shift_y * person_h
    left = min(max(center_x - frame_w / 2.0, 0.0), image_w - frame_w)
    top = min(max(center_y - frame_h / 2.0, 0.0), image_h - frame_h)
    right = min(image_w, max(int(round(left + frame_w)), int(round(left)) + 1))
    bottom = min(image_h, max(int(round(top + frame_h)), int(round(top)) + 1))
    crop = image[int(round(top)) : bottom, int(round(left)) : right]
    return cv2.resize(crop, WINDOW, interpolation=cv2.INTER_AREA)


def crop_negative(image: np.ndarray, rng: random.Random, count: int) -> list[np.ndarray]:
    height, width = image.shape[:2]
    if width < 64 or height < 128:
        return [cv2.resize(image, WINDOW, interpolation=cv2.INTER_AREA)]
    max_x, max_y = width - 64, height - 128
    crops = []
    for _ in range(count):
        x = rng.randint(0, max_x)
        y = rng.randint(0, max_y)
        crops.append(image[y : y + 128, x : x + 64])
    return crops


def build_samples(data_root: Path, negative_crops: int, seed: int, positive_jitter: int):
    positive_dir = data_root / "Train" / "pos"
    annotation_dir = data_root / "Train" / "annotations"
    negative_dir = data_root / "Train" / "neg"
    annotation_paths = sorted(annotation_dir.glob("*.txt"))
    negative_paths = sorted(
        p for p in negative_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not annotation_paths or not negative_paths:
        raise FileNotFoundError(
            f"Expected annotations under {annotation_dir} and negatives under {negative_dir}."
        )

    rng = random.Random(seed)
    hog = hog_descriptor()
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    jitter_variants = [(0.0, 0.0, 1.0), (-0.04, 0.03, 1.05), (0.04, -0.03, 0.95)]
    for annotation_path in annotation_paths:
        path = positive_dir / f"{annotation_path.stem}.png"
        if not path.exists():
            raise FileNotFoundError(f"Annotated source image is missing: {path}")
        image = read_image(path)
        for box in read_boxes(annotation_path):
            for shift_x, shift_y, scale in jitter_variants[: positive_jitter + 1]:
                crop = full_person_crop(image, box, shift_x, shift_y, scale)
                features.append(feature(hog, crop))
                labels.append(1)
                groups.append("pos:" + path.name)
                # Mirror augmentation increases pose and direction robustness.
                features.append(feature(hog, cv2.flip(crop, 1)))
                labels.append(1)
                groups.append("pos:" + path.name)

    for path in negative_paths:
        image = read_image(path)
        for index, crop in enumerate(crop_negative(image, rng, negative_crops)):
            features.append(feature(hog, crop))
            labels.append(0)
            # Every crop from one scene must stay in the same data split.
            groups.append(f"neg:{path.name}")

    return np.asarray(features), np.asarray(labels), np.asarray(groups), hog, negative_paths


def iou(box: tuple[float, float, float, float], target: tuple[int, int, int, int]) -> float:
    x, y, width, height = box
    x1, y1, x2, y2 = target
    inter_left, inter_top = max(x, x1), max(y, y1)
    inter_right, inter_bottom = min(x + width, x2 + 1), min(y + height, y2 + 1)
    intersection = max(0.0, inter_right - inter_left) * max(0.0, inter_bottom - inter_top)
    union = width * height + (x2 - x1 + 1) * (y2 - y1 + 1) - intersection
    return intersection / union if union else 0.0


def mine_positive_scene_hard_negatives(
    classifier: LinearSVC,
    hog: cv2.HOGDescriptor,
    data_root: Path,
    training_names: set[str],
    max_images: int,
    max_per_image: int,
    score_threshold: float,
    seed: int,
) -> np.ndarray:
    """Mine only high-score windows with negligible overlap with all person boxes."""
    rng = random.Random(seed)
    image_dir = data_root / "Train" / "pos"
    annotation_dir = data_root / "Train" / "annotations"
    candidates = [path for path in image_dir.glob("*.png") if path.name in training_names]
    rng.shuffle(candidates)
    mined: list[np.ndarray] = []
    for number, path in enumerate(candidates[:max_images], start=1):
        image = read_image(path)
        boxes = read_boxes(annotation_dir / f"{path.stem}.txt")
        original_h, original_w = image.shape[:2]
        level, scale = image, 1.0
        found: list[tuple[float, np.ndarray]] = []
        while level.shape[0] >= 128 and level.shape[1] >= 64:
            features: list[np.ndarray] = []
            patches: list[np.ndarray] = []
            locations: list[tuple[int, int]] = []
            for y in range(0, level.shape[0] - 128 + 1, 16):
                for x in range(0, level.shape[1] - 64 + 1, 16):
                    patch = level[y : y + 128, x : x + 64]
                    features.append(feature(hog, patch))
                    patches.append(patch)
                    locations.append((x, y))
            if features:
                scores = classifier.decision_function(np.asarray(features, dtype=np.float32))
                for score, patch, (x, y) in zip(scores, patches, locations):
                    candidate_box = (x / scale, y / scale, 64 / scale, 128 / scale)
                    # Leave any partial-person candidate unlabeled instead of calling it background.
                    if score >= score_threshold and max((iou(candidate_box, box) for box in boxes), default=0.0) < 0.10:
                        found.append((float(score), patch))
            next_width, next_height = int(round(original_w * scale * 0.8)), int(round(original_h * scale * 0.8))
            if next_width < 64 or next_height < 128:
                break
            scale *= 0.8
            level = cv2.resize(image, (next_width, next_height), interpolation=cv2.INTER_AREA)
        found.sort(key=lambda item: item[0], reverse=True)
        mined.extend(feature(hog, patch) for _, patch in found[:max_per_image])
        if number % 25 == 0 or number == min(len(candidates), max_images):
            print(f"Positive-scene mining: {number}/{min(len(candidates), max_images)} scenes, {len(mined)} windows")
    return np.asarray(mined, dtype=np.float32)


def choose_detection_parameters(
    classifier: LinearSVC,
    data_root: Path,
    validation_positive_names: set[str],
    validation_negative_names: set[str],
    scene_limit: int,
) -> tuple[float, float, dict]:
    """Tune detector settings on held-out training scenes, never on Test."""
    model = {"classifier": classifier}
    positive_dir = data_root / "Train" / "pos"
    negative_dir = data_root / "Train" / "neg"
    annotation_dir = data_root / "Train" / "annotations"
    positives = sorted(path for path in positive_dir.glob("*.png") if path.name in validation_positive_names)[:scene_limit]
    negatives = sorted(path for path in negative_dir.iterdir() if path.name in validation_negative_names)[:scene_limit]
    scenes: list[tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]], list[float]]] = []
    for path in positives + negatives:
        boxes, scores = candidate_detections(read_image(path), model, stride=8, scale_step=0.8)
        truths = read_boxes(annotation_dir / f"{path.stem}.txt") if path in positives else []
        scenes.append((truths, boxes, scores))
    trials = []
    for threshold in (0.6, 0.8, 1.0, 1.2):
        for overlap in (0.25, 0.35, 0.45):
            tp = fp = fn = 0
            for truths, boxes, scores in scenes:
                chosen = [index for index, score in enumerate(scores) if score >= threshold]
                selected_boxes = [boxes[index] for index in chosen]
                selected_scores = [scores[index] for index in chosen]
                keep = nms(selected_boxes, selected_scores, overlap)
                matched: set[int] = set()
                for index in keep:
                    overlaps = [iou(selected_boxes[index], truth) for truth in truths]
                    best = int(np.argmax(overlaps)) if overlaps else -1
                    if best >= 0 and overlaps[best] >= 0.5 and best not in matched:
                        tp += 1
                        matched.add(best)
                    else:
                        fp += 1
                fn += len(truths) - len(matched)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            trials.append({"threshold": threshold, "nms_overlap": overlap, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    best = max(trials, key=lambda item: item["f1"])
    print("Detector calibration:", best)
    return float(best["threshold"]), float(best["nms_overlap"]), {"scenes": len(scenes), "best": best, "trials": trials}


def mine_hard_negatives(
    classifier: LinearSVC,
    hog: cv2.HOGDescriptor,
    negative_paths: list[Path],
    training_names: set[str],
    max_images: int,
    max_per_image: int,
    score_threshold: float,
    seed: int,
) -> np.ndarray:
    """Collect high-scoring non-person windows from training background scenes."""
    rng = random.Random(seed)
    candidates = [path for path in negative_paths if path.name in training_names]
    rng.shuffle(candidates)
    mined: list[np.ndarray] = []
    for number, path in enumerate(candidates[:max_images], start=1):
        image = read_image(path)
        height, width = image.shape[:2]
        level = image
        scale = 1.0
        found: list[tuple[float, np.ndarray]] = []
        while level.shape[0] >= 128 and level.shape[1] >= 64:
            windows: list[np.ndarray] = []
            patches: list[np.ndarray] = []
            for y in range(0, level.shape[0] - 128 + 1, 16):
                for x in range(0, level.shape[1] - 64 + 1, 16):
                    patch = level[y : y + 128, x : x + 64]
                    windows.append(feature(hog, patch))
                    patches.append(patch)
            if windows:
                scores = classifier.decision_function(np.asarray(windows, dtype=np.float32))
                found.extend((float(score), patch) for score, patch in zip(scores, patches) if score >= score_threshold)
            next_width = int(round(width * scale * 0.8))
            next_height = int(round(height * scale * 0.8))
            if next_width < 64 or next_height < 128:
                break
            scale *= 0.8
            level = cv2.resize(image, (next_width, next_height), interpolation=cv2.INTER_AREA)
        found.sort(key=lambda item: item[0], reverse=True)
        mined.extend(feature(hog, patch) for _, patch in found[:max_per_image])
        if number % 25 == 0 or number == min(len(candidates), max_images):
            print(f"Hard-negative mining: {number}/{min(len(candidates), max_images)} scenes, {len(mined)} windows")
    return np.asarray(mined, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/INRIAPerson_complete/INRIAPerson"),
    )
    parser.add_argument("--model", type=Path, default=Path("models/hog_svm.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("models/hog_svm.metrics.json"))
    parser.add_argument("--negative-crops", type=int, default=4)
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negative-images", type=int, default=200)
    parser.add_argument("--hard-negatives-per-image", type=int, default=5)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.3)
    parser.add_argument("--positive-jitter", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--positive-hard-negative-images", type=int, default=200)
    parser.add_argument("--positive-hard-negatives-per-image", type=int, default=5)
    parser.add_argument("--calibration-scenes", type=int, default=15)
    args = parser.parse_args()

    x_data, y_data, groups, hog, negative_paths = build_samples(
        args.data_root, args.negative_crops, args.seed, args.positive_jitter
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x_data, y_data, groups))
    classifier = LinearSVC(C=args.c, class_weight="balanced", max_iter=10000, random_state=args.seed)
    classifier.fit(x_data[train_idx], y_data[train_idx])
    training_negative_names = {group.removeprefix("neg:") for group in groups[train_idx] if group.startswith("neg:")}
    hard_features = mine_hard_negatives(
        classifier,
        hog,
        negative_paths,
        training_negative_names,
        args.hard_negative_images,
        args.hard_negatives_per_image,
        args.hard_negative_threshold,
        args.seed,
    )
    training_positive_names = {group.removeprefix("pos:") for group in groups[train_idx] if group.startswith("pos:")}
    positive_scene_hard_features = mine_positive_scene_hard_negatives(
        classifier,
        hog,
        args.data_root,
        training_positive_names,
        args.positive_hard_negative_images,
        args.positive_hard_negatives_per_image,
        args.hard_negative_threshold,
        args.seed,
    )
    hard_sets = [item for item in (hard_features, positive_scene_hard_features) if len(item)]
    all_hard_features = np.vstack(hard_sets) if hard_sets else np.empty((0, x_data.shape[1]), dtype=np.float32)
    if len(all_hard_features):
        x_train = np.vstack((x_data[train_idx], all_hard_features))
        y_train = np.concatenate((y_data[train_idx], np.zeros(len(all_hard_features), dtype=np.int64)))
        classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_data[val_idx])
    report = classification_report(y_data[val_idx], prediction, output_dict=True, zero_division=0)
    validation_positive_names = {group.removeprefix("pos:") for group in groups[val_idx] if group.startswith("pos:")}
    validation_negative_names = {group.removeprefix("neg:") for group in groups[val_idx] if group.startswith("neg:")}
    detection_threshold, nms_overlap, calibration = choose_detection_parameters(
        classifier,
        args.data_root,
        validation_positive_names,
        validation_negative_names,
        args.calibration_scenes,
    )

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "window_size": WINDOW,
            "hog": {"winSize": [64, 128], "blockSize": [16, 16], "blockStride": [8, 8], "cellSize": [8, 8], "nbins": 9},
            "positive_samples": "Train/pos cropped from Train/annotations bounding boxes",
            "detection_threshold": detection_threshold,
            "nms_overlap": nms_overlap,
        },
        args.model,
    )
    metrics = {
        "samples": int(len(y_data)),
        "positive_samples": int((y_data == 1).sum()),
        "negative_samples": int((y_data == 0).sum()),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "hard_negative_samples": int(len(hard_features)),
        "positive_scene_hard_negative_samples": int(len(positive_scene_hard_features)),
        "hard_negative_images_scanned": min(args.hard_negative_images, len(training_negative_names)),
        "detection_calibration": calibration,
        "c": args.c,
        "accuracy": float(accuracy_score(y_data[val_idx], prediction)),
        "classification_report": report,
    }
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {args.model}")


if __name__ == "__main__":
    main()
