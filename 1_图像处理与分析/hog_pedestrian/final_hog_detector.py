"""Detect pedestrians in one image with OpenCV's pretrained HOG + linear SVM."""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="HOG + SVM pedestrian detection")
    parser.add_argument("--image", type=Path, required=True, help="input scene image")
    parser.add_argument("--output", type=Path, default=Path("outputs/pedestrian_detection.png"))
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="SVM hit threshold; higher values reduce weak detections")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = detector.detectMultiScale(
        image,
        hitThreshold=args.threshold,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
    )

    result = image.copy()
    for x, y, width, height in boxes:
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 180, 0), 2)
        cv2.putText(result, "person", (x, max(22, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 180, 0), 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise OSError(f"Could not write output: {args.output}")
    print(f"Detected pedestrians: {len(boxes)}")
    print(f"Saved result: {args.output}")


if __name__ == "__main__":
    main()
