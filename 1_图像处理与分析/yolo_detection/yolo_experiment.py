"""Reproducible local YOLOv8 image detection experiment."""
import argparse
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL = ROOT / "models" / "yolov8n.pt"
OUTPUT = ROOT / "outputs"
# Keep framework settings with this experiment, including in restricted shells.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare():
    download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", MODEL)
    archive = DATA / "coco128.zip"
    download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", archive)
    dataset = DATA / "coco128"
    if not (dataset / "images" / "train2017").is_dir():
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                target = (DATA / entry.filename).resolve()
                if DATA.resolve() not in target.parents:
                    raise ValueError(f"Unsafe archive member: {entry.filename}")
            zipped.extractall(DATA)
    # Retain official class names while explicitly using a local dataset path.
    import ultralytics
    import yaml
    config = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg/datasets/coco128.yaml").read_text(encoding="utf-8"))
    config.update(path=dataset.as_posix(), train="images/train2017", val="images/train2017")
    config.pop("download", None)
    local_config = DATA / "coco128_local.yaml"
    local_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Local dataset: {local_config}")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "predict", "validate", "info", "all"])
    parser.add_argument("--source", help="Image or image directory; default: local COCO128")
    parser.add_argument("--device", default="auto", help="auto, cpu, or GPU index such as 0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    if args.command in {"prepare", "all"}:
        prepare()
    if args.command == "prepare":
        return
    import torch
    import ultralytics
    from ultralytics import YOLO
    if not MODEL.exists():
        raise FileNotFoundError("Run 'python yolo_experiment.py prepare' first.")
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    environment = {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "ultralytics": ultralytics.__version__,
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "cuda_available": torch.cuda.is_available(), "device": device,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "model": str(MODEL), "imgsz": args.imgsz}
    save_json(OUTPUT / "environment.json", environment)
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (OUTPUT / "environment-lock.txt").write_text(packages, encoding="utf-8")
    if args.command in {"info", "all"}:
        architecture = io.StringIO()
        with contextlib.redirect_stdout(architecture):
            print(model.model)
            print("\nParameter count:", sum(p.numel() for p in model.model.parameters()))
            print("\nModel YAML:", model.model.yaml)
        (OUTPUT / "model_structure.txt").write_text(architecture.getvalue(), encoding="utf-8")
        print(json.dumps(environment, ensure_ascii=False, indent=2))
        model.info()
    if args.command in {"predict", "all"}:
        source = args.source or str(DATA / "coco128/images/train2017")
        records = []
        results = model.predict(source=source, device=device, imgsz=args.imgsz, conf=args.conf,
                                save=True, save_txt=True, save_conf=True, stream=True,
                                project=str(OUTPUT), name="predict", exist_ok=True)
        for result in results:
            boxes = result.boxes.cpu()
            detections = [{"class_id": int(cls), "class_name": result.names[int(cls)],
                           "confidence": float(score), "xyxy": box}
                          for cls, score, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]
            records.append({"image": result.path, "detections": detections, "speed_ms": result.speed})
        save_json(OUTPUT / "detections.json", {"confidence_threshold": args.conf, "images": records})
        counts = {}
        for record in records:
            for detection in record["detections"]:
                name = detection["class_name"]
                counts[name] = counts.get(name, 0) + 1
        save_json(OUTPUT / "prediction_summary.json", {"image_count": len(records),
                  "detection_count": sum(counts.values()), "class_counts": counts})
        print(f"Detected {sum(counts.values())} objects in {len(records)} images.")
    if args.command in {"validate", "all"}:
        config = DATA / "coco128_local.yaml"
        if not config.exists():
            raise FileNotFoundError("Run prepare first to create coco128_local.yaml.")
        # Validation uses its standard low confidence threshold for PR curves.
        metrics = model.val(data=str(config), device=device, imgsz=args.imgsz, batch=8,
                            workers=0, plots=True, project=str(OUTPUT), name="val", exist_ok=True)
        save_json(OUTPUT / "metrics.json", {"dataset": "COCO128", "image_count": 128,
                  "note": "Small demonstration subset of COCO train2017; not an independent benchmark. No training performed.",
                  "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
                  "speed_ms": metrics.speed})


if __name__ == "__main__":
    main()
