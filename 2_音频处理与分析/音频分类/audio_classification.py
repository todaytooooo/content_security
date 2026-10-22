"""UrbanSound8K: Log-Mel + LeNet-5 (audio adaptation), no pretrained weights.

Run from any working directory; see README.md for the experimental protocol.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
import sklearn
import soundfile as sf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent
CLASSES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
           "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
FEATURE_CONFIG = dict(version=1, sample_rate=16000, seconds=4, n_fft=512,
                      hop=256, n_mels=64, time_bins=128, scale="HTK", top_db=80)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_root(given=None):
    if given:
        root = Path(given).resolve()
        if not (root / "metadata/UrbanSound8K.csv").is_file():
            raise FileNotFoundError(f"Not a dataset root (metadata/UrbanSound8K.csv missing): {root}")
        return root
    matches = sorted(BASE.glob("**/metadata/UrbanSound8K.csv"))
    if len(matches) != 1:
        raise ValueError("Specify --data-root: expected exactly one metadata/UrbanSound8K.csv")
    return matches[0].parent.parent


def read_metadata(root):
    with (root / "metadata/UrbanSound8K.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8732:
        raise ValueError(f"Expected 8732 excerpts, found {len(rows)}")
    files, sources = set(), {}
    for row in rows:
        fold, label = int(row["fold"]), int(row["classID"])
        if not 1 <= fold <= 10 or not 0 <= label < 10 or CLASSES[label] != row["class"]:
            raise ValueError(f"Invalid metadata: {row}")
        key = (fold, row["slice_file_name"])
        if key in files:
            raise ValueError(f"Duplicate sample: {key}")
        files.add(key)
        sources.setdefault(row["fsID"], set()).add(fold)
        if not (root / "audio" / f"fold{fold}" / row["slice_file_name"]).is_file():
            raise FileNotFoundError(f"Missing audio: {key}")
    shared = {source: sorted(folds) for source, folds in sources.items() if len(folds) > 1}
    if shared:
        print(f"Source audit: {len(shared)} recordings span official folds; "
              "overlapping sources will be purged from training/validation.", flush=True)
    return rows


def load_audio(path):
    # scipy supports signed PCM (including left-justified 24 bit) and float WAV.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        try:
            sr, audio = wavfile.read(path)
        except ValueError as exc:
            # Some official excerpts use DVI/IMA ADPCM, unsupported by scipy.
            if "Unknown wave file format" not in str(exc):
                raise
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
    dtype = audio.dtype
    audio = audio.astype(np.float32)
    if np.issubdtype(dtype, np.signedinteger):
        audio /= float(2 ** (np.iinfo(dtype).bits - 1))
    elif dtype == np.uint8:
        audio = (audio - 128.0) / 128.0
    elif not np.issubdtype(dtype, np.floating):
        raise ValueError(f"Unsupported WAV type: {dtype}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    if sr != FEATURE_CONFIG["sample_rate"]:
        gcd = math.gcd(sr, FEATURE_CONFIG["sample_rate"])
        audio = resample_poly(audio, FEATURE_CONFIG["sample_rate"] // gcd, sr // gcd)
    length = FEATURE_CONFIG["sample_rate"] * FEATURE_CONFIG["seconds"]
    return np.pad(audio[:length], (0, max(0, length - len(audio)))).astype(np.float32)


def mel_filterbank():
    cfg = FEATURE_CONFIG
    max_mel = 2595 * np.log10(1 + cfg["sample_rate"] / 2 / 700)
    edges = 700 * (10 ** (np.linspace(0, max_mel, cfg["n_mels"] + 2) / 2595) - 1)
    hz = np.fft.rfftfreq(cfg["n_fft"], 1 / cfg["sample_rate"])
    up = (hz[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    down = (edges[2:, None] - hz[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    bank = np.maximum(0, np.minimum(up, down)).astype(np.float32)
    if not np.all(bank.sum(axis=1) > 0):
        raise ValueError("Empty Mel filters")
    return bank


MEL_BANK = mel_filterbank()


def log_mel(audio):
    cfg = FEATURE_CONFIG
    _, _, spectrum = stft(audio, fs=cfg["sample_rate"], window="hann",
                          nperseg=cfg["n_fft"], noverlap=cfg["n_fft"] - cfg["hop"],
                          boundary="zeros", padded=True)
    power = MEL_BANK @ (np.abs(spectrum) ** 2)
    db = 10 * np.log10(np.maximum(power, 1e-10))
    db = np.maximum(db, db.max() - cfg["top_db"])
    # Fixed linear time interpolation; no parameters learned from validation/test.
    old = np.linspace(0, 1, db.shape[1])
    new = np.linspace(0, 1, cfg["time_bins"])
    return np.stack([np.interp(new, old, channel) for channel in db]).astype(np.float32)


def prepare(root, cache):
    rows = read_metadata(root)
    metadata_hash = hashlib.sha256((root / "metadata/UrbanSound8K.csv").read_bytes()).hexdigest()
    signature = json.dumps(dict(config=FEATURE_CONFIG, metadata_sha256=metadata_hash), sort_keys=True)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "logmel_v1.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["signature"]) != signature:
                raise ValueError("Cache configuration differs; use a new --cache directory")
            x, y, folds = archive["x"], archive["y"], archive["folds"]
        if x.shape != (len(rows), 64, 128) or not np.isfinite(x).all():
            raise ValueError("Invalid feature cache")
        if not np.array_equal(y, [int(r["classID"]) for r in rows]) or not np.array_equal(folds, [int(r["fold"]) for r in rows]):
            raise ValueError("Cache labels/folds do not match metadata")
        return rows, x, y, folds
    x = np.empty((len(rows), 64, 128), dtype=np.float32)
    start = time.time()
    with threadpool_limits(limits=1):
        for i, row in enumerate(rows):
            audio = load_audio(root / "audio" / f"fold{row['fold']}" / row["slice_file_name"])
            x[i] = log_mel(audio)
            if (i + 1) % 500 == 0 or i + 1 == len(rows):
                print(f"Features {i + 1}/{len(rows)} ({time.time() - start:.1f}s)", flush=True)
    y = np.array([int(r["classID"]) for r in rows], dtype=np.int64)
    folds = np.array([int(r["fold"]) for r in rows], dtype=np.int64)
    np.savez_compressed(path, x=x, y=y, folds=folds, signature=np.array(signature))
    return rows, x, y, folds


class LeNet5(nn.Module):
    """LeNet-5 adaptation: dense C3, fixed average pooling, softmax CE head.

    1x64x128 -> 6x60x124 -> 6x30x62 -> 16x26x58 -> 16x13x29
    -> 120 -> 84 -> 10. Tanh follows the original activation choice.
    """
    def __init__(self, head="dense"):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
                                      nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2))
        if head == "dense":
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 13 * 29, 120), nn.Tanh(),
                                            nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10))
        elif head == "gap":
            self.classifier = nn.Sequential(nn.Conv2d(16, 64, 1), nn.ReLU(),
                                            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                            nn.Dropout(0.3), nn.Linear(64, 10))
        else:
            raise ValueError(f"Unknown classifier head: {head}")

    def forward(self, x):
        return self.classifier(self.features(x))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_masks(folds, test_fold, val_fold):
    if test_fold == val_fold:
        raise ValueError("Validation fold must differ from test fold")
    train = (folds != test_fold) & (folds != val_fold)
    return train, folds == val_fold, folds == test_fold


def source_safe_split(rows, folds, test_fold, val_fold):
    train, val, test = split_masks(folds, test_fold, val_fold)
    sources = np.array([r["fsID"] for r in rows])
    # Preserve official test fold. Test > validation > train precedence.
    val &= ~np.isin(sources, sources[test])
    train &= ~np.isin(sources, sources[test | val])
    groups = [set(sources[mask]) for mask in [train, val, test]]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    if not all(mask.any() for mask in [train, val, test]):
        raise ValueError("Empty split after source purging")
    return train, val, test


def loader(x, y, mask, mean, std, batch, shuffle=False):
    data = torch.from_numpy(((x[mask] - mean) / std)[:, None].astype(np.float32))
    return DataLoader(TensorDataset(data, torch.from_numpy(y[mask])), batch_size=batch,
                      shuffle=shuffle, num_workers=0)


def run_epoch(model, batches, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, truth, predictions = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in batches:
            inputs, labels = inputs.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            truth.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return total_loss / len(truth), accuracy_score(truth, predictions), truth, predictions


def save_evaluation(out, rows, mask, truth, predictions, loss):
    report = classification_report(truth, predictions, labels=list(range(10)),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predictions, labels=list(range(10)))
    result = dict(test_loss=loss, accuracy=accuracy_score(truth, predictions),
                  macro_f1=f1_score(truth, predictions, labels=list(range(10)), average="macro", zero_division=0),
                  n_test=len(truth), classification_report=report, confusion_matrix=matrix.tolist())
    write_json(out / "metrics.json", result)
    with (out / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "fold", "true_id", "true_class", "predicted_id", "predicted_class"])
        for row, label, pred in zip([r for r, keep in zip(rows, mask) if keep], truth, predictions):
            writer.writerow([row["slice_file_name"], row["fold"], label, CLASSES[label], pred, CLASSES[pred]])
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="True class", title="UrbanSound8K - test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return result


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def model_from_checkpoint(checkpoint):
    # Historical checkpoints did not store a head field and use the dense head.
    model = LeNet5(head=checkpoint.get("config", {}).get("head", "dense"))
    model.load_state_dict(checkpoint["state_dict"])
    return model


def train(args, rows, x, y, folds, test_fold, val_fold):
    seed_everything(args.seed)
    out = args.output / f"test_fold{test_fold}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError(f"Preserving existing experiment: choose a new --output ({out})")
    train_mask, val_mask, test_mask = source_safe_split(rows, folds, test_fold, val_fold)
    mean = x[train_mask].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(x[train_mask].std(axis=(0, 2), keepdims=True), 1e-6)
    train_loader = loader(x, y, train_mask, mean, std, args.batch_size, True)
    val_loader = loader(x, y, val_mask, mean, std, args.batch_size)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = LeNet5(head=args.head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = dict(model="LeNet-5 audio adaptation", head=args.head,
                  head_dropout=0.3 if args.head == "gap" else 0.0,
                  features=FEATURE_CONFIG, classes=CLASSES,
                  seed=args.seed, test_fold=test_fold, val_fold=val_fold,
                  train_folds=sorted(set(folds[train_mask].tolist())),
                  n_train=int(train_mask.sum()), n_val=int(val_mask.sum()), n_test=int(test_mask.sum()),
                  split_protocol="official folds, source-purged train/validation; full test fold retained",
                  n_purged=int((~(train_mask | val_mask | test_mask)).sum()),
                  epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                  lr=args.lr, weight_decay=args.weight_decay, optimizer="Adam", device=str(device),
                  parameters=sum(p.numel() for p in model.parameters()),
                  metadata_sha256=hashlib.sha256((args.resolved_root / "metadata/UrbanSound8K.csv").read_bytes()).hexdigest(),
                  environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                   numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    write_json(out / "config.json", config)
    with (out / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "fsID", "fold", "class", "split"])
        for i, row in enumerate(rows):
            split = "train" if train_mask[i] else "validation" if val_mask[i] else "test" if test_mask[i] else "purged"
            writer.writerow([row["slice_file_name"], row["fsID"], row["fold"], row["class"], split])
    print(json.dumps(config, indent=2), flush=True)
    history, best, stale = [], float("inf"), 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, device)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                            val_loss=val_loss, val_accuracy=val_acc))
        write_json(out / "history.json", history)
        print(f"Epoch {epoch:03d} train loss={train_loss:.4f} acc={train_acc:.4f} "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} elapsed={time.time()-start:.1f}s", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                            best_epoch=epoch, config=config), out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    # Only the validation loss selects a checkpoint. Test is evaluated once afterward.
    checkpoint = load_checkpoint(out / "best.pt")
    model.load_state_dict(checkpoint["state_dict"])
    test_loader = loader(x, y, test_mask, mean, std, args.batch_size)
    loss, _, truth, predictions = run_epoch(model, test_loader, device)
    result = save_evaluation(out, rows, test_mask, truth, predictions, loss)
    result.update(best_epoch=checkpoint["best_epoch"], epochs_run=len(history),
                  training_seconds=time.time() - start)
    write_json(out / "metrics.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, suffix in zip(axes, ["loss", "accuracy"]):
        for split in ["train", "val"]:
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{suffix}"] for r in history], label=split)
        ax.axvline(checkpoint["best_epoch"], color="gray", ls="--", label="selected epoch")
        ax.set(xlabel="Epoch", ylabel=suffix.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    sample = int(np.flatnonzero(train_mask)[0])
    image = ax.imshow(x[sample], origin="lower", aspect="auto", extent=[0, 4, 0, 64], cmap="magma")
    ax.set(xlabel="Time (s)", ylabel="Mel band", title=f"Log-Mel: {CLASSES[y[sample]]}")
    fig.colorbar(image, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out / "logmel_example.png", dpi=150)
    plt.close(fig)
    print(f"TEST fold{test_fold}: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}; {out}", flush=True)
    return result


def smoke_test():
    seed_everything(42)
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):
        silent = log_mel(np.zeros(64000, np.float32))
        tone = log_mel(np.sin(2 * np.pi * 440 * np.arange(64000) / 16000).astype(np.float32))
    assert silent.shape == tone.shape == (64, 128)
    assert np.isfinite(silent).all() and np.isfinite(tone).all()
    model = LeNet5()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 1, 64, 128)
    before = model.features[0].weight.detach().clone()
    loss = nn.functional.cross_entropy(model(inputs), torch.tensor([0, 9]))
    loss.backward()
    optimizer.step()
    assert model(inputs).shape == (2, 10)
    assert not torch.equal(before, model.features[0].weight)
    masks = split_masks(np.arange(1, 11), 10, 9)
    assert [int(m.sum()) for m in masks] == [8, 1, 1]
    assert np.all(sum(m.astype(int) for m in masks) == 1)
    safe = source_safe_split([dict(fsID=s) for s in ["a", "b", "c", "c"]], np.array([1, 9, 1, 10]), 10, 9)
    assert [m.tolist() for m in safe] == [[True, False, False, False], [False, True, False, False], [False, False, False, True]]
    print(f"PASS: features, finite silence, output shape, gradient update, disjoint folds; loss={loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "predict", "smoke-test"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache", type=Path, default=BASE / "cache")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--head", choices=["dense", "gap"], default="dense")
    parser.add_argument("--test-fold", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--val-fold", type=int, choices=range(1, 11))
    parser.add_argument("--cross-validation", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = BASE / "outputs" / ("lenet5_gap" if args.head == "gap" else "lenet5")
    if min(args.epochs, args.patience, args.batch_size, args.threads) <= 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("epochs/patience/batch-size/threads/lr must be positive; weight-decay must be nonnegative")
    torch.set_num_threads(args.threads)
    if args.command == "smoke-test":
        smoke_test()
        return
    if args.command in ["evaluate", "predict"] and not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.command == "predict":
        if not args.audio:
            parser.error("--audio is required")
        checkpoint = load_checkpoint(args.checkpoint)
        model = model_from_checkpoint(checkpoint).eval()
        with threadpool_limits(limits=1):
            feature = log_mel(load_audio(args.audio))[None]
        inputs = ((torch.from_numpy(feature) - checkpoint["mean"]) / checkpoint["std"])[:, None]
        with torch.no_grad():
            probs = model(inputs).softmax(1)[0]
        for index in probs.argsort(descending=True)[:3].tolist():
            print(f"{CLASSES[index]}: {probs[index].item():.4f}")
        return
    args.resolved_root = dataset_root(args.data_root)
    rows, x, y, folds = prepare(args.resolved_root, args.cache)
    if args.command == "prepare":
        print(f"Validated {len(rows)} files; feature shape {x.shape}")
    elif args.command == "evaluate":
        checkpoint = load_checkpoint(args.checkpoint)
        digest = hashlib.sha256((args.resolved_root / "metadata/UrbanSound8K.csv").read_bytes()).hexdigest()
        if checkpoint["config"]["metadata_sha256"] != digest or checkpoint["config"]["features"] != FEATURE_CONFIG:
            raise ValueError("Checkpoint metadata/features do not match current data")
        model = model_from_checkpoint(checkpoint).eval()
        mask = folds == checkpoint["config"]["test_fold"]
        batches = loader(x, y, mask, checkpoint["mean"].numpy(), checkpoint["std"].numpy(), args.batch_size)
        loss, acc, truth, predictions = run_epoch(model, batches, "cpu")
        print(json.dumps(dict(accuracy=acc, macro_f1=f1_score(truth, predictions, average="macro"), loss=loss), indent=2))
    else:
        if args.cross_validation and args.val_fold is not None:
            parser.error("Cross-validation uses cyclic validation folds; omit --val-fold")
        results = []
        for test_fold in (range(1, 11) if args.cross_validation else [args.test_fold]):
            val_fold = args.val_fold if args.val_fold is not None else (test_fold - 2) % 10 + 1
            results.append(train(args, rows, x, y, folds, test_fold, val_fold))
        if args.cross_validation:
            write_json(args.output / "cross_validation.json",
                       dict(folds=results, mean_accuracy=float(np.mean([r["accuracy"] for r in results])),
                            std_accuracy=float(np.std([r["accuracy"] for r in results])),
                            mean_macro_f1=float(np.mean([r["macro_f1"] for r in results]))))


if __name__ == "__main__":
    main()
