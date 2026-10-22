"""Audit and summarize the three fixed MFCC CNN runs; never select a test winner."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


BASE = Path(__file__).resolve().parent
RUNS = [(42, "mfcc_cnn"), (7, "mfcc_cnn_seed7"), (2026, "mfcc_cnn_seed2026")]
SETTINGS = ["model_type", "features", "feature_type", "classes", "test_fold",
            "val_fold", "train_folds", "n_train", "n_val", "n_test", "split_protocol",
            "n_purged", "epochs", "patience", "batch_size", "lr", "weight_decay",
            "optimizer", "parameters", "metadata_sha256", "environment"]


def summarize():
    records = []
    reference = None
    split_hash = None
    for seed, name in RUNS:
        run = BASE / "outputs" / name / "test_fold10"
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        settings = {key: config[key] for key in SETTINGS}
        digest = hashlib.sha256((run / "split_manifest.csv").read_bytes()).hexdigest()
        if config["seed"] != seed:
            raise ValueError(f"Unexpected seed in {run}")
        if reference is None:
            reference, split_hash = settings, digest
        elif settings != reference or digest != split_hash:
            raise ValueError(f"Different protocol or split in {run}")
        records.append(dict(seed=seed, accuracy=metrics["accuracy"],
                            macro_f1=metrics["macro_f1"], test_loss=metrics["test_loss"],
                            best_epoch=metrics["best_epoch"], epochs_run=metrics["epochs_run"],
                            output=str(run.relative_to(BASE))))
    summary = dict(protocol="Fixed MFCC CNN; fold9 validation, fold10 test; source-purged",
                   n_runs=len(records), std_definition="sample standard deviation (ddof=1)",
                   split_manifest_sha256=split_hash, settings=reference, runs=records)
    for metric in ("accuracy", "macro_f1"):
        values = [row[metric] for row in records]
        summary[metric] = dict(mean=mean(values), std=stdev(values),
                               minimum=min(values), maximum=max(values))
    out = BASE / "outputs" / "mfcc_seed_validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    with (out / "runs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({key: summary[key] for key in ("n_runs", "accuracy", "macro_f1")}, indent=2))
    print(f"Matching split manifests and settings verified. Summary: {out}")


if __name__ == "__main__":
    summarize()
