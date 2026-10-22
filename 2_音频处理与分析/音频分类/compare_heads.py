"""Compare saved experiments; never trains or selects models using test scores."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from audio_classification import BASE, load_checkpoint, write_json


def main():
    out = BASE / "outputs/head_comparison"
    out.mkdir(parents=True, exist_ok=True)
    paths = [BASE / "outputs" / name / "test_fold10" for name in ["lenet5", "lenet5_gap"]]
    summaries, histories, checkpoints, manifests = [], [], [], []
    for path in paths:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        history = json.loads((path / "history.json").read_text(encoding="utf-8"))
        selected = next(row for row in history if row["epoch"] == metrics["best_epoch"])
        summaries.append(dict(head=config.get("head", "dense"), parameters=config["parameters"],
                              accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                              test_loss=metrics["test_loss"], best_epoch=metrics["best_epoch"],
                              epochs_run=metrics["epochs_run"], **selected))
        histories.append(history)
        checkpoints.append(load_checkpoint(path / "best.pt"))
        manifests.append((path / "split_manifest.csv").read_bytes())
    # Require identical data membership, normalizers and optimization settings.
    assert manifests[0] == manifests[1], "Split manifests differ"
    for key in ["mean", "std"]:
        torch.testing.assert_close(checkpoints[0][key], checkpoints[1][key], rtol=0, atol=0)
    for key in ["seed", "features", "metadata_sha256", "test_fold", "val_fold", "epochs",
                "patience", "batch_size", "lr", "weight_decay", "optimizer"]:
        assert checkpoints[0]["config"][key] == checkpoints[1]["config"][key], key
    # Both runs start from the same seed: verify the unchanged backbone initializes identically.
    from audio_classification import LeNet5, seed_everything
    seed_everything(42)
    dense = LeNet5("dense")
    seed_everything(42)
    gap = LeNet5("gap")
    for key, value in dense.features.state_dict().items():
        torch.testing.assert_close(value, gap.features.state_dict()[key], rtol=0, atol=0)
    write_json(out / "comparison.json", dict(experiments=summaries,
               split_and_normalization_identical=True, initial_backbone_identical=True,
               accuracy_delta_pp=100 * (summaries[1]["accuracy"] - summaries[0]["accuracy"]),
               macro_f1_delta_pp=100 * (summaries[1]["macro_f1"] - summaries[0]["macro_f1"])))
    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for summary, history, color in zip(summaries, histories, ["tab:blue", "tab:orange"]):
        epochs = [row["epoch"] for row in history]
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            for split, style in [("train", "-"), ("val", "--")]:
                ax.plot(epochs, [row[f"{split}_{metric}"] for row in history],
                        style, color=color, label=f"{summary['head']} {split}")
            ax.set(xlabel="Epoch", ylabel=metric.title())
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / "head_comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summaries, indent=2))
    print("PASS: identical split manifests, normalization, settings and initial backbone")


if __name__ == "__main__":
    main()
