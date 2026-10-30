"""Leak-free split, fixed FGSM vs online RS-FGSM, epoch-wise PGD validation."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from adversarial_training import (ROOT, BASE, MNISTCNN, fresh_model, seed_all,
                                  generate_training_adversaries, fgsm_attack)
from pgd_mnist import pgd_attack


def split_indices(size, validation_size, seed):
    if not 0 < validation_size < size:
        raise ValueError("Validation size must lie between zero and dataset size")
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    return indices[validation_size:].tolist(), indices[:validation_size].tolist()


def rs_fgsm_attack(model, images, labels, epsilon, alpha):
    """Uniform random start followed by ONE gradient step, projected about original."""
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid epsilon or alpha")
    original = images.detach()
    if epsilon == 0:
        return original.clone()
    with torch.enable_grad():
        start = (original + torch.empty_like(original).uniform_(-epsilon, epsilon)).clamp(0, 1)
        start.requires_grad_(True)
        gradient, = torch.autograd.grad(F.cross_entropy(model(start), labels), start)
    result = start.detach() + alpha * gradient.sign()
    lower, upper = (original - epsilon).clamp(0, 1), (original + epsilon).clamp(0, 1)
    return torch.maximum(torch.minimum(result, upper), lower).detach()


def robust_metrics(model, loader, device, epsilon, steps, alpha, restarts, seed):
    """PGD survivor accuracy: must remain correct at every restart's final iterate.

    Includes the clean input as a candidate; finite attacks are empirical evidence,
    not a robustness certificate. fork_rng keeps monitoring out of training RNG.
    """
    model.eval()
    total = clean_count = fgsm_count = pgd_count = 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                clean = model(images).argmax(1).eq(labels)
            adversarial = fgsm_attack(model, images, labels, epsilon)
            with torch.no_grad():
                fgsm_correct = model(adversarial).argmax(1).eq(labels)
            survived = clean.clone()
            for _ in range(restarts):
                attack = pgd_attack(model, images, labels, epsilon, alpha, steps, True)
                if ((attack - images).abs().max().item() > epsilon + 1e-6
                        or not torch.isfinite(attack).all().item()
                        or attack.min().item() < 0 or attack.max().item() > 1):
                    raise RuntimeError("Invalid PGD evaluation samples")
                with torch.no_grad():
                    survived &= model(attack).argmax(1).eq(labels)
            total += len(labels)
            clean_count += clean.sum().item()
            fgsm_count += fgsm_correct.sum().item()
            pgd_count += survived.sum().item()
    return {"samples": total, "epsilon": epsilon, "clean_accuracy": clean_count / total,
            "fgsm_accuracy": fgsm_count / total, "pgd_accuracy": pgd_count / total,
            "pgd_success_rate": 1 - pgd_count / clean_count if clean_count else None}


def train_epoch(model, loader, device, optimizer, mode, epsilon, alpha):
    loss_sum = correct = total = 0
    for images, labels, fixed in loader:
        images, labels = images.to(device), labels.to(device)
        if mode == "OnlineRSFGSM":
            model.eval()
            adversarial = rs_fgsm_attack(model, images, labels, epsilon, alpha)
        elif mode == "FixedFGSM":
            adversarial = fixed.to(device)
        else:
            adversarial = images
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # All branches have the same batch size, clean/adversarial weights and updates.
        inputs = torch.cat((images, adversarial))
        targets = torch.cat((labels, labels))
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        total += len(targets)
        loss_sum += loss.item() * len(targets)
        correct += logits.argmax(1).eq(targets).sum().item()
    return {"train_loss": loss_sum / total, "train_accuracy": correct / total}


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "FGSM" / "data")
    parser.add_argument("--output-dir", type=Path, default=BASE / "outputs_online")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rs-alpha", type=float, default=0.25, help="Default is 1.25 times default epsilon")
    parser.add_argument("--pgd-alpha", type=float, default=0.01)
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--val-restarts", type=int, default=1)
    parser.add_argument("--test-restarts", type=int, default=3)
    parser.add_argument("--test-epsilons", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if (min(args.baseline_epochs, args.epochs, args.batch_size, args.pgd_steps, args.val_restarts, args.test_restarts) < 1
            or min(args.train_limit, args.test_limit) < 0
            or any(not math.isfinite(e) or not 0 <= e <= 1 for e in [args.epsilon] + args.test_epsilons)
            or any(not math.isfinite(v) or v <= 0 for v in (args.lr, args.rs_alpha, args.pgd_alpha))):
        parser.error("Invalid training/attack parameters")
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    dump_json(output / "config.json", {**config, "actual_device": str(device), "torch_version": str(torch.__version__)})
    full = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    try:
        train_indices, val_indices = split_indices(len(full), args.validation_size, args.seed)
    except ValueError as error:
        parser.error(str(error))
    if args.train_limit:
        train_indices = train_indices[:args.train_limit]
    dump_json(output / "split_indices.json", {"train": train_indices, "validation": val_indices})
    train_set, val_set = Subset(full, train_indices), Subset(full, val_indices)
    # Materialize only the TRAIN split for efficient paired training and augmentation.
    images, labels = [], []
    for x, y in DataLoader(train_set, batch_size=1024, shuffle=False):
        images.append(x)
        labels.append(y)
    images, labels = torch.cat(images), torch.cat(labels)
    clean_data = TensorDataset(images, labels, images)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    def training_loader(data):
        return DataLoader(data, batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(args.seed))
    print(f"Device={device}; independent train={len(train_set)}, validation={len(val_set)}", flush=True)
    baseline = MNISTCNN().to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    baseline_history = []
    for epoch in range(1, args.baseline_epochs + 1):
        metrics = train_epoch(baseline, training_loader(clean_data), device, optimizer, "Clean", args.epsilon, args.rs_alpha)
        baseline_history.append({"epoch": epoch, **metrics})
        print(f"Baseline epoch={epoch} loss={metrics['train_loss']:.4f}", flush=True)
    torch.save(baseline.state_dict(), output / "Baseline.pt")
    state = {key: value.detach().cpu().clone() for key, value in baseline.state_dict().items()}
    baseline_val = robust_metrics(baseline, val_loader, device, args.epsilon, args.pgd_steps,
                                  args.pgd_alpha, args.val_restarts, args.seed + 100)
    print(f"Baseline validation: {baseline_val}", flush=True)
    generation_loader = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size)
    fixed = generate_training_adversaries(baseline, generation_loader, device, args.epsilon).tensors[0]
    history, selections = [], {}
    for mode in ("CleanControl", "FixedFGSM", "OnlineRSFGSM"):
        seed_all(args.seed)
        model = fresh_model(state, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loader = training_loader(TensorDataset(images, labels, fixed if mode == "FixedFGSM" else images))
        # Epoch zero is a candidate, preventing selection of a strictly worse defense.
        best_score, best_epoch = baseline_val["pgd_accuracy"], 0
        torch.save(model.state_dict(), output / f"{mode}_best.pt")
        history.append({"model": mode, "epoch": 0, "train_loss": None, "train_accuracy": None, **baseline_val})
        for epoch in range(1, args.epochs + 1):
            train_result = train_epoch(model, loader, device, optimizer, mode, args.epsilon, args.rs_alpha)
            validation = robust_metrics(model, val_loader, device, args.epsilon, args.pgd_steps,
                                        args.pgd_alpha, args.val_restarts, args.seed + 100)
            row = {"model": mode, "epoch": epoch, **train_result, **validation}
            history.append(row)
            print(json.dumps(row), flush=True)
            if validation["pgd_accuracy"] > best_score:
                best_score, best_epoch = validation["pgd_accuracy"], epoch
                torch.save(model.state_dict(), output / f"{mode}_best.pt")
            torch.save(model.state_dict(), output / f"{mode}_last.pt")
            dump_json(output / "validation_history.json", history)
        selections[mode] = {"epoch": best_epoch, "validation_pgd_accuracy": best_score}
        dump_json(output / "selection.json", selections)
    # The official test set is first evaluated AFTER all epoch selections are frozen.
    testing = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    if args.test_limit:
        testing = Subset(testing, range(min(args.test_limit, len(testing))))
    test_loader = DataLoader(testing, batch_size=args.batch_size, shuffle=False)
    results = []
    checkpoints = {"Baseline": output / "Baseline.pt"}
    for mode in selections:
        checkpoints[f"{mode}_best"] = output / f"{mode}_best.pt"
        checkpoints[f"{mode}_last"] = output / f"{mode}_last.pt"
    cache = {}
    for name, checkpoint in checkpoints.items():
        mode = name.removesuffix("_best").removesuffix("_last")
        epoch = 0 if name == "Baseline" else selections[mode]["epoch"] if name.endswith("_best") else args.epochs
        # Best epoch 0 has exactly baseline weights. Avoid repeating expensive evaluation.
        cache_key = ("Baseline", 0) if epoch == 0 else (mode, epoch)
        if cache_key not in cache:
            model = fresh_model(torch.load(checkpoint, map_location="cpu", weights_only=True), device)
            measured = []
            for epsilon in sorted(set(args.test_epsilons)):
                metrics = robust_metrics(model, test_loader, device, epsilon, args.pgd_steps,
                                         args.pgd_alpha, args.test_restarts, args.seed + 200)
                measured.append(metrics)
                print(f"TEST {name}: {metrics}", flush=True)
            cache[cache_key] = measured
        results.extend({"model": name, "selected_epoch": epoch, **row} for row in cache[cache_key])
        dump_json(output / "results.json", {"config": config, "selection": selections,
                  "baseline_training": baseline_history, "validation_history": history, "test_metrics": results})
    for filename, rows in (("validation_history.csv", history), ("test_metrics.csv", results)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode in selections:
        selected = [row for row in history if row["model"] == mode]
        for ax, key in zip(axes, ("clean_accuracy", "pgd_accuracy")):
            ax.plot([r["epoch"] for r in selected], [r[key] for r in selected], "o-", label=mode)
    for ax, title in zip(axes, ("Independent validation: clean", f"Independent validation: PGD epsilon={args.epsilon}")):
        ax.set(title=title, xlabel="Fine-tuning epoch", ylabel="Accuracy", ylim=(0, 1.02))
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", dpi=160)
    plt.close(fig)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
