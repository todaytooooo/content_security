"""MNIST white-box untargeted L-infinity projected gradient descent attack."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

BASE = Path(__file__).resolve().parent
FGSM_DIR = BASE.parent / "FGSM"
# Reuse the established classifier and training code, so checkpoints are identical.
sys.path.insert(0, str(FGSM_DIR))
from fgsm_mnist import MNISTCNN, fgsm_attack, train


def pgd_attack(model, images, labels, epsilon, alpha=0.01, steps=40, random_start=True):
    """Return the final iterate of standard PGD; inputs must be raw [0,1] pixels.

    Call model.eval() before attacking. No model weights or parameter gradients
    are updated. Projection is relative to the ORIGINAL images at every step.
    """
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if not math.isfinite(alpha) or alpha <= 0 or steps < 1:
        raise ValueError("alpha must be finite and positive; steps must be >= 1")
    original = images.detach()
    if epsilon == 0:
        return original.clone()
    lower = (original - epsilon).clamp(0, 1)
    upper = (original + epsilon).clamp(0, 1)
    adversarial = original.clone()
    if random_start:
        adversarial = (original + torch.empty_like(original).uniform_(-epsilon, epsilon)).clamp(0, 1)
    for _ in range(steps):
        with torch.enable_grad():
            adversarial = adversarial.detach().requires_grad_(True)
            loss = F.cross_entropy(model(adversarial), labels)
            gradient, = torch.autograd.grad(loss, adversarial)
        adversarial = adversarial.detach() + alpha * gradient.sign()
        adversarial = torch.maximum(torch.minimum(adversarial, upper), lower).detach()
    return adversarial


def evaluate(model, loader, device, epsilons, alpha, steps, random_start, example_count):
    model.eval()
    rows, examples = [], {}
    for epsilon in epsilons:
        total = clean_correct = 0
        counts = {name: {"correct": 0, "successes": 0, "max_linf": 0.0} for name in ("FGSM", "PGD")}
        examples[epsilon] = []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                initially_correct = model(images).argmax(1).eq(labels)
            total += labels.numel()
            clean_correct += initially_correct.sum().item()
            attacks = {
                "FGSM": fgsm_attack(model, images, labels, epsilon),
                "PGD": pgd_attack(model, images, labels, epsilon, alpha, steps, random_start),
            }
            for name, adversarial in attacks.items():
                linf = (adversarial - images).abs().flatten(1).amax().item()
                if (not torch.isfinite(adversarial).all().item() or linf > epsilon + 1e-6
                        or adversarial.min().item() < 0 or adversarial.max().item() > 1):
                    raise RuntimeError(f"{name} violates image/perturbation constraints")
                if epsilon == 0 and not torch.equal(adversarial, images):
                    raise RuntimeError("epsilon=0 must preserve inputs")
                with torch.no_grad():
                    predictions = model(adversarial).argmax(1)
                successful = initially_correct & predictions.ne(labels)
                counts[name]["correct"] += predictions.eq(labels).sum().item()
                counts[name]["successes"] += successful.sum().item()
                counts[name]["max_linf"] = max(counts[name]["max_linf"], linf)
                if name == "PGD":
                    eligible = initially_correct if epsilon == 0 else successful
                    for index in eligible.nonzero(as_tuple=True)[0].tolist():
                        if len(examples[epsilon]) >= example_count:
                            break
                        examples[epsilon].append((images[index, 0].cpu(), adversarial[index, 0].cpu(),
                                                  labels[index].item(), predictions[index].item()))
        for name, counts_for_attack in counts.items():
            if epsilon == 0 and counts_for_attack["correct"] != clean_correct:
                raise RuntimeError("epsilon=0 accuracy must equal clean accuracy")
            row = {"attack": name, "epsilon": epsilon, "test_samples": total,
                   "clean_correct": clean_correct, "clean_accuracy": clean_correct / total,
                   "adversarial_accuracy": counts_for_attack["correct"] / total,
                   "attack_successes": counts_for_attack["successes"],
                   "attack_success_rate": counts_for_attack["successes"] / clean_correct if clean_correct else None,
                   "max_linf": counts_for_attack["max_linf"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows, examples


def save_plots(rows, examples, output):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("FGSM", "PGD"):
        selected = [row for row in rows if row["attack"] == name]
        ax.plot([r["epsilon"] for r in selected], [r["adversarial_accuracy"] for r in selected], "o-", label=name)
    ax.set(xlabel="Epsilon (raw pixel range [0,1])", ylabel="Test accuracy", ylim=(0, 1.02),
           title="MNIST: FGSM vs PGD (same model and test samples)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "accuracy_comparison.png", dpi=160)
    plt.close(fig)
    for epsilon, records in examples.items():
        if not records:
            continue
        fig, axes = plt.subplots(3, len(records), figsize=(2.4 * len(records), 6), squeeze=False)
        for col, (original, adversarial, label, prediction) in enumerate(records):
            axes[0, col].imshow(original, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(f"Original: {label}")
            axes[1, col].imshow(adversarial - original, cmap="seismic", vmin=-max(epsilon, 1e-6), vmax=max(epsilon, 1e-6))
            axes[1, col].set_title("Perturbation (scaled)")
            axes[2, col].imshow(adversarial, cmap="gray", vmin=0, vmax=1)
            axes[2, col].set_title(f"Prediction: {prediction}")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"PGD epsilon={epsilon:g}")
        fig.tight_layout()
        fig.savefig(output / f"examples_eps_{epsilon:g}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=FGSM_DIR / "data")
    parser.add_argument("--checkpoint", type=Path, default=FGSM_DIR / "outputs" / "mnist_cnn.pt")
    parser.add_argument("--train", action="store_true", help="Train a new model instead of loading a checkpoint")
    parser.add_argument("--output-dir", type=Path, default=BASE / "outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epsilon", "--epsilons", dest="epsilons", type=float, nargs="+", default=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.steps) <= 0 or min(args.test_limit, args.train_limit, args.examples) < 0
            or not math.isfinite(args.alpha) or args.alpha <= 0 or not math.isfinite(args.lr) or args.lr <= 0
            or any(not math.isfinite(e) or not 0 <= e <= 1 for e in args.epsilons)):
        parser.error("Invalid parameters: use positive epochs/batch-size/steps/alpha/lr, nonnegative limits, epsilon in [0,1]")
    if not args.train and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}; use --checkpoint PATH or --train")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = MNISTCNN().to(device)
    history = []
    if args.train:
        training = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        if args.train_limit:
            training = Subset(training, range(min(args.train_limit, len(training))))
        train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  generator=torch.Generator().manual_seed(args.seed))
        history = train(model, train_loader, device, args.epochs, args.lr)
        torch.save(model.state_dict(), args.output_dir / "mnist_cnn.pt")
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    testing = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    if args.test_limit:
        testing = Subset(testing, range(min(args.test_limit, len(testing))))
    loader = DataLoader(testing, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Device: {device}; test samples: {len(testing)}; PGD steps: {args.steps}", flush=True)
    rows, examples = evaluate(model, loader, device, sorted(set([0.0] + args.epsilons)), args.alpha,
                              args.steps, not args.no_random_start, args.examples)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {"config": config, "device": str(device), "torch_version": str(torch.__version__),
              "training_history": history, "metrics": rows}
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_plots(rows, examples, args.output_dir)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
