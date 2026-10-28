"""Train a MNIST CNN and evaluate untargeted, white-box FGSM attacks."""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


BASE = Path(__file__).resolve().parent


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def fgsm_attack(model, images, labels, epsilon):
    """epsilon is measured in raw [0, 1] pixel units; model must be in eval mode."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0, 1]")
    with torch.enable_grad():
        inputs = images.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(model(inputs), labels)
        gradient, = torch.autograd.grad(loss, inputs)
    return (inputs.detach() + epsilon * gradient.sign()).clamp(0, 1).detach()


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += model(images).argmax(1).eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def train(model, loader, device, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total += labels.numel()
            loss_sum += loss.item() * labels.numel()
            correct += logits.argmax(1).eq(labels).sum().item()
        row = {"epoch": epoch, "loss": loss_sum / total, "train_accuracy": correct / total}
        history.append(row)
        print(f"Epoch {epoch}/{epochs} loss={row['loss']:.4f} train_acc={row['train_accuracy']:.4%}", flush=True)
    return history


def evaluate_attacks(model, loader, device, epsilons, example_count):
    model.eval()
    stats = {eps: {"correct": 0, "successes": 0, "max_linf": 0.0} for eps in epsilons}
    examples = {eps: [] for eps in epsilons}
    total = clean_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean_predictions = model(images).argmax(1)
        initially_correct = clean_predictions.eq(labels)
        total += labels.numel()
        clean_correct += initially_correct.sum().item()
        # Attack all test images. Reuse the same clean-input gradient for each epsilon.
        with torch.enable_grad():
            inputs = images.detach().clone().requires_grad_(True)
            gradient, = torch.autograd.grad(F.cross_entropy(model(inputs), labels), inputs)
        for eps in epsilons:
            adversarial = (images + eps * gradient.sign()).clamp(0, 1).detach()
            linf = (adversarial - images).abs().flatten(1).amax(1).max().item()
            if linf > eps + 1e-6 or not torch.isfinite(adversarial).all().item():
                raise RuntimeError("FGSM perturbation constraint violated")
            if eps == 0 and not torch.equal(adversarial, images):
                raise RuntimeError("epsilon=0 must preserve the original inputs")
            with torch.no_grad():
                predictions = model(adversarial).argmax(1)
            successful = initially_correct & predictions.ne(labels)
            stats[eps]["correct"] += predictions.eq(labels).sum().item()
            stats[eps]["successes"] += successful.sum().item()
            stats[eps]["max_linf"] = max(stats[eps]["max_linf"], linf)
            # Show successful attacks where available; the zero-budget row shows clean examples.
            eligible = initially_correct if eps == 0 else successful
            for index in eligible.nonzero(as_tuple=True)[0].tolist():
                if len(examples[eps]) >= example_count:
                    break
                examples[eps].append((images[index, 0].cpu(), adversarial[index, 0].cpu(),
                                      labels[index].item(), predictions[index].item()))
    rows = []
    for eps in epsilons:
        state = stats[eps]
        row = {"epsilon": eps, "test_samples": total, "clean_correct": clean_correct,
               "clean_accuracy": clean_correct / total,
               "adversarial_accuracy": state["correct"] / total,
               "attack_successes": state["successes"],
               "attack_success_rate": state["successes"] / clean_correct if clean_correct else None,
               "max_linf": state["max_linf"]}
        if eps == 0 and state["correct"] != clean_correct:
            raise RuntimeError("epsilon=0 accuracy differs from clean accuracy")
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows, examples


def save_figures(rows, examples, output):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["epsilon"] for r in rows], [r["adversarial_accuracy"] for r in rows], "o-")
    ax.set(xlabel="Epsilon (pixel range [0, 1])", ylabel="Test accuracy", ylim=(0, 1.02),
           title="MNIST: untargeted FGSM")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_epsilon.png", dpi=160)
    plt.close(fig)
    for eps, records in examples.items():
        if not records:
            continue
        fig, axes = plt.subplots(3, len(records), figsize=(2.4 * len(records), 6), squeeze=False)
        for col, (original, adversarial, label, prediction) in enumerate(records):
            axes[0, col].imshow(original, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(f"Original: {label}")
            axes[1, col].imshow(adversarial - original, cmap="seismic", vmin=-max(eps, 1e-6), vmax=max(eps, 1e-6))
            axes[1, col].set_title("Perturbation (scaled)")
            axes[2, col].imshow(adversarial, cmap="gray", vmin=0, vmax=1)
            axes[2, col].set_title(f"Prediction: {prediction}")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"FGSM epsilon={eps:g}")
        fig.tight_layout()
        fig.savefig(output / f"examples_eps_{eps:g}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--output-dir", type=Path, default=BASE / "outputs")
    parser.add_argument("--checkpoint", type=Path, help="Load weights and skip training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    parser.add_argument("--train-limit", type=int, default=0, help="0 uses the full training set")
    parser.add_argument("--test-limit", type=int, default=0, help="0 uses the full test set")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.examples < 0:
        parser.error("epochs, batch-size, lr must be positive; examples must be nonnegative")
    if min(args.train_limit, args.test_limit) < 0 or any(not 0 <= e <= 1 for e in args.epsilons):
        parser.error("dataset limits must be nonnegative and epsilon must lie in [0, 1]")
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
    transform = transforms.ToTensor()  # Raw [0,1] pixels: epsilon has an unambiguous scale.
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    if args.test_limit:
        test_set = Subset(test_set, range(min(args.test_limit, len(test_set))))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = MNISTCNN().to(device)
    history = []
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    else:
        train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        if args.train_limit:
            train_set = Subset(train_set, range(min(args.train_limit, len(train_set))))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  generator=torch.Generator().manual_seed(args.seed))
        history = train(model, train_loader, device, args.epochs, args.lr)
        torch.save(model.state_dict(), args.output_dir / "mnist_cnn.pt")
    print(f"Device: {device}; clean test accuracy: {accuracy(model, test_loader, device):.4%}", flush=True)
    rows, examples = evaluate_attacks(model, test_loader, device, sorted(set([0.0] + args.epsilons)), args.examples)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {"config": config, "device": str(device), "torch_version": str(torch.__version__),
              "training_history": history, "metrics": rows}
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_figures(rows, examples, args.output_dir)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
