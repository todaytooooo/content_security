"""Experiment 3: fixed FGSM augmentation, clean fine-tuning control, adaptive evaluation."""

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
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT / "FGSM"))
sys.path.insert(0, str(ROOT / "PGD"))
from fgsm_mnist import MNISTCNN, accuracy, fgsm_attack, train
from pgd_mnist import evaluate


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_training_adversaries(model, loader, device, epsilon):
    """Only pass the TRAINING split here; keep original labels, never predictions."""
    model.eval()
    images_out, labels_out = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adversarial = fgsm_attack(model, images, labels, epsilon)
        if ((adversarial - images).abs().max().item() > epsilon + 1e-6
                or not torch.isfinite(adversarial).all().item()
                or adversarial.min().item() < 0 or adversarial.max().item() > 1):
            raise RuntimeError("Invalid adversarial training sample")
        images_out.append(adversarial.cpu())
        labels_out.append(labels.cpu())
    return TensorDataset(torch.cat(images_out), torch.cat(labels_out))


def fresh_model(state, device):
    model = MNISTCNN().to(device)
    model.load_state_dict(state)
    return model


def save_comparison(rows, output):
    names = list(dict.fromkeys(row["model"] for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    clean = [next(r["clean_accuracy"] for r in rows if r["model"] == name) for name in names]
    axes[0].bar(range(len(names)), clean)
    axes[0].set_xticks(range(len(names)), names, rotation=15)
    axes[0].set(title="Clean MNIST accuracy", ylabel="Accuracy", ylim=(0, 1.02))
    for ax, attack in zip(axes[1:], ("FGSM", "PGD")):
        for name in names:
            selected = [r for r in rows if r["model"] == name and r["attack"] == attack]
            ax.plot([r["epsilon"] for r in selected], [r["adversarial_accuracy"] for r in selected], "o-", label=name)
        ax.set(title=f"Adaptive {attack} attack", xlabel="Epsilon", ylabel="Accuracy", ylim=(0, 1.02))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "defense_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "FGSM" / "data")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "FGSM" / "outputs" / "mnist_cnn.pt")
    parser.add_argument("--output-dir", type=Path, default=BASE / "outputs")
    parser.add_argument("--train-epsilon", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128, help="Number of clean items per training batch; mixed batch size is twice this")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0, 0.1, 0.2, 0.3])
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-adversaries", action="store_true", help="Save the fixed training attack tensors (~188 MB for full MNIST)")
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.pgd_steps) < 1 or min(args.train_limit, args.test_limit) < 0
            or any(not math.isfinite(e) or not 0 <= e <= 1 for e in [args.train_epsilon] + args.epsilons)
            or any(not math.isfinite(v) or v <= 0 for v in (args.lr, args.alpha))):
        parser.error("Invalid epochs, batch size, limits, epsilon, learning rate or PGD parameters")
    if not args.checkpoint.is_file():
        parser.error(f"Missing baseline checkpoint: {args.checkpoint}")
    seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    testing = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    if args.train_limit:
        training = Subset(training, range(min(args.train_limit, len(training))))
    if args.test_limit:
        testing = Subset(testing, range(min(args.test_limit, len(testing))))
    test_loader = DataLoader(testing, batch_size=args.batch_size, shuffle=False, num_workers=0)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    baseline = fresh_model(state, device)
    print(f"Device={device}; train={len(training)}; test={len(testing)}", flush=True)
    print("Generating fixed FGSM examples ONLY from training split...", flush=True)
    generation_loader = DataLoader(training, batch_size=args.batch_size, shuffle=False, num_workers=0)
    adversaries = generate_training_adversaries(baseline, generation_loader, device, args.train_epsilon)
    if args.save_adversaries:
        torch.save({"images": adversaries.tensors[0], "labels": adversaries.tensors[1],
                    "epsilon": args.train_epsilon, "split": "train"}, args.output_dir / "training_adversaries.pt")
    # One clean AND its fixed adversarial example per item, preserving an exact 1:1 mix.
    class PairedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(training)

        def __getitem__(self, index):
            image, label = training[index]
            adv_image, adv_label = adversaries[index]
            if label != adv_label.item():
                raise RuntimeError("Training labels do not align")
            return torch.stack((image, adv_image)), torch.tensor([label, label])

    def flatten_pairs(batch):
        images, labels = torch.utils.data.default_collate(batch)
        return images.flatten(0, 1), labels.flatten()

    histories = {}
    # Clean control duplicates each clean image: same batch size, updates and training exposure.
    # It separates gains due to additional fine-tuning from gains due to adversarial augmentation.
    clean_control = fresh_model(state, device)
    augmented = fresh_model(state, device)
    models = {"Baseline": baseline, "CleanControl": clean_control, "FGSMAugmented": augmented}
    clean_pair_dataset = ConcatDataset([training, training])
    for name, model in (("CleanControl", clean_control), ("FGSMAugmented", augmented)):
        seed_all(args.seed)
        if name == "CleanControl":
            loader = DataLoader(clean_pair_dataset, batch_size=2 * args.batch_size, shuffle=True,
                                num_workers=0, generator=torch.Generator().manual_seed(args.seed))
        else:
            loader = DataLoader(PairedDataset(), batch_size=args.batch_size, shuffle=True, num_workers=0,
                                collate_fn=flatten_pairs, generator=torch.Generator().manual_seed(args.seed))
        print(f"Fine-tuning {name}...", flush=True)
        histories[name] = train(model, loader, device, args.epochs, args.lr)
        torch.save(model.state_dict(), args.output_dir / f"{name}.pt")
        print(f"{name} clean accuracy={accuracy(model, test_loader, device):.4%}", flush=True)
    del adversaries
    rows = []
    for name, model in models.items():
        print(f"Adaptive evaluation: {name}", flush=True)
        seed_all(args.seed)  # Same randomized PGD initial offsets for each model.
        evaluated, _ = evaluate(model, test_loader, device, sorted(set([0.0] + args.epsilons)),
                                args.alpha, args.pgd_steps, True, 0)
        for row in evaluated:
            rows.append({"model": name, **row})
        # Save incrementally, so completed model evaluations survive interruption.
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        report = {"config": config, "device": str(device), "torch_version": str(torch.__version__),
                  "train_samples": len(training), "adversarial_train_samples": len(training),
                  "test_samples": len(testing), "augmentation": "fixed FGSM from baseline; train split only",
                  "training_history": histories, "metrics": rows}
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_comparison(rows, args.output_dir)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
