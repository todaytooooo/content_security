"""Run all LeNet experiments sequentially, stopping on failure; retain old results."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    jobs = [("FGSM", "fgsm_mnist.py"), ("PGD", "pgd_mnist.py"),
            ("AdversarialTraining", "adversarial_training.py"),
            ("AdversarialTraining", "online_rs_fgsm.py")]
    for directory, script in jobs:
        print(f"\n=== LeNet: {directory}/{script} ===", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / directory / script)],
                       cwd=ROOT / directory, check=True)
    print("All four LeNet experiments completed.", flush=True)


if __name__ == "__main__":
    main()
