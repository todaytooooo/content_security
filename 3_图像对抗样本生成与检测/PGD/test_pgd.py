"""Algorithmic checks; run python -m unittest -v from this directory."""
import unittest

import torch
from torch import nn

from pgd_mnist import fgsm_attack, pgd_attack


class PGDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(16, 3)).eval()
        self.images = torch.rand(6, 1, 4, 4)
        self.labels = torch.tensor([0, 1, 2, 0, 1, 2])

    def test_zero_budget_is_identity(self):
        result = pgd_attack(self.model, self.images, self.labels, 0)
        self.assertTrue(torch.equal(result, self.images))

    def test_projection_and_model_unchanged(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        original = self.images.clone()
        # Deliberately oversize step: projection must still enforce the budget.
        result = pgd_attack(self.model, self.images, self.labels, 0.1, alpha=0.5, steps=5)
        self.assertLessEqual((result - original).abs().max().item(), 0.100001)
        self.assertGreaterEqual(result.min().item(), 0)
        self.assertLessEqual(result.max().item(), 1)
        self.assertFalse(result.requires_grad)
        self.assertTrue(torch.equal(original, self.images))
        for previous, parameter in zip(before, self.model.parameters()):
            self.assertTrue(torch.equal(previous, parameter))
            self.assertIsNone(parameter.grad)

    def test_one_step_without_random_start_matches_fgsm(self):
        result = pgd_attack(self.model, self.images, self.labels, 0.2, alpha=0.2, steps=1, random_start=False)
        expected = fgsm_attack(self.model, self.images, self.labels, 0.2)
        torch.testing.assert_close(result, expected)

    def test_invalid_arguments(self):
        for options in ({"epsilon": -0.1}, {"epsilon": float("nan")},
                        {"epsilon": 0.1, "alpha": 0}, {"epsilon": 0.1, "steps": 0}):
            with self.assertRaises(ValueError):
                pgd_attack(self.model, self.images, self.labels, **options)


if __name__ == "__main__":
    unittest.main()
