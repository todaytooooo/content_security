import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from online_rs_fgsm import rs_fgsm_attack, robust_metrics, split_indices


class OnlineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(16, 3)).eval()
        self.x = torch.rand(8, 1, 4, 4)
        self.y = torch.arange(8) % 3

    def test_split_disjoint_complete_reproducible(self):
        training, validation = split_indices(100, 20, 42)
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(set(training) | set(validation), set(range(100)))
        self.assertEqual((training, validation), split_indices(100, 20, 42))

    def test_rs_constraints_source_unchanged_and_random_start_matters(self):
        source = self.x.clone()
        state = [p.detach().clone() for p in self.model.parameters()]
        torch.manual_seed(1)
        first = rs_fgsm_attack(self.model, self.x, self.y, 0.2, 0.25)
        torch.manual_seed(2)
        second = rs_fgsm_attack(self.model, self.x, self.y, 0.2, 0.25)
        self.assertFalse(torch.equal(first, second))
        self.assertLessEqual((first - source).abs().max().item(), 0.200001)
        self.assertGreaterEqual(first.min().item(), 0)
        self.assertLessEqual(first.max().item(), 1)
        self.assertTrue(torch.equal(source, self.x))
        self.assertFalse(first.requires_grad)
        for before, after in zip(state, self.model.parameters()):
            self.assertTrue(torch.equal(before, after))
            self.assertIsNone(after.grad)

    def test_monitor_preserves_rng_and_zero_budget(self):
        loader = DataLoader(TensorDataset(self.x, self.y), batch_size=4)
        rng = torch.get_rng_state().clone()
        result = robust_metrics(self.model, loader, torch.device("cpu"), 0, 2, 0.01, 2, 123)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(result["clean_accuracy"], result["pgd_accuracy"])
        self.assertEqual(result["clean_accuracy"], result["fgsm_accuracy"])


if __name__ == "__main__":
    unittest.main()
