import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from adversarial_training import generate_training_adversaries


class AugmentationTests(unittest.TestCase):
    def test_generated_data_keeps_labels_order_budget_and_model(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Flatten(), nn.Linear(16, 3)).eval()
        images = torch.rand(7, 1, 4, 4)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
        parameters = [p.detach().clone() for p in model.parameters()]
        loader = DataLoader(TensorDataset(images, labels), batch_size=3, shuffle=False)
        dataset = generate_training_adversaries(model, loader, torch.device("cpu"), 0.2)
        adversarial, actual_labels = dataset.tensors
        self.assertEqual(len(dataset), len(images))
        self.assertTrue(torch.equal(labels, actual_labels))
        self.assertLessEqual((adversarial - images).abs().max().item(), 0.200001)
        self.assertFalse(adversarial.requires_grad)
        self.assertGreaterEqual(adversarial.min().item(), 0)
        self.assertLessEqual(adversarial.max().item(), 1)
        for old, current in zip(parameters, model.parameters()):
            self.assertTrue(torch.equal(old, current))
            self.assertIsNone(current.grad)


if __name__ == "__main__":
    unittest.main()
