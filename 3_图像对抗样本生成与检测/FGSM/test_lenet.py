import unittest
import torch
from torch import nn
from fgsm_mnist import LeNet5, MNISTCNN, fgsm_attack


class LeNetTests(unittest.TestCase):
    def test_structure_and_shapes(self):
        model = LeNet5()
        self.assertIs(MNISTCNN, LeNet5)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 61706)
        self.assertEqual(tuple(model(torch.rand(3, 1, 28, 28)).shape), (3, 10))
        conv = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
        self.assertEqual([(c.in_channels, c.out_channels, c.kernel_size) for c in conv],
                         [(1, 6, (5, 5)), (6, 16, (5, 5))])
        linear = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
        self.assertEqual([(l.in_features, l.out_features) for l in linear],
                         [(400, 120), (120, 84), (84, 10)])

    def test_gradient_attack_and_weight_roundtrip(self):
        model = LeNet5().eval()
        inputs = torch.rand(2, 1, 28, 28)
        labels = torch.tensor([1, 2])
        adv = fgsm_attack(model, inputs, labels, 0.2)
        self.assertEqual(adv.shape, inputs.shape)
        self.assertGreater((adv - inputs).abs().max().item(), 0)
        self.assertLessEqual((adv - inputs).abs().max().item(), 0.200001)
        other = LeNet5().eval()
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(inputs), other(inputs))


if __name__ == "__main__":
    unittest.main()
