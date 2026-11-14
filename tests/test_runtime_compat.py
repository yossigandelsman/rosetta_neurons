"""CPU checks for compatibility required by the supported dependency stack."""

from argparse import Namespace
from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import mae
from torch_utils.ops import grid_sample_gradfix


class GridSampleCompatibilityTests(unittest.TestCase):
    def setUp(self):
        original = grid_sample_gradfix.enabled
        grid_sample_gradfix.enabled = True
        self.addCleanup(setattr, grid_sample_gradfix, "enabled", original)
        torch.manual_seed(0)

    def test_custom_gradients_match_pytorch(self):
        image = torch.randn(1, 2, 4, 5, dtype=torch.double, requires_grad=True)
        grid = (torch.rand(1, 3, 4, 2, dtype=torch.double) * 1.6 - 0.8).requires_grad_()
        actual = grid_sample_gradfix.grid_sample(image, grid)
        expected = F.grid_sample(image, grid, align_corners=False)
        upstream = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(actual, (image, grid), upstream)
        expected_gradients = torch.autograd.grad(expected, (image, grid), upstream)
        torch.testing.assert_close(actual, expected)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_second_gradient_with_fixed_grid(self):
        image = torch.randn(1, 2, 4, 5, dtype=torch.double, requires_grad=True)
        grid = torch.rand(1, 3, 4, 2, dtype=torch.double) * 1.6 - 0.8
        output = grid_sample_gradfix.grid_sample(image, grid)
        upstream = torch.randn_like(output, requires_grad=True)
        first_gradient = torch.autograd.grad(output, image, upstream, create_graph=True)[0]
        probe = torch.randn_like(image)
        actual = torch.autograd.grad(first_gradient, upstream, probe)[0]
        expected = F.grid_sample(probe, grid, align_corners=False)
        torch.testing.assert_close(actual, expected)


class MAECompatibilityTests(unittest.TestCase):
    @staticmethod
    def tiny_model(**kwargs):
        return mae.VisionTransformer(
            img_size=32,
            patch_size=8,
            embed_dim=32,
            depth=1,
            num_heads=4,
            mlp_ratio=2,
            num_classes=5,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )

    def test_small_mae_forward(self):
        sample = torch.randn(2, 3, 32, 32)
        for global_pool in (False, True):
            with self.subTest(global_pool=global_pool):
                model = self.tiny_model(global_pool=global_pool).eval()
                with torch.no_grad():
                    result = model(sample)
                self.assertEqual(tuple(result.shape), (2, 5))
                self.assertTrue(torch.isfinite(result).all())

    def test_load_checkpoint_with_training_arguments(self):
        original = self.tiny_model()
        checkpoint = {"model": original.state_dict(), "args": Namespace(input_size=32)}
        safe_globals_before = list(torch.serialization.get_safe_globals())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mae.pth"
            torch.save(checkpoint, path)
            with patch("mae.vit_base_patch16", side_effect=self.tiny_model):
                restored = mae.load_mae(path)
        self.assertEqual(torch.serialization.get_safe_globals(), safe_globals_before)
        for name, expected in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], expected)


if __name__ == "__main__":
    unittest.main()
