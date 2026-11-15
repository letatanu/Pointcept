import unittest
import torch
from .quaternion_rope import QuaternionRotary, hamilton
from .model import QuatRoPEPackedHybrid


class RotaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        torch.set_num_threads(2)

    def test_identity_norm_and_gradients(self):
        rope = QuaternionRotary(3, 16)
        feature = torch.randn(7, 3, 16, requires_grad=True)
        position = torch.randn(7, 3, requires_grad=True)
        torch.testing.assert_close(rope(feature, torch.zeros_like(position)), feature)
        output = rope(feature, position)
        torch.testing.assert_close(output.unflatten(-1, (-1, 4)).norm(dim=-1),
                                   feature.unflatten(-1, (-1, 4)).norm(dim=-1))
        (output * torch.randn_like(output)).sum().backward()
        for grad in (feature.grad, position.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)
        # Preserve standard nn.Module.apply used by initialization hooks.
        visited = []
        rope.apply(lambda module: visited.append(module))
        self.assertIn(rope, visited)

    def test_relative_dot_product_and_shared_translation(self):
        rope = QuaternionRotary(2, 16)
        q, k = torch.randn(9, 2, 16), torch.randn(9, 2, 16)
        x, y = torch.randn(9, 3), torch.randn(9, 3)
        actual = (rope(q, x) * rope(k, y)).sum(-1)
        expected = (q * rope(k, y - x)).sum(-1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=4e-6)
        shift = torch.tensor([2., -3., .5])
        shifted = (rope(q, x + shift) * rope(k, y + shift)).sum(-1)
        torch.testing.assert_close(actual, shifted, rtol=1e-5, atol=4e-6)
        rx, ry = rope.rotors(x), rope.rotors(y)
        inverse = torch.cat((rx[..., :1], -rx[..., 1:]), -1)
        torch.testing.assert_close(hamilton(inverse, ry), rope.rotors(y - x), atol=2e-6, rtol=1e-5)

    def test_disabled_rotary_matches_original_model(self):
        try:
            from packed_hybrid_cell_query.model import PackedHybridCellQuery
        except ImportError:
            self.skipTest("Original packed_hybrid_cell_query package unavailable for comparison")
        settings = dict(channels=(16, 32), heads=(2, 4), depths=(1, 1),
            voxel_sizes=(.2, .4), cell_sizes=(.4, .8), query_ratios=(.25, .5),
            max_queries=(2, 3), keys_per_cell=(2, 2), chunk_size=3,
            sparse_backend="reference", attention_backend="reference")
        torch.manual_seed(67)
        baseline = PackedHybridCellQuery(**settings).eval()
        torch.manual_seed(67)
        disabled = QuatRoPEPackedHybrid(**settings, rotary_encoding="none").eval()
        torch.manual_seed(67)
        enabled = QuatRoPEPackedHybrid(**settings, rotary_encoding="quaternion").eval()
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(value, enabled.state_dict()[name], rtol=0, atol=0)
        data = dict(coord=torch.rand(23, 3), feat=torch.randn(23, 4))
        with torch.no_grad():
            a, b, c = baseline(data), disabled(data), enabled(data)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertGreater(float((a - c).abs().max()), 1e-6)

    def test_invalid_dimensions_and_frequency(self):
        with self.assertRaises(ValueError):
            QuaternionRotary(2, 18)
        with self.assertRaises(ValueError):
            QuaternionRotary(2, 16, min_frequency=0)
