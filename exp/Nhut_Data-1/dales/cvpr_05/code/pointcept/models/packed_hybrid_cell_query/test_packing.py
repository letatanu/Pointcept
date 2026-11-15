import unittest
from unittest.mock import patch
import torch
from .geometry import make_voxels
from .sampling import cell_order, neighboring_cells
from .packed_attention import PackedNeighborAttention, packed_attention


class PackingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)

    def test_packed_matches_masked_reference_and_gradients(self):
        lengths_q, lengths_k = [1, 3, 2], [2, 5, 1]
        cuq = torch.tensor([0, 1, 4, 6], dtype=torch.int32)
        cuk = torch.tensor([0, 2, 7, 8], dtype=torch.int32)
        q = torch.randn(6, 2, 8, requires_grad=True)
        k = torch.randn(8, 2, 8, requires_grad=True)
        v = torch.randn(8, 2, 8, requires_grad=True)
        actual = packed_attention(q, k, v, cuq, cuk, 3, 5, "reference")
        padded_q = torch.zeros(3, 2, 3, 8)
        padded_k = torch.zeros(3, 2, 5, 8)
        padded_v = torch.zeros(3, 2, 5, 8)
        mask = torch.zeros(3, 1, 1, 5, dtype=torch.bool)
        for i, (nq, nk) in enumerate(zip(lengths_q, lengths_k)):
            padded_q[i, :, :nq] = q[cuq[i]:cuq[i+1]].transpose(0, 1)
            padded_k[i, :, :nk] = k[cuk[i]:cuk[i+1]].transpose(0, 1)
            padded_v[i, :, :nk] = v[cuk[i]:cuk[i+1]].transpose(0, 1)
            mask[i, :, :, :nk] = True
        padded = torch.nn.functional.scaled_dot_product_attention(padded_q, padded_k, padded_v, attn_mask=mask)
        expected = torch.cat([padded[i, :, :nq].transpose(0, 1) for i, nq in enumerate(lengths_q)])
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
        gb = torch.autograd.grad(expected.square().sum(), (q, k, v))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-6)

    def test_empty_neighbors_removed_and_position_computed_once(self):
        xyz = torch.tensor([[0., 0, 0], [.05, 0, 0], [.41, 0, 0], [2., 0, 0]])
        layout = make_voxels(xyz, torch.zeros(4, dtype=torch.long), .4)
        order, starts = cell_order(layout)
        neighbors = neighboring_cells(layout)
        attention = PackedNeighborAttention(16, 2, 2, 2, 1., 2, "reference").eval()
        feature = torch.randn(4, 16, requires_grad=True)
        with patch.object(attention.position, "forward", wraps=attention.position.forward) as position:
            out, record = attention(feature, xyz, layout, order, starts, neighbors, .4, True)
            self.assertEqual(position.call_count, 1)
            self.assertEqual(position.call_args.args[0].shape, (4, 3))
        expected = []
        for row in neighbors:
            expected.append(sum(min(int(layout.counts[n]), 2) for n in row if n >= 0))
        torch.testing.assert_close(record["packed_key_counts"], torch.tensor(expected))
        self.assertLess(int(record["packed_key_counts"].sum()), len(layout.counts) * 54)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())

    def test_flash_backend_requires_cuda(self):
        q = torch.randn(1, 2, 8)
        cu = torch.tensor([0, 1], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "CUDA FP16/BF16"):
            packed_attention(q, q, q, cu, cu, 1, 1, "flash_varlen")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_flash_varlen_matches_reference_forward_backward(self):
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError:
            self.skipTest("flash-attn unavailable")
        q = torch.randn(6, 2, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(8, 2, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        cuq = torch.tensor([0, 1, 4, 6], device="cuda", dtype=torch.int32)
        cuk = torch.tensor([0, 2, 7, 8], device="cuda", dtype=torch.int32)
        # Bounds may exceed actual maxima, as in the production fixed caps.
        actual = packed_attention(q, k, v, cuq, cuk, 8, 108, "flash_varlen")
        expected = packed_attention(q, k, v, cuq, cuk, 8, 108, "reference")
        torch.testing.assert_close(actual, expected, rtol=.01, atol=.003)
        ga = torch.autograd.grad(actual.float().square().mean(), (q, k, v), retain_graph=True)
        gb = torch.autograd.grad(expected.float().square().mean(), (q, k, v))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=.03, atol=.001)
