"""Pooling math, hierarchy, gradients and full-model controls."""
import copy
import itertools
import runpy
from pathlib import Path
import unittest
import torch
from .geometry import make_voxels, make_grid_stride_voxels
from .model import LearnedPooling, MeanPooling, PTv3Pooling, SerializedLocalCross


class PoolingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(52)
        torch.set_num_threads(2)
        self.coord = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [.5, 0., 0.], [.6, 0., 0.]])
        self.batch = torch.zeros(4, dtype=torch.long)
        self.mass = torch.tensor([1., 3., 2., 4.])
        self.cells = make_voxels(self.coord, self.batch, .4)

    def test_mean_matches_uniform_learned_pool_and_preserves_geometry(self):
        learned, mean = LearnedPooling(8, 16), MeanPooling(8, 16)
        mean.project.load_state_dict(learned.project.state_dict())
        torch.nn.init.zeros_(learned.score[-1].weight)
        torch.nn.init.zeros_(learned.score[-1].bias)
        feature = torch.randn(4, 8, requires_grad=True)
        a = learned(feature, self.coord, self.mass, self.cells, .4, True)
        b = mean(feature, self.coord, self.mass, self.cells, .4, True)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
        b[0].square().mean().backward()
        self.assertTrue((feature.grad.abs().sum(-1) > 0).all())
        self.assertFalse(any("score" in name for name, _ in mean.named_parameters()))

    def test_ptv3_projection_reduce_norm_activation_and_coordinate_means(self):
        for reduce, norm in itertools.product(("mean", "max", "sum", "min"), ("batch", "layer", "none")):
            with self.subTest(reduce=reduce, norm=norm):
                pool = PTv3Pooling(8, 16, reduce, norm).eval()
                feature = torch.randn(4, 8, requires_grad=True)
                value = pool.proj(feature)
                rows = []
                for i in range(2):
                    v = value[self.cells.inverse == i]
                    rows.append(v.mean(0) if reduce == "mean" else v.sum(0) if reduce == "sum"
                                else v.max(0).values if reduce == "max" else v.min(0).values)
                expected = pool.act(pool.norm(torch.stack(rows)))
                output, center, mass, batch = pool(feature, self.coord, self.mass, self.cells, .4)
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(center, torch.stack((self.coord[:2].mean(0), self.coord[2:].mean(0))))
                torch.testing.assert_close(mass, torch.tensor([4., 6.]))
                self.assertEqual(batch.tolist(), [0, 0])
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(feature.grad).all())
                for parameter in pool.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
        pool = PTv3Pooling(8, 16).train()
        self.assertEqual(pool.norm.eps, 1e-3)
        self.assertEqual(pool.norm.momentum, .01)
        pool(torch.randn(4, 8), self.coord, self.mass, self.cells, .4)[0].square().mean().backward()
        self.assertEqual(int(pool.norm.num_batches_tracked), 1)

    def test_stride_partition_matches_bit_prefix_and_stays_cloud_safe(self):
        grid = torch.tensor([[0,0,0], [1,1,1], [2,0,0], [3,0,0], [8,0,0], [0,0,0], [2,0,0]])
        batch = torch.tensor([0,0,0,0,0,1,1])
        from .serialization import reference_encode
        for stride in (2, 4, 8):
            cells = make_grid_stride_voxels(grid, batch, stride)
            depth = stride.bit_length() - 1
            for order in ("z", "z-trans", "hilbert", "hilbert-trans"):
                prefix = reference_encode(grid, 5, order) >> (3 * depth)
                same_prefix = (prefix[:, None] == prefix[None]) & (batch[:, None] == batch[None])
                same_cell = cells.inverse[:, None] == cells.inverse[None]
                self.assertTrue(torch.equal(same_prefix, same_cell), (stride, order))
        first = make_grid_stride_voxels(grid, batch, 2)
        second = make_grid_stride_voxels(first.indices[:, 1:], first.indices[:, 0], 2)
        direct = make_grid_stride_voxels(grid, batch, 4)
        torch.testing.assert_close(second.inverse[first.inverse], direct.inverse)

    def test_modes_and_partitions_full_model_forward_backward(self):
        for mode, partition in itertools.product(("learned", "mean", "ptv3"), ("metric", "grid_stride")):
            with self.subTest(mode=mode, partition=partition):
                model = SerializedLocalCross(
                    channels=(16,32), heads=(2,4), depths=(2,2), patch_sizes=(4,4),
                    pool_sizes=(.4,), position_scales=(.4,.8), cross_queries=(2,2),
                    cross_keys=(2,2), cross_every=1, cross_mode="central",
                    enable_coarse_attention=True, coarse_channels=16, coarse_heads=2,
                    coarse_grid_size=(2,2,2), coarse_stages=(0,1),
                    attention_backend="reference", serialization_backend="reference",
                    pooling_mode=mode, pooling_partition=partition, pooling_grid_size=.2,
                    pooling_strides=(2,)).train()
                data = dict(coord=torch.rand(39,3)*1.3, feat=torch.randn(39,4), offset=torch.tensor([23,39]))
                out = model(data)
                self.assertEqual(out.shape, (39,64))
                out.square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                # Evaluation uses fixed BN statistics and must retain cloud isolation.
                model.eval()
                with torch.no_grad():
                    together = model(data)
                    single = model(dict(coord=data["coord"][:23], feat=data["feat"][:23]))
                    torch.testing.assert_close(together[:23], single, rtol=2e-5, atol=3e-6)

    def test_generated_pooling_configs(self):
        root = Path(__file__).resolve().parent.parent
        expected = (("cvpr_pool_mean_mlp.py", "mean", "metric", "max"),
                    ("cvpr_pool_ptv3_max.py", "ptv3", "grid_stride", "max"),
                    ("cvpr_pool_ptv3_mean.py", "ptv3", "grid_stride", "mean"))
        for filename, mode, partition, reduce in expected:
            cfg = runpy.run_path(str(root / filename))
            backbone = cfg["model"]["backbone"]
            self.assertEqual(backbone["pooling_mode"], mode)
            self.assertEqual(backbone["pooling_partition"], partition)
            self.assertEqual(backbone["pooling_reduce"], reduce)
            self.assertEqual(backbone["pooling_strides"], (2,2,2))


if __name__ == "__main__":
    unittest.main()
