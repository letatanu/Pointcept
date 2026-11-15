import unittest
import torch
from .geometry import make_voxels
from .model import HybridCellQuery, neighboring_cells, cell_order, sample_members


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def model(self):
        return HybridCellQuery(channels=(12, 24), heads=(3, 3), depths=(2, 1),
            voxel_sizes=(.2, .4), cell_sizes=(.4, .8), query_ratios=(.25, .5),
            max_queries=(2, 3), keys_per_cell=(2, 2), chunk_size=3, sparse_backend="reference")

    def test_neighbor_lookup_matches_brute_force(self):
        coord = torch.tensor([[0., 0, 0], [.41, 0, 0], [1.21, 0, 0], [0., 0, 0], [.41, 0, 0]])
        layout = make_voxels(coord, torch.tensor([0, 0, 0, 1, 1]), .4)
        neighbors = neighboring_cells(layout)
        for i, index in enumerate(layout.indices):
            expected = ((layout.indices[:, 0] == index[0]) & ((layout.indices[:, 1:] - index[1:]).abs().amax(-1) <= 1)).nonzero(as_tuple=True)[0]
            actual = neighbors[i][neighbors[i] >= 0].sort().values
            torch.testing.assert_close(actual, expected)
        self.assertIn(1, neighbors[0].tolist())  # crosses the query-cell boundary
        self.assertNotIn(2, neighbors[0].tolist())

    def test_sampling_caps_uniqueness_and_coverage(self):
        coord = torch.cat((torch.zeros(19, 3), torch.ones(3, 3), torch.full((1, 3), 2.)))
        layout = make_voxels(coord, torch.zeros(len(coord), dtype=torch.long), .4)
        order, starts = cell_order(layout)
        for training in (False, True):
            members, valid = sample_members(layout, order, starts, 4, .25, training)
            actual = torch.bincount(layout.inverse[members[valid]], minlength=len(layout.counts))
            torch.testing.assert_close(actual, torch.ceil(layout.counts * .25).long().clamp(1, 4))
            self.assertEqual(len(members[valid]), len(members[valid].unique()))

    def test_training_backward_all_parameters(self):
        model = self.model().train()
        coord = torch.cat((torch.rand(23, 3) * 1.2, torch.rand(17, 3) * 1.2))
        feat = torch.randn(40, 4, requires_grad=True)
        out, debug = model(dict(coord=coord, feat=feat, offset=torch.tensor([23, 40])), True)
        self.assertEqual(out.shape, (40, 64))
        out.square().mean().backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        self.assertTrue((feat.grad.abs().sum(-1) > 0).all())
        for record in debug:
            selected = record["selected"]
            self.assertEqual(len(selected), len(selected.unique()))
            neighbors = record["neighbors"]
            rows, cols = (neighbors >= 0).nonzero(as_tuple=True)
            indices = record["cell_indices"]
            self.assertTrue((indices[rows, 0] == indices[neighbors[rows, cols], 0]).all())

    def test_eval_batch_isolation_and_chunk_equivalence(self):
        model = self.model().eval()
        coord, feat = torch.rand(29, 3), torch.randn(29, 4)
        data = dict(coord=coord, feat=feat, offset=torch.tensor([16, 29]))
        with torch.no_grad():
            output = model(data)
            torch.testing.assert_close(output, model(data), rtol=0, atol=0)
            for lo, hi in ((0, 16), (16, 29)):
                torch.testing.assert_close(output[lo:hi], model(dict(coord=coord[lo:hi], feat=feat[lo:hi])), rtol=2e-5, atol=2e-6)
            for stage in model.blocks:
                for block in stage:
                    block.attention.chunk_size = 100
            torch.testing.assert_close(output, model(data), rtol=2e-5, atol=2e-6)

    def test_singleton(self):
        model = self.model()
        output = model(dict(coord=torch.zeros(1, 3), feat=torch.ones(1, 4)))
        self.assertEqual(output.shape, (1, 64))
        output.mean().backward()
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_spconv_amp_backward(self):
        try:
            import spconv.pytorch
        except ImportError:
            self.skipTest("spconv unavailable")
        model = HybridCellQuery(channels=(12, 24), heads=(3, 3), depths=(1, 1),
            voxel_sizes=(.2, .4), cell_sizes=(.4, .8), query_ratios=(.25, .5),
            max_queries=(2, 3), keys_per_cell=(2, 2), chunk_size=16).cuda().train()
        data = dict(coord=torch.rand(80, 3, device="cuda"), feat=torch.randn(80, 4, device="cuda"))
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(data)
        output.float().square().mean().backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
