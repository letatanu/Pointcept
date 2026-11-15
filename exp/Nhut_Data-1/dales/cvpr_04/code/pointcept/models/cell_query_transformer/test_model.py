"""CPU checks; run from the directory containing cell_query_transformer."""
import unittest
import torch
from .model import CellQueryTransformer


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.model = CellQueryTransformer(
            channels=(12, 24), heads=(3, 3), depths=(2, 2),
            cell_sizes=(.4, .8), query_ratios=(.25, .5), max_queries=(2, 3),
            patch_sizes=(4, 4), chunk_size=3)
        self.coord = torch.cat((torch.rand(31, 3) * 1.2, torch.rand(19, 3) * 1.2))
        self.feat = torch.randn(50, 4)
        self.data = dict(coord=self.coord, feat=self.feat, offset=torch.tensor([31, 50]))

    def test_training_gradients_and_query_coverage(self):
        self.model.train()
        output, debug = self.model(self.data, return_debug=True)
        self.assertEqual(output.shape, (50, 64))
        self.assertTrue(torch.isfinite(output).all())
        for stage, record in enumerate(debug):
            selected = record["selected"]
            self.assertEqual(len(selected), len(selected.unique()))
            actual = torch.bincount(record["cell_inverse"][selected], minlength=len(record["cell_counts"]))
            expected = torch.ceil(record["cell_counts"] * (.25, .5)[stage]).long().clamp(1, (2, 3)[stage])
            torch.testing.assert_close(actual, expected)
            for patch in record["patches"]:
                for row in patch:
                    self.assertEqual(len(record["cell_batch"][row[row >= 0]].unique()), 1)
        output.square().mean().backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_evaluation_is_deterministic_and_batch_isolated(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.data)
            torch.testing.assert_close(output, self.model(self.data), rtol=0, atol=0)
            for lo, hi in ((0, 31), (31, 50)):
                single = self.model(dict(coord=self.coord[lo:hi], feat=self.feat[lo:hi]))
                torch.testing.assert_close(output[lo:hi], single, rtol=2e-5, atol=2e-6)

    def test_singleton_and_dense_cell(self):
        for n in (1, 37):
            data = dict(coord=torch.zeros(n, 3), feat=torch.randn(n, 4))
            output = self.model(data)
            self.assertEqual(output.shape, (n, 64))
            self.assertTrue(torch.isfinite(output).all())
            output.mean().backward()
            self.model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
