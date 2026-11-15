"""Behavioral checks for ablation flags, sliding windows and global communication."""
import copy
import itertools
import unittest
import torch
from .model import SerializedLocalCross
from .geometry import make_patches
from .layout import sample_central_cross
from .coarse import CoarseGlobalContext, coarse_cells


class ThreeAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def model(self, local=True, cross=True, coarse=True, checkpoint=False):
        return SerializedLocalCross(
            channels=(16, 32), heads=(2, 4), depths=(2, 2), patch_sizes=(4, 4),
            pool_sizes=(.4,), position_scales=(.4, .8), cross_queries=(2, 2),
            cross_keys=(2, 2), cross_every=1, cross_dilations=(1, 2),
            local_rotary="qk", cross_rotary="qk", cross_mode="central",
            cross_window_size=5, attention_backend="reference", serialization_backend="reference",
            enable_local_attention=local, enable_cross_attention=cross,
            enable_coarse_attention=coarse, coarse_channels=16, coarse_heads=2,
            coarse_grid_size=(2, 2, 2), checkpoint_blocks=checkpoint, chunk_size=3)

    def data(self):
        return dict(coord=torch.rand(39, 3) * 1.3, feat=torch.randn(39, 4),
                    offset=torch.tensor([23, 39]))

    def test_all_eight_ablations_forward_backward_and_module_removal(self):
        for flags in itertools.product((False, True), repeat=3):
            with self.subTest(flags=flags):
                model = self.model(*flags).train()
                data = self.data()
                data["feat"].requires_grad_()
                output, debug = model(data, True)
                self.assertEqual(output.shape, (39, 64))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(model.coarse is not None, flags[2])
                for stage in model.stages:
                    for block in stage:
                        self.assertEqual(block.local is not None, flags[0])
                        self.assertEqual(block.cross is not None, flags[1])
                output.square().mean().backward()
                self.assertTrue((data["feat"].grad.abs().sum(-1) > 0).all())
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, (flags, name))
                    self.assertTrue(torch.isfinite(parameter.grad).all(), (flags, name))
                if flags[2]:
                    self.assertTrue((debug[-1]["token_counts"] <= 8).all())

    def test_exact_central_window_cloud_isolation_and_unique_queries(self):
        coord = torch.arange(38).float()[:, None].expand(-1, 3) * .1
        batch = torch.tensor([0] * 21 + [1] * 17)
        patches = make_patches(coord, batch, .05, (0, 1, 2), 4)
        owner = torch.empty(len(coord), dtype=torch.long)
        for i, row in enumerate(patches.members):
            owner[row[row >= 0]] = i
        patch_batch = batch[patches.members[:, 0]]
        for training, dilation in itertools.product((False, True), (1, 2, 4)):
            q, k = sample_central_cross(patches, batch, 5, dilation, 3, 2, training)
            ids = q[q >= 0]
            self.assertEqual(len(ids), len(ids.unique()))
            for center in range(len(q)):
                queries, keys = q[center][q[center] >= 0], k[center][k[center] >= 0]
                self.assertTrue((owner[queries] == center).all())
                expected = {center + d * dilation for d in range(-2, 3)
                            if 0 <= center + d * dilation < len(q)
                            and patch_batch[center + d * dilation] == patch_batch[center]}
                self.assertEqual(set(owner[keys].tolist()), expected)
                self.assertEqual(len(keys), len(keys.unique()))
                self.assertTrue((batch[keys] == patch_batch[center]).all())

    def test_batch_matches_independent_clouds_and_deterministic_eval(self):
        model = self.model().eval()
        data = self.data()
        with torch.no_grad():
            output = model(data)
            torch.testing.assert_close(output, model(data), rtol=0, atol=0)
            for lo, hi in ((0, 23), (23, 39)):
                single = model(dict(coord=data["coord"][lo:hi], feat=data["feat"][lo:hi]))
                torch.testing.assert_close(output[lo:hi], single, rtol=2e-5, atol=3e-6)
            from .attention import PackedAttention
            for module in model.modules():
                if isinstance(module, PackedAttention):
                    module.chunk_size = 100
            torch.testing.assert_close(output, model(data), rtol=2e-5, atol=3e-6)

    def test_commented_config_and_generated_ablations(self):
        import runpy
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        for flags in itertools.product((False, True), repeat=3):
            tag = "L%d_X%d_G%d" % tuple(map(int, flags))
            config = runpy.run_path(str(root / f"cvpr_{tag}.py"))
            backbone = config["model"]["backbone"].copy()
            self.assertEqual(tuple(backbone[name] for name in (
                "enable_local_attention", "enable_cross_attention", "enable_coarse_attention")), flags)
            self.assertTrue(config["save_path"].endswith(tag))
        config = runpy.run_path(str(root / "cvpr_three_attention.py"))
        backbone = config["model"]["backbone"].copy()
        backbone.pop("type")
        backbone.update(attention_backend="reference", serialization_backend="reference")
        model = SerializedLocalCross(**backbone).eval()
        data = self.data()
        output = model(data)
        self.assertEqual(output.shape, (39, 64))
        output.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_global_branch_connects_different_cells(self):
        context = CoarseGlobalContext(16, 16, 2, grid_size=(2, 2, 2), backend="reference")
        feature = torch.randn(3, 16, requires_grad=True)
        coord = torch.tensor([[0., 0., 0.], [1., 1., 1.], [10., 10., 10.]])
        batch = torch.zeros(3, dtype=torch.long)
        _, inverse = coarse_cells(coord, batch, (2, 2, 2))
        self.assertNotEqual(int(inverse[0]), int(inverse[-1]))
        output = context(feature, coord, batch, torch.ones(3))
        output[0].square().sum().backward()
        self.assertGreater(float(feature.grad[-1].abs().sum()), 0.)

    def test_mass_weighted_pooling_and_global_token_bound(self):
        context = CoarseGlobalContext(16, 16, 2, grid_size=(1, 1, 1), backend="reference")
        feature = torch.randn(3, 16)
        coord = torch.rand(3, 3)
        mass = torch.tensor([1., 2., 4.])
        seen = []
        handle = context.project_in.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        _, debug = context(feature, coord, torch.zeros(3, dtype=torch.long), mass, True)
        handle.remove()
        torch.testing.assert_close(seen[0][0], (feature * mass[:, None]).sum(0) / mass.sum())
        self.assertEqual(debug["token_counts"].tolist(), [1])

    def test_singleton_degenerate_axes_and_checkpoint_equivalence(self):
        model = self.model().eval()
        singleton = model(dict(coord=torch.zeros(1, 3), feat=torch.randn(1, 4)))
        self.assertTrue(torch.isfinite(singleton).all())
        singleton.mean().backward()
        plain = self.model().train()
        checked = copy.deepcopy(plain)
        checked.checkpoint_blocks = True
        data = self.data()
        torch.manual_seed(100)
        a = plain(data)
        a.square().mean().backward()
        torch.manual_seed(100)
        b = checked(data)
        b.square().mean().backward()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for (name, x), (_, y) in zip(plain.named_parameters(), checked.named_parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=2e-6, msg=name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_all_branches_flash_amp_and_fp32_evaluation(self):
        import importlib.util
        if importlib.util.find_spec("flash_attn") is None:
            self.skipTest("flash-attn unavailable")
        from .attention import PackedAttention
        model = self.model().cuda().train()
        for module in model.modules():
            if isinstance(module, PackedAttention):
                module.backend = "flash_varlen"
        data = {key: value.cuda() for key, value in self.data().items()}
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(data)
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            self.assertTrue(torch.isfinite(model.eval()(data)).all())


if __name__ == "__main__":
    unittest.main()
