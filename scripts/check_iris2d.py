"""Numerical/model smoke checks; no real images, no training results."""
from pathlib import Path
import ast
import copy
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from iris2d.model import Iris2D, high_resolution_masked_pool, segmentation_loss
from iris2d.protocol import ROOT, load_recipe, portable_ast_sha256, set_cosine_lr, verify_reference


class IrisChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_python_ast_versions_and_comment_compatibility(self):
        source = (ROOT/'scripts/train_unet.py').read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        simulated_old = copy.deepcopy(tree)
        for node in ast.walk(simulated_old):
            if 'type_params' in node._fields and not getattr(node, 'type_params', None):
                node._fields = tuple(name for name in node._fields if name != 'type_params')
        self.assertEqual(portable_ast_sha256(tree), portable_ast_sha256(simulated_old))
        simulated_new = copy.deepcopy(simulated_old)
        for node in ast.walk(simulated_new):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and 'type_params' not in node._fields:
                node._fields = (*node._fields, 'type_params')
                node.type_params = []
        self.assertEqual(portable_ast_sha256(tree), portable_ast_sha256(simulated_new))
        self.assertEqual(portable_ast_sha256(ast.parse('x = 1\n')),
                         portable_ast_sha256(ast.parse('# comment\nx=1  # comment\n')))
        self.assertNotEqual(portable_ast_sha256(ast.parse('x=1')),
                            portable_ast_sha256(ast.parse('x=2')))

    def test_lr_exactly_matches_four_existing_baselines(self):
        manifest = verify_reference()
        recipe, common = load_recipe()
        self.assertEqual(recipe['lr'], 1e-4)
        self.assertEqual(recipe['min_lr'], 1e-6)
        self.assertEqual(common['warmup_epochs'], 5)
        class Optimizer:
            def __init__(self):
                self.param_groups = [{}, {}]
        for relative in manifest['learning_rate_function_sha256']:
            tree = ast.parse((ROOT/relative).read_text(encoding='utf-8-sig'))
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'set_cosine_lr')
            namespace = {'math': math, 'torch': torch}
            exec(compile(ast.Module(body=[function], type_ignores=[]), relative, 'exec'), namespace)
            reference = namespace['set_cosine_lr']
            for epochs, warmup in ((150, 5), (8, 0), (3, 5)):
                a, b = Optimizer(), Optimizer()
                for epoch in range(epochs + 2):
                    actual = set_cosine_lr(a, epoch, epochs, recipe['lr'], recipe['min_lr'], warmup)
                    expected = reference(b, epoch, epochs, recipe['lr'], recipe['min_lr'], warmup)
                    self.assertEqual(actual, expected, (relative, epochs, warmup, epoch))
                    self.assertEqual(a.param_groups, b.param_groups)

    def test_full_resolution_pool_values_and_gradients(self):
        for h, w, H, W in [(3, 4, 31, 47), (4, 4, 64, 64), (3, 5, 19, 35)]:
            feature = torch.randn(2, 7, h, w, requires_grad=True)
            mask = (torch.rand(2, 1, H, W) > .7).float()
            mask[0] = 0
            mask[0, 0, 0, 0] = 1  # tiny foreground at the interpolation boundary
            actual = high_resolution_masked_pool(feature, mask)
            up = F.interpolate(feature, size=(H, W), mode="bilinear", align_corners=False)
            expected = ((up * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1e-6))[:, None]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            ga = torch.autograd.grad(actual.square().sum(), feature, retain_graph=True)[0]
            ge = torch.autograd.grad(expected.square().sum(), feature)[0]
            torch.testing.assert_close(ga, ge, atol=2e-6, rtol=2e-5)

    def test_support_ensemble_backprop_and_roundtrip(self):
        model = Iris2D(base_channels=16, context_queries=3, attention_depth=1, dropout=0.)
        xs, xq = torch.randn(2, 1, 32, 48), torch.randn(2, 1, 32, 48)
        ys = torch.zeros(2, 1, 32, 48)
        ys[:, :, 9:20, 10:25] = 1
        target = torch.zeros(2, 32, 48, dtype=torch.long)
        target[0, 9:20, 10:25] = 1  # the second query is a valid negative case
        model.eval()
        separate = model.encode_support(xs, ys, aggregate=False)
        ensemble = model.encode_support(xs, ys)
        torch.testing.assert_close(ensemble, separate.mean(0, keepdim=True))
        logits = model.segment(xq, ensemble)
        self.assertEqual(tuple(logits.shape), (2, 2, 32, 48))
        loss = segmentation_loss(logits, target)
        loss.backward()
        for prefix in ("encoder", "task_encoder.mask_fusion", "task_encoder.queries", "two_way", "mask_token_head"):
            gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
            self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), prefix)
        restored = Iris2D(**model.config).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(xs, ys, xq), logits.detach())
            changed = model.encode_support(xs, ys.roll(12, dims=-1))
            self.assertGreater(float((model.segment(xq, changed) - logits.detach()).abs().max()), 1e-6)

    def test_invalid_or_empty_support(self):
        model = Iris2D(base_channels=16, attention_depth=1)
        with self.assertRaises(ValueError):
            model.encode_support(torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 32, 32))
        with self.assertRaises(ValueError):
            model.encode_support(torch.zeros(1, 1, 31, 32), torch.ones(1, 1, 31, 32))


if __name__ == "__main__":
    unittest.main()
