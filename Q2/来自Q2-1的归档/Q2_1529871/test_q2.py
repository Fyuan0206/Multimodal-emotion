"""Focused checks for the Q2 masking, pooling, and model interface."""

import unittest

import numpy as np
import torch

from Q2.run_q2 import RobustFusion, augmented_masks, fit_variant, pool_features, predict


class Q2InterfaceTests(unittest.TestCase):
    def test_contiguous_local_mask_keeps_other_modalities(self):
        eligible = np.array([[False, True, True, True, True, True, True, False]])
        base = np.repeat(eligible[:, :, None], 3, axis=2)
        changed = augmented_masks(base, eligible, seed=1, modes=(1,), rate=0.5, position="middle")
        removed = np.flatnonzero(base[0, :, 1] & ~changed[0, :, 1])
        self.assertEqual(len(removed), 3)
        self.assertTrue(np.all(np.diff(removed) == 1))
        self.assertTrue(np.array_equal(changed[:, :, 0], base[:, :, 0]))
        self.assertTrue(np.array_equal(changed[:, :, 2], base[:, :, 2]))
        self.assertGreater(changed[0, :, 1].sum(), 0)

    def test_pooling_ignores_unobserved_values(self):
        eligible = np.array([[False, True, True, False]])
        masks = np.zeros((1, 4, 3), dtype=bool)
        masks[0, 1, :] = True
        values = {}
        for mode, dim in (("text", 768), ("audio", 74), ("vision", 35)):
            arr = np.zeros((1, 4, dim), dtype=np.float32)
            arr[0, 1, :] = 2
            arr[0, 2, :] = 1000
            values[mode] = arr
        pooled = pool_features({"x": values, "eligible": eligible, "masks": masks})
        self.assertAlmostEqual(float(pooled["coverage"][0, 0]), 0.5)
        self.assertTrue(np.allclose(pooled["text"][0, :768], 2))
        self.assertTrue(np.allclose(pooled["audio"][0, :74], 2))

    def test_three_modality_model_output_shapes(self):
        model = RobustFusion()
        inputs = [torch.zeros(2, 5 * dim) for dim in (768, 74, 35)]
        regression, logits = model(*inputs, torch.ones(2, 3))
        self.assertEqual(tuple(regression.shape), (2,))
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(regression).all().item())
        self.assertTrue(torch.isfinite(logits).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_training_and_prediction_on_cuda(self):
        rng = np.random.default_rng(7)
        size = 12
        features = {
            mode: rng.normal(size=(size, 5 * dim)).astype(np.float32)
            for mode, dim in (("text", 768), ("audio", 74), ("vision", 35))
        }
        features["coverage"] = np.ones((size, 3), dtype=np.float32)
        labels_reg = np.linspace(-1, 1, size, dtype=np.float32)
        labels_cls = np.arange(size, dtype=np.int64) % 3
        model, _ = fit_variant(features, labels_reg, labels_cls, features, features,
                               labels_reg, labels_cls, True, 7, torch.device("cuda"),
                               epochs=1, patience=1)
        regression, logits = predict(model, features)
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        self.assertEqual(regression.shape, (size,))
        self.assertEqual(logits.shape, (size, 3))
        self.assertTrue(np.isfinite(regression).all())


if __name__ == "__main__":
    unittest.main()
