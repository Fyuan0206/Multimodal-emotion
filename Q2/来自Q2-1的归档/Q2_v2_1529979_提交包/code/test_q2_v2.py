"""Tests for scientific protocol invariants, independent of server/BERT weights."""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
import run_q2_v2 as v2


class ProtocolTests(unittest.TestCase):
    def test_contiguous_original_observation_retained(self):
        eligible = np.ones((20, 20), bool)
        observed = np.ones((20, 20, 3), bool)
        observed[:, :, 1] = False
        observed[:, [1, 8, 18], 1] = True
        masks, manifest = v2.span_masks(observed, eligible, 8, (0, 1, 2), .35, "middle")
        self.assertTrue(manifest["valid"].all())
        self.assertTrue(np.all(masks.sum(1) >= 1))
        for i in range(20):
            removed_text = np.flatnonzero(observed[i, :, 0] & ~masks[i, :, 0])
            np.testing.assert_array_equal(np.diff(removed_text), np.ones(6))
            self.assertEqual(len(removed_text), 7)
        second, _ = v2.span_masks(observed, eligible, 8, (0, 1, 2), .35, "middle")
        np.testing.assert_array_equal(masks, second)

    def test_infeasible_not_changed(self):
        eligible = np.ones((2, 10), bool)
        observed = np.zeros((2, 10, 3), bool)
        observed[:, 3, :] = True
        masks, manifest = v2.span_masks(observed, eligible, 7, (1,), .55, "early")
        self.assertFalse(manifest["valid"].any())
        np.testing.assert_array_equal(masks, observed)
        self.assertTrue((manifest["width"] == 6).all())

    def test_padding_and_position(self):
        eligible = np.zeros((1, 50), bool)
        eligible[:, 1:21] = True
        observed = np.repeat(eligible[:, :, None], 3, axis=2)
        for pos, idx in zip(v2.POSITIONS, range(3)):
            masks, manifest = v2.span_masks(observed, eligible, 7, (0,), .55, pos)
            self.assertTrue(manifest["valid"][0])
            self.assertEqual(int(manifest["center"][0] * 3), idx)
            self.assertFalse(masks[:, 21:].any())

    def test_metrics_and_decoding(self):
        y = np.array([-1., 0., .8, -.5, 1.])
        cls = np.sign(y).astype(int) + 1
        raw = np.array([-.6, .04, -.1, -.3, .9])
        reg, pred = v2.decode(raw, .05)
        np.testing.assert_array_equal(pred, np.sign(reg).astype(int) + 1)
        stats = v2.batch_metrics(y, cls, reg, pred, np.ones(5, bool))[0]
        self.assertAlmostEqual(stats[0], accuracy_score(cls, pred))
        self.assertAlmostEqual(stats[1], f1_score(cls, pred, labels=[0, 1, 2], average="macro"))
        self.assertAlmostEqual(stats[2], mean_absolute_error(y, reg))
        self.assertAlmostEqual(stats[3], np.corrcoef(y, reg)[0, 1])
        masked = v2.batch_metrics(y, cls, reg, pred, np.array([1,1,0,0,1], bool))[0]
        self.assertAlmostEqual(masked[0], 1.)

    def test_shared_initialization(self):
        import torch
        torch.manual_seed(2026)
        a = v2.base.RobustFusion(False)
        torch.manual_seed(2026)
        b = v2.base.RobustFusion(True)
        for key in a.state_dict():
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key])

    def test_group_bootstrap_metric_matches_direct_resampling(self):
        from report_q2_v2 import bootstrap_metrics
        y = np.array([-1., 0., .8, -.5, 1.], dtype=np.float32)
        cls = np.sign(y).astype(int) + 1
        reg = np.array([[-.6, 0., -.1, -.3, .9], [-.7, .2, .6, -.1, .8]], dtype=np.float32)
        pred = np.sign(reg).astype(int) + 1
        valid = np.ones_like(reg, dtype=bool)
        weights = np.array([[1,2],[1,2],[1,0],[1,1],[1,1]], dtype=np.float32)
        observed = bootstrap_metrics(reg, pred, valid, y, cls, weights)
        for index in range(2):
            samples = np.repeat(np.arange(5), weights[:, index].astype(int))
            expected = v2.batch_metrics(y[samples], cls[samples], reg[:, samples], pred[:, samples], valid[:, samples]).mean(0)
            np.testing.assert_allclose(observed[index], expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
