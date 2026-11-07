"""Tests for the Recall@k and nDCG@k multilabel metrics.

Contributor: Alon Levy (NetID: alonl2, alonl2@illinois.edu)
Paper: "Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction"
    (Yu et al., CHIL 2025), https://arxiv.org/abs/2502.10689
"""

import math
import unittest

import numpy as np

from pyhealth.metrics import multilabel_metrics_fn, ndcg_at_k, recall_at_k


class TestRankingAtK(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([[0, 1, 1, 0], [1, 0, 0, 0]])
        self.y_prob = np.array([[0.1, 0.9, 0.2, 0.3], [0.3, 0.6, 0.1, 0.0]])

    def test_recall_at_k_hand_computed(self):
        # Sample 1: top-1 = {1} -> 1/2; top-2 = {1, 3} -> 1/2; top-3 adds 0 -> 1/2
        # Sample 2: top-1 = {1} -> 0;  top-2 = {1, 0} -> 1
        self.assertAlmostEqual(recall_at_k(self.y_true, self.y_prob, 1), 0.25)
        self.assertAlmostEqual(recall_at_k(self.y_true, self.y_prob, 2), 0.75)
        self.assertAlmostEqual(recall_at_k(self.y_true, self.y_prob, 4), 1.0)

    def test_ndcg_at_k_hand_computed(self):
        # Sample 1 at k=2: hits at rank 1 only -> 1 / (1 + 1/log2(3))
        # Sample 2 at k=2: hit at rank 2 -> (1/log2(3)) / 1
        s1 = 1.0 / (1.0 + 1.0 / math.log2(3))
        s2 = 1.0 / math.log2(3)
        expected = (s1 + s2) / 2
        self.assertAlmostEqual(ndcg_at_k(self.y_true, self.y_prob, 2), expected)

    def test_perfect_ranking(self):
        y_prob = self.y_true.astype(float)
        self.assertAlmostEqual(ndcg_at_k(self.y_true, y_prob, 3), 1.0)
        self.assertAlmostEqual(recall_at_k(self.y_true, y_prob, 2), 1.0)

    def test_k_larger_than_labels(self):
        self.assertAlmostEqual(recall_at_k(self.y_true, self.y_prob, 100), 1.0)
        self.assertLessEqual(ndcg_at_k(self.y_true, self.y_prob, 100), 1.0)

    def test_samples_without_labels_skipped(self):
        y_true = np.vstack([self.y_true, np.zeros((1, 4), dtype=int)])
        y_prob = np.vstack([self.y_prob, np.ones((1, 4))])
        self.assertAlmostEqual(recall_at_k(y_true, y_prob, 2), 0.75)
        self.assertEqual(recall_at_k(np.zeros((2, 3)), np.ones((2, 3)), 1), 0.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            recall_at_k(self.y_true, self.y_prob, 0)
        with self.assertRaises(ValueError):
            ndcg_at_k(self.y_true, self.y_prob, 0)

    def test_multilabel_metrics_fn_names(self):
        out = multilabel_metrics_fn(
            self.y_true,
            self.y_prob,
            metrics=["recall@2", "ndcg@2", "pr_auc_samples"],
        )
        self.assertAlmostEqual(out["recall@2"], 0.75)
        self.assertIn("ndcg@2", out)
        self.assertIn("pr_auc_samples", out)

    def test_unknown_at_k_name_rejected(self):
        with self.assertRaises(ValueError):
            multilabel_metrics_fn(self.y_true, self.y_prob, metrics=["precision@2"])
        with self.assertRaises(ValueError):
            multilabel_metrics_fn(self.y_true, self.y_prob, metrics=["recall@x"])


if __name__ == "__main__":
    unittest.main()
