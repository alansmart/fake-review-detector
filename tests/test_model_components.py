import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dml import DMLNet, DMLScorer, supcon_loss
from data_utils import load_review_dataset
from features import FEATURE_COLS, extract_features
from trust import dml_trust_score, trust_score


class FeatureAndTrustTests(unittest.TestCase):
    def test_dataset_loader_rejects_unknown_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            pd.DataFrame(
                [{"text_": "Example review", "label": "unknown", "rating": 4}]
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "Expected only 'CG' and 'OR'"):
                load_review_dataset(path)

    def test_feature_extraction_returns_model_inputs(self):
        features = extract_features("I really like this useful product!", 5)

        self.assertEqual(set(features), set(FEATURE_COLS))
        self.assertGreater(features["sentiment"], 0)
        self.assertGreaterEqual(features["rating_sentiment_gap"], 0)

    def test_trust_scores_have_valid_ranges_and_levels(self):
        features = extract_features("Useful product, but setup was difficult.", 4)

        for result in (
            trust_score(0.35, features),
            dml_trust_score(0.35, 0.3, 0.1, features),
        ):
            self.assertIsInstance(result, dict)
            self.assertGreaterEqual(result["trust_score"], 0)
            self.assertLessEqual(result["trust_score"], 100)
            self.assertIn(result["level"], {"High", "Medium", "Low", "Uncertain"})

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            dml_trust_score(0.35, 0.3, 0.1, features, novelty_threshold=0)


class DMLTests(unittest.TestCase):
    def test_network_loss_and_scorer_run_for_both_classes(self):
        torch.manual_seed(7)
        model = DMLNet(text_dim=8, meta_dim=3, emb_dim=4).eval()
        text = torch.randn(4, 8)
        metadata = torch.randn(4, 3)
        labels = torch.tensor([0, 0, 1, 1])

        with torch.no_grad():
            embeddings = model(text, metadata)
        self.assertEqual(tuple(embeddings.shape), (4, 4))
        np.testing.assert_allclose(
            embeddings.norm(dim=1).numpy(), np.ones(4), atol=1e-5
        )

        loss = supcon_loss(embeddings, labels)
        self.assertTrue(torch.isfinite(loss).item())

        scorer = DMLScorer(embeddings, labels, k=10)
        probability, margin, novelty = scorer.score(embeddings[:2])
        self.assertEqual(tuple(probability.shape), (2,))
        self.assertTrue(torch.isfinite(probability).all().item())
        self.assertTrue(torch.isfinite(margin).all().item())
        self.assertTrue(torch.isfinite(novelty).all().item())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all().item())


if __name__ == "__main__":
    unittest.main()
