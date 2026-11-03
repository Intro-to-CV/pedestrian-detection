"""Small protocol checks for the shared evaluator's test threshold."""

import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate import average_precision, main, precision_recall_at_confidence


class TestSharedEvaluator(unittest.TestCase):
    def test_test_split_requires_locked_threshold(self):
        args = ["evaluate.py", "--labels", "/missing", "--predictions", "/missing",
                "--output-dir", "/missing", "--split", "test"]
        with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "validation-selected"):
            main()

    def test_perfect_ap(self):
        self.assertEqual(average_precision(np.array([1.0, 1.0]), 2), 1.0)

    def test_fixed_confidence_counts(self):
        tp = np.array([1.0, 0.0, 1.0])
        confidence = np.array([0.9, 0.7, 0.2])
        self.assertEqual(precision_recall_at_confidence(tp, confidence, 3, 0.5),
                         (0.5, 1 / 3, 1, 1, 2))
        self.assertEqual(precision_recall_at_confidence(tp, confidence, 3, 0.95),
                         (0.0, 0.0, 0, 0, 3))


if __name__ == "__main__":
    unittest.main()
