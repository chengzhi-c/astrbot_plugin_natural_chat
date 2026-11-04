"""检测器评测门禁：标注集与实现同源，任一不一致即失败。"""

from __future__ import annotations

import json
import unittest

from tests._support import ensure_plugin_package

ensure_plugin_package()


from scripts import eval_detector


class FixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = json.loads(eval_detector.FIXTURE.read_text(encoding="utf-8"))

    def test_every_signal_name_has_a_positive_sample(self) -> None:
        self.assertEqual(eval_detector.uncovered_signal_names(self.rows), [])

    def test_annotations_match_the_detector(self) -> None:
        for split in ("dev", "holdout"):
            report = eval_detector.evaluate([row for row in self.rows if row["split"] == split])
            self.assertEqual(report["mismatches"], [], split)

    def test_both_splits_are_populated(self) -> None:
        splits = {row["split"] for row in self.rows}
        self.assertEqual(splits, {"dev", "holdout"})

    def test_ids_are_unique(self) -> None:
        ids = [row["id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))


class CheckModeTest(unittest.TestCase):
    def test_check_passes_on_the_frozen_fixture(self) -> None:
        self.assertEqual(eval_detector.main(["--check"]), 0)

    def test_uncovered_signal_fails_closed(self) -> None:
        rows = [
            {
                "answer": "普通回答",
                "user": "",
                "expected_signals": [],
                "formal_bypass": False,
                "id": "x",
                "split": "dev",
                "category": "casual",
            }
        ]
        self.assertTrue(eval_detector.uncovered_signal_names(rows))


class MetricTest(unittest.TestCase):
    def test_empty_category_reports_not_applicable(self) -> None:
        self.assertEqual(
            eval_detector._binary_metrics([]), {"precision": None, "recall": None, "fp": 0, "fn": 0, "n": 0}
        )

    def test_counts_split_between_false_positive_and_negative(self) -> None:
        metrics = eval_detector._binary_metrics([(True, True), (True, False), (False, True), (False, False)])
        self.assertEqual((metrics["fp"], metrics["fn"], metrics["n"]), (1, 1, 4))


if __name__ == "__main__":
    unittest.main()
