#!/usr/bin/env python3
"""用冻结的人工标注集跑检测器与让位判定，输出 precision/recall 与误判明细。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from tests._support import ensure_plugin_package

ensure_plugin_package()

from astrbot_plugin_natural_chat.scene_guard import is_tone_request
from astrbot_plugin_natural_chat.signal_detectors import detect_cliches, signal_names

FIXTURE = repo / "tests" / "fixtures" / "detector_eval.json"


def _binary_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, float | int | None]:
    tp = sum(want and got for want, got in pairs)
    fp = sum(not want and got for want, got in pairs)
    fn = sum(want and not got for want, got in pairs)
    return {
        # 空类别 precision/recall 记 None（n/a），与「有样本且满分」区分
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "fp": fp,
        "fn": fn,
        "n": len(pairs),
    }


def evaluate(rows: list[dict[str, object]]) -> dict[str, object]:
    signal_pairs: list[tuple[bool, bool]] = []
    bypass_pairs: list[tuple[bool, bool]] = []
    mismatches: list[str] = []
    by_category: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for row in rows:
        expected = {str(item) for item in row["expected_signals"]}
        actual = set(detect_cliches(str(row["answer"]), user_text=str(row["user"])))
        for signal in sorted(expected | actual):
            pair = (signal in expected, signal in actual)
            signal_pairs.append(pair)
            by_category[str(row["category"])].append(pair)
            if pair[0] != pair[1]:
                mismatches.append(f"{row['id']} {signal}: expected={pair[0]} actual={pair[1]}")
        want_bypass = bool(row["formal_bypass"])
        got_bypass = is_tone_request(str(row["user"])) is not None
        bypass_pairs.append((want_bypass, got_bypass))
        if want_bypass != got_bypass:
            mismatches.append(f"{row['id']} bypass: expected={want_bypass} actual={got_bypass}")
    return {
        "count": len(rows),
        "signals": _binary_metrics(signal_pairs),
        "bypass": _binary_metrics(bypass_pairs),
        "categories": {name: _binary_metrics(pairs) for name, pairs in sorted(by_category.items())},
        "mismatches": mismatches,
    }


def uncovered_signal_names(rows: list[dict[str, object]]) -> list[str]:
    """固定词表里没有任何正例覆盖的信号名（新增词表项必须同时补样例）。"""
    covered = {str(item) for row in rows for item in row["expected_signals"]}
    return sorted(name for name in signal_names() if name not in covered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="任一标注不一致或有未覆盖信号就返回非零")
    args = parser.parse_args(argv)
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    uncovered = uncovered_signal_names(rows)
    report = {
        "dev": evaluate([row for row in rows if row["split"] == "dev"]),
        "holdout": evaluate([row for row in rows if row["split"] == "holdout"]),
        "uncovered_signals": uncovered,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not args.check:
        return 0
    mismatched = any(report[split]["mismatches"] for split in ("dev", "holdout"))
    return 1 if uncovered or mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
