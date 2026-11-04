#!/usr/bin/env python3
"""检测注入规则相对上游 natural-talk 的漂移。

上游 SKILL.md 是规则语义的来源，本插件只保留日常对话与日常工作够用的子集；
本脚本核对两件事：锚点仍在本地规则里，上游对这些锚点仍然这么说。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANCHORS_PATH = REPO / "tests" / "fixtures" / "stable-rules-anchors.json"
LITE_FIXTURE_PATH = REPO / "tests" / "fixtures" / "system-prompt-lite.txt"


def load_anchors() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    data = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    return (
        tuple(data["guideline_tags"]),
        tuple(data["anchors"]),
        tuple(data["local_only"]),
        tuple(data["forbidden"]),
    )


def upstream_text(upstream: Path) -> str:
    """上游规范源：SKILL.md 正文 + rules-full.md（存在才读）。"""
    parts = [(upstream / "SKILL.md").read_text(encoding="utf-8")]
    full = upstream / "references" / "rules-full.md"
    if full.is_file():
        parts.append(full.read_text(encoding="utf-8"))
    return "\n".join(parts)


def check(upstream: Path, lite_core: str) -> list[str]:
    tags, anchors, local_only, forbidden = load_anchors()
    upstream_body = upstream_text(upstream)
    return [
        *(f"上游已无 {tag} 条" for tag in tags if f"- {tag} " not in upstream_body),
        *(f"锚点不在本地规则：{anchor}" for anchor in anchors if anchor not in lite_core),
        *(f"本地专有锚点丢失：{anchor}" for anchor in local_only if anchor not in lite_core),
        *(f"脱节残留词渗入本地规则：{phrase}" for phrase in forbidden if phrase in lite_core),
        *(f"锚点已不在上游：{anchor}" for anchor in anchors if anchor not in upstream_body),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check drift against a natural-talk checkout.")
    parser.add_argument("--upstream", required=True, help="path to a natural-talk checkout")
    args = parser.parse_args(argv)

    upstream = Path(args.upstream)
    if not (upstream / "SKILL.md").is_file():
        print(f"upstream SKILL.md missing: {upstream / 'SKILL.md'}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(REPO))
    from tests._support import ensure_plugin_package

    ensure_plugin_package()
    from astrbot_plugin_natural_chat.rules import _LITE_CORE

    if _LITE_CORE != LITE_FIXTURE_PATH.read_text(encoding="utf-8"):
        print("rules._LITE_CORE 与 tests/fixtures/system-prompt-lite.txt 不同步", file=sys.stderr)
        return 1

    problems = check(upstream, _LITE_CORE)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("upstream sync: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
