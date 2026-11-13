#!/usr/bin/env python3
"""重新生成 rules._LEGACY_STABLE_SIGNATURES。

旧规则块是各发布版本固定生成的字符串，剥离侧靠 (行数, SHA-256) 精确匹配，
这样既不会误删用户改过的块，也不会因为找锚点猜边界而连带删掉人设。
发布新版本或发现遗漏的历史版本时，跑一次本脚本并把输出贴回 rules.py。
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# 曾向 system_prompt 写入过规则块的 tag；只列真正发布过的版本。
TAGS = ("0.2.0", "0.5.2", "0.5.3", "0.6.2", "0.8.0", "1.2.0", "1.3.0", "2.1.0", "2.3.0", "3.3.0", "3.10.1")


def _stub_host() -> None:
    import logging
    import types

    package = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot")
    package.api = api
    sys.modules["astrbot"] = package
    sys.modules["astrbot.api"] = api


def _load_block(tag: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", tag], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
        root = Path(tmp)
        package = root / "astrbot_plugin_natural_chat"
        package.mkdir()
        for item in root.iterdir():
            if item.is_file():
                shutil.copy2(item, package / item.name)
        (package / "__init__.py").write_text("", encoding="utf-8")
        sys.path.insert(0, str(root))
        for name in [name for name in sys.modules if name.startswith("astrbot_plugin_natural_chat")]:
            del sys.modules[name]
        try:
            from astrbot_plugin_natural_chat.quality_rules import build_stable_rules

            return build_stable_rules().replace("\r\n", "\n")
        finally:
            sys.path.remove(str(root))
            for name in [name for name in sys.modules if name.startswith("astrbot_plugin_natural_chat")]:
                del sys.modules[name]


def main() -> int:
    _stub_host()
    for tag in TAGS:
        text = _load_block(tag)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(f'    ({len(text.splitlines())}, "{digest}"),  # {tag}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
