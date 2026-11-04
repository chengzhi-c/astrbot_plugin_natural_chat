"""发布与运行脚本的门禁行为。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import build_release
from tests._support import ROOT


def _copy_runtime_sources(target: Path) -> None:
    for relative in build_release.RUNTIME_MANIFEST:
        source = ROOT / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


class ValidateReleaseTest(unittest.TestCase):
    def make_repo(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        repo = Path(temp_dir.name) / "plugin"
        _copy_runtime_sources(repo)
        return repo

    def test_current_tree_validates(self) -> None:
        info = build_release.validate_release(ROOT)
        self.assertEqual(info.name, "astrbot_plugin_natural_chat")
        self.assertNotEqual(info.version, "0.0.0")

    def test_missing_file_is_rejected(self) -> None:
        repo = self.make_repo()
        (repo / "core.py").unlink()
        with self.assertRaises(ValueError):
            build_release.validate_release(repo)

    def test_version_mismatch_is_rejected(self) -> None:
        repo = self.make_repo()
        metadata = (repo / "metadata.yaml").read_text(encoding="utf-8")
        (repo / "metadata.yaml").write_text(metadata.replace("version: ", "version: 9.9.9-", 1), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_release.validate_release(repo)

    def test_archive_contains_exactly_the_runtime_manifest(self) -> None:
        repo = self.make_repo()
        out_dir = repo.parent / "dist"
        archive = build_release.build_archive(repo, out_dir)
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        expected = {f"{build_release.validate_release(repo).name}/{item}" for item in build_release.RUNTIME_MANIFEST}
        self.assertEqual(names, expected)


class UpstreamFixtureTest(unittest.TestCase):
    def test_lite_core_matches_the_upstream_fixture(self) -> None:
        """规则正文与上游同步 fixture 必须逐字一致；check_upstream_sync.py 靠这条配对判断漂移。"""
        from astrbot_plugin_natural_chat.rules import _LITE_CORE

        fixture = ROOT / "tests" / "fixtures" / "system-prompt-lite.txt"
        self.assertEqual(_LITE_CORE, fixture.read_text(encoding="utf-8"))


class RunTestsTest(unittest.TestCase):
    def test_usage_error_exits_with_two(self) -> None:
        result = subprocess.run(
            [sys.executable, "-S", "scripts/run_tests.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 2)

    def test_suite_names_are_declared(self) -> None:
        from scripts import run_tests

        self.assertEqual(set(run_tests.SUITES), {"core", "host"})


if __name__ == "__main__":
    unittest.main()
