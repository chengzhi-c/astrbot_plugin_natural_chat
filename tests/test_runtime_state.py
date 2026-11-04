"""runtime_state 的合并口径、持久化与损伤恢复。"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from tests._support import FakeEvent, ensure_plugin_package, stub_astrbot_logger

stub_astrbot_logger()
ensure_plugin_package()

from astrbot_plugin_natural_chat.constants import MAX_AVOID_ITEMS
from astrbot_plugin_natural_chat.runtime_state import (
    RuntimeStateStore,
    extract_opener,
    is_session_disabled,
    match_keys,
    repeated_items,
)


class StoreCase(unittest.IsolatedAsyncioTestCase):
    def make_store(self, **kwargs: object) -> RuntimeStateStore:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        options: dict[str, object] = {"retention_days": 14, "recent_reply_window": 8}
        options.update(kwargs)
        return RuntimeStateStore(Path(temp_dir.name) / "state.json", **options)  # type: ignore[arg-type]


class RecordResponseTest(StoreCase):
    async def test_signals_enter_the_avoid_list(self) -> None:
        store = self.make_store()
        added = await store.record_response("s1", "然而连发。然而又是。", ("然而连发",))
        self.assertEqual(added, 1)
        self.assertEqual(store.get("s1").avoid_openers, ["然而连发"])

    async def test_blank_text_records_nothing(self) -> None:
        store = self.make_store()
        self.assertEqual(await store.record_response("s1", "   "), 0)
        self.assertEqual(store.get("s1").avoid_openers, [])

    async def test_avoid_list_is_capped_and_deduplicated(self) -> None:
        store = self.make_store()
        names = [f"信号{i}" for i in range(MAX_AVOID_ITEMS + 4)]
        await store.record_response("s1", "正文", tuple(names))
        self.assertEqual(len(store.get("s1").avoid_openers), MAX_AVOID_ITEMS)
        await store.record_response("s1", "正文", (names[0],))
        self.assertEqual(store.get("s1").avoid_openers.count(names[0]), 1)

    async def test_overlong_signal_is_dropped(self) -> None:
        store = self.make_store()
        await store.record_response("s1", "正文", ("长" * 40,))
        self.assertEqual(store.get("s1").avoid_openers, [])

    async def test_repeated_opener_enters_after_threshold(self) -> None:
        store = self.make_store()
        for _ in range(3):
            await store.record_response("s1", "好的，这就办。")
        self.assertIn("好的", store.get("s1").avoid_openers)


class EnableStateTest(StoreCase):
    async def test_toggle_round_trip(self) -> None:
        store = self.make_store()
        self.assertTrue(store.is_enabled("s1"))
        await store.set_enabled("s1", False)
        self.assertFalse(store.is_enabled("s1"))
        await store.set_enabled("s1", True)
        self.assertTrue(store.is_enabled("s1"))

    async def test_reset_clears_records(self) -> None:
        store = self.make_store()
        await store.record_response("s1", "然而连发。然而又是。", ("然而连发",))
        await store.reset("s1")
        self.assertEqual(store.get("s1").avoid_openers, [])


class PersistenceTest(StoreCase):
    async def test_state_survives_restart(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "state.json"
        first = RuntimeStateStore(path, retention_days=14, recent_reply_window=8)
        await first.record_response("s1", "然而连发。", ("然而连发",))
        await first.set_enabled("s2", False)
        await first.terminate()

        second = RuntimeStateStore(path, retention_days=14, recent_reply_window=8)
        self.assertEqual(second.get("s1").avoid_openers, ["然而连发"])
        self.assertFalse(second.is_enabled("s2"))

    async def test_corrupt_file_is_backed_up_and_reset(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "state.json"
        path.write_text("{not json", encoding="utf-8")
        store = RuntimeStateStore(path, retention_days=14, recent_reply_window=8)
        self.assertEqual(store.sessions, {})
        self.assertTrue(list(path.parent.glob("state.corrupt.*.json")))

    async def test_bad_session_entry_is_skipped_others_survive(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "state.json"
        now = int(time.time())
        path.write_text(
            json.dumps({"sessions": {"good": {"a": ["好的"], "r": "", "t": now}, "bad": "not-a-dict"}}),
            encoding="utf-8",
        )
        store = RuntimeStateStore(path, retention_days=14, recent_reply_window=8)
        self.assertIn("good", store.sessions)
        self.assertNotIn("bad", store.sessions)

    async def test_non_finite_timestamp_is_ignored(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "state.json"
        path.write_text(json.dumps({"sessions": {"s1": {"a": [], "r": "", "t": "nan"}}}), encoding="utf-8")
        store = RuntimeStateStore(path, retention_days=1, recent_reply_window=8)
        # 时间戳不可解析时回落到文件 mtime，否则整条会话会被当成过期清掉
        self.assertIsNotNone(store.get("s1").updated_at)

    async def test_expired_session_is_pruned(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "state.json"
        path.write_text(json.dumps({"sessions": {"old": {"a": ["好的"], "r": "", "t": 1}}}), encoding="utf-8")
        store = RuntimeStateStore(path, retention_days=1, recent_reply_window=8)
        self.assertNotIn("old", store.sessions)

    async def test_write_failure_is_reported_not_raised(self) -> None:
        store = self.make_store()
        store._write_snapshot_sync = lambda payload: (_ for _ in ()).throw(OSError("disk full"))  # type: ignore[method-assign]
        await store.record_response("s1", "然而连发。", ("然而连发",))
        self.assertFalse(await store.flush())
        self.assertTrue(store.has_pending_save)


class CustomClicheTest(StoreCase):
    def test_invalid_entries_are_counted_by_reason(self) -> None:
        store = self.make_store(custom_cliches=["正常", "  ", "正常", "长" * 40])
        self.assertEqual(store.custom_cliches, ("正常",))
        reasons = dict(store.custom_cliches_ignored_reasons)
        self.assertEqual(reasons, {"empty": 1, "too_long": 1, "duplicate": 1})

    async def test_custom_word_is_persisted_as_a_tag(self) -> None:
        """自定义词可能敏感，落盘只留序号与指纹，原文不写进文件。"""
        store = self.make_store(custom_cliches=["暗号是蓝色"])
        await store.record_response("s1", "正文", ("暗号是蓝色",))
        await store.flush()
        raw = store.state_path.read_text(encoding="utf-8")
        self.assertNotIn("暗号是蓝色", raw)
        self.assertEqual(store.resolve_name(store.get("s1").avoid_openers[0]), "暗号是蓝色")

    async def test_reordered_config_does_not_swap_words(self) -> None:
        """配置改序后旧序号会指向另一个词，指纹对不上必须丢弃，不能张冠李戴。"""
        store = self.make_store(custom_cliches=["啊这", "好的吧"])
        await store.record_response("s1", "正文", ("啊这",))
        await store.flush()
        tag = store.get("s1").avoid_openers[0]

        reordered = self.make_store(custom_cliches=["好的吧", "啊这"])
        reordered.sessions = store.sessions
        self.assertEqual(reordered.resolve_name(tag), "")

    async def test_removed_word_drops_the_tag(self) -> None:
        store = self.make_store(custom_cliches=["啊这"])
        await store.record_response("s1", "正文", ("啊这",))
        await store.flush()
        tag = store.get("s1").avoid_openers[0]

        emptied = self.make_store(custom_cliches=[])
        emptied.sessions = store.sessions
        self.assertEqual(emptied.resolve_name(tag), "")

    async def test_legacy_plaintext_entries_are_tagged_on_load(self) -> None:
        """升级用户的状态文件里存的是自定义词原文，加载后必须就地转成标签，否则敏感词一直留在文件里。"""
        store = self.make_store(custom_cliches=["暗号是蓝色"])
        store.state_path.write_text(
            '{"disabled_sessions": [], "sessions": {"s1": {"a": ["暗号是蓝色"], "r": "", "t": 4102444800}}}',
            encoding="utf-8",
        )
        reloaded = RuntimeStateStore(
            store.state_path, retention_days=365, recent_reply_window=8, custom_cliches=["暗号是蓝色"]
        )
        tagged = reloaded.get("s1").avoid_openers[0]
        self.assertNotIn("暗号是蓝色", tagged)
        self.assertEqual(reloaded.resolve_name(tagged), "暗号是蓝色")
        await reloaded.flush()
        self.assertNotIn("暗号是蓝色", store.state_path.read_text(encoding="utf-8"))


class OpenerTest(unittest.TestCase):
    def test_prefixes_win(self) -> None:
        self.assertEqual(extract_opener("好的，这就去办"), "好的")

    def test_long_clause_is_truncated(self) -> None:
        self.assertEqual(len(extract_opener("这个问题的答案需要先看日志再说")), 8)

    def test_single_char_is_ignored(self) -> None:
        self.assertEqual(extract_opener("我"), "")

    def test_repeated_items_respects_threshold(self) -> None:
        self.assertEqual(repeated_items(["a", "b", "a", "a"], limit=5), ["a"])


class DisabledMatchTest(unittest.TestCase):
    def test_group_keys_cover_common_forms(self) -> None:
        keys = match_keys("aiocqhttp:GroupMessage:42", "42")
        for expected in ("group:42", "groupmessage:42", "42"):
            self.assertIn(expected, keys)

    def test_session_is_matched_by_group(self) -> None:
        disabled = frozenset({"group:42"})
        self.assertTrue(is_session_disabled(disabled, "x:GroupMessage:42", FakeEvent(group_id="42")))
        self.assertFalse(is_session_disabled(disabled, "x:GroupMessage:7", FakeEvent(group_id="7")))


if __name__ == "__main__":
    asyncio.run(unittest.main())
