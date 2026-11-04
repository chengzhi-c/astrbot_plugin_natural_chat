"""core 的请求/响应链路：注入、让位、统计、状态文案。"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._support import FakeEvent, FakePart, FakeReq, FakeResp, ensure_plugin_package, stub_astrbot_logger

stub_astrbot_logger()
ensure_plugin_package()

from astrbot_plugin_natural_chat import rules as qr
from astrbot_plugin_natural_chat.constants import (
    DEFAULT_RECENT_REPLY_WINDOW,
    DEFAULT_STATE_RETENTION_DAYS,
    MAX_RECENT_REPLY_WINDOW,
    MAX_RUNTIME_HINT_CHARS,
    MAX_STATE_RETENTION_DAYS,
    MIN_RECENT_REPLY_WINDOW,
    MIN_RUNTIME_HINT_CHARS,
    MIN_STATE_RETENTION_DAYS,
    YIELD_STICKY_TTL_SECONDS,
)
from astrbot_plugin_natural_chat.core import AppConfig, NaturalChatCore, extract_response_text
from astrbot_plugin_natural_chat.runtime_state import RuntimeStateStore


class CoreCase(unittest.IsolatedAsyncioTestCase):
    def make_core(self, config: AppConfig | None = None, *, factory: object = FakePart) -> NaturalChatCore:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = RuntimeStateStore(Path(temp_dir.name) / "state.json", retention_days=14, recent_reply_window=8)
        return NaturalChatCore(config or AppConfig(), store, text_part_factory=factory)


class ConfigTest(unittest.TestCase):
    def test_defaults_are_complete(self) -> None:
        cfg = AppConfig.from_config({})
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.inject_stable_rules)
        self.assertEqual(cfg.custom_cliches, ())
        self.assertEqual(cfg.disabled_sessions, frozenset())

    def test_out_of_range_values_are_clamped(self) -> None:
        cfg = AppConfig.from_config({"max_runtime_hint_chars": 10_000, "recent_reply_window": -5})
        self.assertEqual(cfg.max_runtime_hint_chars, MAX_RUNTIME_HINT_CHARS)
        self.assertEqual(cfg.recent_reply_window, MIN_RECENT_REPLY_WINDOW)

    def test_bool_written_as_int_does_not_silently_clamp(self) -> None:
        cfg = AppConfig.from_config({"max_runtime_hint_chars": True})
        self.assertEqual(cfg.max_runtime_hint_chars, MAX_RUNTIME_HINT_CHARS)

    def test_string_booleans_are_parsed(self) -> None:
        self.assertFalse(AppConfig.from_config({"enabled": "false"}).enabled)
        self.assertTrue(AppConfig.from_config({"enabled": "yes"}).enabled)

    def test_list_fields_accept_newline_text(self) -> None:
        cfg = AppConfig.from_config({"custom_cliches": "甲\n乙\n\n", "disabled_sessions": ["Group:1"]})
        self.assertEqual(cfg.custom_cliches, ("甲", "乙"))
        self.assertEqual(cfg.disabled_sessions, frozenset({"group:1"}))

    def test_schema_mirrors_the_python_defaults_and_bounds(self) -> None:
        """_conf_schema.json 是 Python 常量的手抄副本，抄错会让面板给出越界默认值，这里锚定一致。"""
        schema = json.loads((Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8"))

        numeric = {
            "max_runtime_hint_chars": (MAX_RUNTIME_HINT_CHARS, MIN_RUNTIME_HINT_CHARS, MAX_RUNTIME_HINT_CHARS),
            "recent_reply_window": (DEFAULT_RECENT_REPLY_WINDOW, MIN_RECENT_REPLY_WINDOW, MAX_RECENT_REPLY_WINDOW),
            "state_retention_days": (DEFAULT_STATE_RETENTION_DAYS, MIN_STATE_RETENTION_DAYS, MAX_STATE_RETENTION_DAYS),
        }
        for key, (default, minimum, maximum) in numeric.items():
            with self.subTest(key=key):
                self.assertEqual(schema[key]["default"], default)
                self.assertEqual((schema[key]["slider"]["min"], schema[key]["slider"]["max"]), (minimum, maximum))

        bools = {"enabled": True, "inject_stable_rules": True, "inject_runtime_state": True, "debug_log": False}
        for key, default in bools.items():
            with self.subTest(key=key):
                self.assertEqual(schema[key]["type"], "bool")
                self.assertIs(schema[key]["default"], default)


class InjectTest(CoreCase):
    async def test_rules_go_into_system_prompt(self) -> None:
        core = self.make_core()
        req = FakeReq(system_prompt="原人设")
        await core.on_llm_request(FakeEvent("帮我看看这段代码"), req)
        self.assertTrue(req.system_prompt.startswith("原人设"))
        self.assertIn(qr.STABLE_RULE_MARKER, req.system_prompt)
        self.assertEqual(core.stats.total_injections, 1)

    async def test_missing_system_prompt_is_created(self) -> None:
        core = self.make_core()
        req = FakeReq(system_prompt=None)
        await core.on_llm_request(FakeEvent("帮我看看这段代码"), req)
        self.assertIsNone(req.system_prompt)

    async def test_global_disable_strips_rules_but_keeps_persona(self) -> None:
        core = self.make_core(AppConfig(enabled=False))
        req = FakeReq(system_prompt=f"原人设\n\n{qr.build_stable_rules()}")
        await core.on_llm_request(FakeEvent("在吗"), req)
        self.assertEqual(req.system_prompt, "原人设")

    async def test_session_disable_stops_injection(self) -> None:
        core = self.make_core()
        await core.set_session_enabled("test:GroupMessage:1", False)
        req = FakeReq()
        await core.on_llm_request(FakeEvent("在吗"), req)
        self.assertEqual(req.system_prompt, "原人设")

    async def test_runtime_hint_uses_avoid_list(self) -> None:
        core = self.make_core()
        await core.on_llm_response(FakeEvent("帮我看下", "test:GroupMessage:1"), FakeResp("作为AI我不能确定。"))
        req = FakeReq(parts=[])
        await core.on_llm_request(FakeEvent("接着说", "test:GroupMessage:1"), req)
        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.assertTrue(req.extra_user_content_parts[0].text.startswith(qr.RUNTIME_HINT_MARKER))

    async def test_no_hint_without_a_factory(self) -> None:
        core = self.make_core(factory=None)
        await core.on_llm_response(FakeEvent("帮我看下", "test:GroupMessage:1"), FakeResp("作为AI我不能确定。"))
        req = FakeReq(parts=[])
        await core.on_llm_request(FakeEvent("接着说", "test:GroupMessage:1"), req)
        self.assertEqual(req.extra_user_content_parts, [])


class YieldTest(CoreCase):
    async def test_tone_rewrite_suppresses_injection(self) -> None:
        core = self.make_core()
        req = FakeReq(system_prompt="原人设")
        await core.on_llm_request(FakeEvent("这段话帮我改成正式语气"), req)
        self.assertEqual(req.system_prompt, "原人设")

    async def test_yield_expires_after_ttl(self) -> None:
        """TTL 之外同一句续写口令不再让位；时间轴用 monotonic 注入，不依赖真实等待。"""
        core = self.make_core()
        with mock.patch("astrbot_plugin_natural_chat.core.time.monotonic", return_value=1000.0):
            await core.on_llm_request(FakeEvent("这段话帮我润色一下"), FakeReq())
            # 窗口内：续写口令仍然让位
            inside = FakeReq(system_prompt="原人设")
            await core.on_llm_request(FakeEvent("继续改"), inside)
            self.assertEqual(inside.system_prompt, "原人设")
        with mock.patch(
            "astrbot_plugin_natural_chat.core.time.monotonic",
            return_value=1000.0 + YIELD_STICKY_TTL_SECONDS + 1,
        ):
            expired = FakeReq(system_prompt="原人设")
            await core.on_llm_request(FakeEvent("继续改"), expired)
            self.assertIn(qr.STABLE_RULE_MARKER, expired.system_prompt or "")

    async def test_sticky_followup_keeps_the_window(self) -> None:
        core = self.make_core()
        await core.on_llm_request(FakeEvent("这段话帮我润色一下"), FakeReq())
        req = FakeReq(system_prompt="原人设")
        await core.on_llm_request(FakeEvent("继续改"), req)
        self.assertEqual(req.system_prompt, "原人设")

    async def test_plain_short_reply_does_not_extend_the_window(self) -> None:
        core = self.make_core()
        await core.on_llm_request(FakeEvent("这段话帮我润色一下"), FakeReq())
        req = FakeReq(system_prompt="原人设")
        await core.on_llm_request(FakeEvent("好的"), req)
        self.assertIn(qr.STABLE_RULE_MARKER, req.system_prompt or "")

    async def test_blank_event_text_does_not_extend_the_window(self) -> None:
        """纯图片/取不到正文的消息不是续写口令，不能在让位窗口内吞掉这一轮。"""

        class NoTextEvent:
            unified_msg_origin = "test:GroupMessage:1"

            def get_group_id(self) -> str:
                return "1"

        core = self.make_core()
        await core.on_llm_request(FakeEvent("这段话帮我润色一下"), FakeReq())
        req = FakeReq(system_prompt="原人设")
        await core.on_llm_request(NoTextEvent(), req)
        self.assertIn(qr.STABLE_RULE_MARKER, req.system_prompt or "")

    async def test_yield_sessions_are_bounded(self) -> None:
        core = self.make_core()
        for index in range(400):
            core._pending_yield[f"session-{index}"] = (float(index), "tone")
        core._evict_yield_if_needed()
        from astrbot_plugin_natural_chat.constants import PENDING_SESSION_CAP

        self.assertLessEqual(len(core._pending_yield), PENDING_SESSION_CAP)


class ResponseTest(CoreCase):
    async def test_signals_are_counted_and_recorded(self) -> None:
        core = self.make_core()
        await core.on_llm_response(FakeEvent("随便问问", "test:GroupMessage:1"), FakeResp("好问题，作为AI我不确定。"))
        self.assertEqual(core.stats.signal_hits.get("好问题"), 1)
        self.assertGreaterEqual(core.stats.avoid_openers_seen, 1)
        self.assertIn("作为AI", core.store.get("test:GroupMessage:1").avoid_openers)

    async def test_reliability_signals_survive_competition(self) -> None:
        core = self.make_core()
        response = "作为AI，好问题，希望能帮到你。" + "太好了！" * 4
        await core.on_llm_response(FakeEvent("随便问问", "test:GroupMessage:1"), FakeResp(response))
        self.assertIn("作为AI", core.store.get("test:GroupMessage:1").avoid_openers)

    async def test_session_disabled_skips_recording(self) -> None:
        core = self.make_core()
        await core.set_session_enabled("test:GroupMessage:1", False)
        await core.on_llm_response(FakeEvent("随便问问", "test:GroupMessage:1"), FakeResp("好问题，作为AI我不确定。"))
        self.assertEqual(core.stats.signal_hits, {})

    async def test_empty_response_is_ignored(self) -> None:
        core = self.make_core()
        await core.on_llm_response(FakeEvent("随便问问"), FakeResp(""))
        self.assertEqual(core.stats.signal_hits, {})


class StatusTextTest(CoreCase):
    def test_disabled_reasons_are_listed(self) -> None:
        core = self.make_core(AppConfig(enabled=False))
        text = core.status_text("test:GroupMessage:1", FakeEvent("在吗"))
        self.assertIn("全局配置：关闭", text)

    def test_enabled_state_reports_empty_avoid_list(self) -> None:
        core = self.make_core()
        text = core.status_text("test:GroupMessage:1", FakeEvent("在吗"))
        self.assertIn("当前会话：启用", text)
        self.assertIn("尚未形成", text)

    def test_config_ignored_entries_are_reported(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = RuntimeStateStore(
            Path(temp_dir.name) / "state.json", retention_days=14, recent_reply_window=8, custom_cliches=["", "长" * 40]
        )
        text = NaturalChatCore(AppConfig(), store, text_part_factory=FakePart).status_text("s", FakeEvent())
        self.assertIn("配置忽略：2 项", text)


class ExtractTextTest(unittest.TestCase):
    def test_completion_text_wins(self) -> None:
        self.assertEqual(extract_response_text(FakeResp("正文")), "正文")

    def test_chain_fallback(self) -> None:
        resp = FakeResp("链式正文", completion=False)
        self.assertEqual(extract_response_text(resp), "链式正文")


if __name__ == "__main__":
    asyncio.run(unittest.main())
