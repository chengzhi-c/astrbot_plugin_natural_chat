"""Host-independent Core behavior and configuration contracts."""

import asyncio
import json
import os
import unittest
from unittest import mock

from tests._support import (
    V2_RULES_E4AA983,
    V5_RULES_B46BD0D,
    FakePart,
    FakeReq,
    ensure_plugin_package,
    temporary_directory,
)

ensure_plugin_package()

from astrbot_plugin_human_chat_quality import core as core_module
from astrbot_plugin_human_chat_quality.constants import MAX_RUNTIME_HINT_CHARS
from astrbot_plugin_human_chat_quality.core import AppConfig, HumanChatQualityCore
from astrbot_plugin_human_chat_quality.quality_rules import (
    RUNTIME_HINT_MARKER,
    STABLE_RULE_MARKER,
    build_runtime_hint,
    build_stable_rules,
)
from astrbot_plugin_human_chat_quality.runtime_state import RuntimeStateStore, SessionState


class FakeEvent:
    def __init__(self, origin, text=""):
        self.unified_msg_origin = origin
        self.text = text


class FakeLLMResp:
    def __init__(self, text):
        self.completion_text = text
        self.result_chain = None


class TestCoreFlow(unittest.TestCase):
    def setUp(self):
        self.dir = temporary_directory(self)
        self.store = RuntimeStateStore(os.path.join(self.dir, "s.json"), 14, 8, ())
        self.core = HumanChatQualityCore(AppConfig.from_config(None), self.store, text_part_factory=FakePart)
        self.ev = FakeEvent("aiocqhttp:GroupMessage:111")

    def test_legacy_block_is_preserved_as_ordinary_text(self):
        """3.0.0 破坏性变更：旧规则块不再被剥离，按普通文本保留。"""
        req = FakeReq()
        req.system_prompt = f"原人设：你是XX\n\n{V5_RULES_B46BD0D}"
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertIn("[Human Chat Quality Rules v5]", req.system_prompt)
        self.assertEqual(req.system_prompt.count(STABLE_RULE_MARKER), 1)  # v9 正常注入
        self.assertTrue(req.system_prompt.startswith("原人设：你是XX"))

    def test_current_stable_rules_injected(self):
        req = FakeReq()
        req.contexts = [{"role": "user", "content": [{"type": "text", "text": "在吗"}]}]
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertIn(STABLE_RULE_MARKER, req.system_prompt)
        self.assertEqual(req.system_prompt.count(STABLE_RULE_MARKER), 1)
        self.assertIn("natural-talk", req.system_prompt)

    def test_legacy_v2_block_preserved_in_history(self):
        """3.0.0：旧规则块在历史中按普通文本保留，不再清扫。"""
        req = FakeReq()
        req.contexts = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "原话"},
                    {"type": "text", "text": V2_RULES_E4AA983},
                ],
            }
        ]
        asyncio.run(self.core.on_llm_request(self.ev, req))
        texts = [p.get("text", "") for ctx in req.contexts for p in ctx["content"]]
        self.assertEqual(texts, ["原话", V2_RULES_E4AA983])  # 用户原话与旧块都保留
        self.assertIn(STABLE_RULE_MARKER, req.system_prompt)  # 稳定规则正常注入

    def test_no_hint_first_round_then_hint_after_three_repeats(self):
        for _ in range(3):
            req = FakeReq()
            asyncio.run(self.core.on_llm_request(self.ev, req))
            asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("好的，回答")))
        req = FakeReq()
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.assertIn(RUNTIME_HINT_MARKER, req.extra_user_content_parts[0].text)
        self.assertIn("好的", req.extra_user_content_parts[0].text)

    def test_stale_extra_part_is_replaced_by_factory_product_not_dict(self):
        for _ in range(3):
            req = FakeReq()
            asyncio.run(self.core.on_llm_request(self.ev, req))
            asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("好的，回答")))
        old_hint = build_runtime_hint(["旧开头"], MAX_RUNTIME_HINT_CHARS)
        req = FakeReq()
        req.extra_user_content_parts = [FakePart(old_hint)]
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertEqual(len(req.extra_user_content_parts), 1)
        part = req.extra_user_content_parts[0]
        self.assertIsInstance(part, FakePart)
        self.assertFalse(isinstance(part, dict))
        self.assertIn("好的", part.text)
        self.assertNotIn("旧开头", part.text)

    def test_replace_in_history_no_accumulation(self):
        for _ in range(3):
            req = FakeReq()
            asyncio.run(self.core.on_llm_request(self.ev, req))
            asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("好的，回答")))
        req = FakeReq()
        old_hint = build_runtime_hint(
            self.store.get(self.ev.unified_msg_origin).avoid_openers, MAX_RUNTIME_HINT_CHARS
        ).replace("好的", "旧开头")
        req.contexts = [{"role": "user", "content": [{"type": "text", "text": old_hint}]}]
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.assertIn("好的", req.extra_user_content_parts[0].text)
        self.assertEqual(req.contexts[0]["content"], [])
        self.assertNotIn("旧开头", json.dumps(req.contexts, ensure_ascii=False))

    def test_global_off_no_inject(self):
        core_off = HumanChatQualityCore(
            AppConfig.from_config({"enabled": False}), self.store, text_part_factory=FakePart
        )
        req = FakeReq()
        asyncio.run(core_off.on_llm_request(self.ev, req))
        self.assertNotIn(STABLE_RULE_MARKER, req.system_prompt)

    def test_blacklist_hit_and_miss(self):
        core = HumanChatQualityCore(
            AppConfig.from_config({"disabled_sessions": ["222"]}), self.store, text_part_factory=FakePart
        )
        req_hit = FakeReq()
        asyncio.run(core.on_llm_request(FakeEvent("aiocqhttp:GroupMessage:222"), req_hit))
        self.assertNotIn(STABLE_RULE_MARKER, req_hit.system_prompt)
        req_miss = FakeReq()
        asyncio.run(core.on_llm_request(FakeEvent("aiocqhttp:GroupMessage:333"), req_miss))
        self.assertIn(STABLE_RULE_MARKER, req_miss.system_prompt)

    def test_status_text_active_and_inactive(self):
        text_active = self.core.status_text(self.ev.unified_msg_origin, self.ev)
        self.assertIn("启用", text_active)
        self.assertIn("下一轮避用：无（尚未形成重复或套话信号）", text_active)
        self.assertNotIn("下一轮请求会带上动态提醒", text_active)
        core_off = HumanChatQualityCore(
            AppConfig.from_config({"enabled": False}), self.store, text_part_factory=FakePart
        )
        self.assertIn("关闭", core_off.status_text(self.ev.unified_msg_origin, self.ev))

    def test_status_text_names_next_round_hint(self):
        for _ in range(3):
            asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("好的，回答")))
        text = self.core.status_text(self.ev.unified_msg_origin, self.ev)
        self.assertIn("下一轮避用：好的", text)
        self.assertIn("下一轮请求会带上动态提醒", text)
        core_no_hint = HumanChatQualityCore(
            AppConfig.from_config({"inject_runtime_state": False}), self.store, text_part_factory=FakePart
        )
        quiet = core_no_hint.status_text(self.ev.unified_msg_origin, self.ev)
        self.assertIn("下一轮避用：好的", quiet)
        self.assertNotIn("下一轮请求会带上动态提醒", quiet)

    def test_status_distinguishes_global_static_and_session_disable(self):
        global_off = HumanChatQualityCore(
            AppConfig.from_config({"enabled": False}), self.store, text_part_factory=FakePart
        ).status_text(self.ev.unified_msg_origin, self.ev)
        self.assertIn("全局配置：关闭", global_off)

        static_off = HumanChatQualityCore(
            AppConfig.from_config({"disabled_sessions": ["111"]}), self.store, text_part_factory=FakePart
        ).status_text(self.ev.unified_msg_origin, self.ev)
        self.assertIn("配置静态禁用", static_off)

        asyncio.run(self.core.set_session_enabled(self.ev.unified_msg_origin, False))
        session_off = self.core.status_text(self.ev.unified_msg_origin, self.ev)
        self.assertIn("/humanq off", session_off)

    def test_status_reports_runtime_capability_and_invalid_config_summary(self):
        store = RuntimeStateStore(self.dir + "-status.json", 14, 8, ["", "词", "词", "x" * 21])
        core = HumanChatQualityCore(AppConfig.from_config(None), store, text_part_factory=None)
        text = core.status_text(self.ev.unified_msg_origin, self.ev)
        self.assertIn("宿主临时文本部件不可用", text)
        self.assertIn("配置忽略：3 项", text)
        self.assertIn("空值 1 项、重复 1 项、过长 1 项", text)
        self.assertNotIn("x" * 21, text)


class TestCoreFlowExtra(unittest.TestCase):
    """Core cleanup, counting, and response-recording contracts."""

    def setUp(self):
        self.dir = temporary_directory(self)
        self.store = RuntimeStateStore(os.path.join(self.dir, "s.json"), 14, 8, ())
        self.core = HumanChatQualityCore(AppConfig.from_config(None), self.store, text_part_factory=FakePart)
        self.ev = FakeEvent("aiocqhttp:GroupMessage:111")

    def test_overlong_custom_cliche_filtered_end_to_end(self):
        # 超长自定义词在 Store 构造期被过滤，Core 全流程不入 avoid_openers
        store = RuntimeStateStore(os.path.join(self.dir, "s2.json"), 14, 8, ("x" * 21,))
        core = HumanChatQualityCore(AppConfig.from_config(None), store, text_part_factory=FakePart)
        asyncio.run(core.on_llm_response(self.ev, FakeLLMResp("这是" + "x" * 21 + "的回复")))
        self.assertEqual(store.get(self.ev.unified_msg_origin).avoid_openers, [])

    def test_session_off_stops_inject_and_record(self):
        asyncio.run(self.core.set_session_enabled(self.ev.unified_msg_origin, False))
        req = FakeReq()
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertNotIn(STABLE_RULE_MARKER, req.system_prompt)
        asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("好的，回答")))
        self.assertEqual(self.store.get(self.ev.unified_msg_origin).recent_openers, [])

    def _request_with_owned_blocks(self):
        req = FakeReq()
        req.system_prompt = f"原人设\n\n{build_stable_rules()}"
        runtime = build_runtime_hint(["旧开头"], MAX_RUNTIME_HINT_CHARS)
        req.contexts = [{"role": "user", "content": [{"type": "text", "text": runtime}]}]
        return req

    def test_global_off_cleans_owned_history_without_counting_injection(self):
        core = HumanChatQualityCore(AppConfig.from_config({"enabled": False}), self.store, text_part_factory=FakePart)
        req = self._request_with_owned_blocks()
        asyncio.run(core.on_llm_request(self.ev, req))
        self.assertEqual(req.system_prompt, "原人设")
        self.assertEqual(req.contexts[0]["content"], [])
        self.assertEqual(core.stats.total_injections, 0)

    def test_static_disabled_session_cleans_owned_history(self):
        core = HumanChatQualityCore(
            AppConfig.from_config({"disabled_sessions": ["111"]}), self.store, text_part_factory=FakePart
        )
        req = self._request_with_owned_blocks()
        asyncio.run(core.on_llm_request(self.ev, req))
        self.assertEqual(req.system_prompt, "原人设")
        self.assertEqual(req.contexts[0]["content"], [])

    def test_session_off_cleans_owned_history(self):
        asyncio.run(self.core.set_session_enabled(self.ev.unified_msg_origin, False))
        req = self._request_with_owned_blocks()
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertEqual(req.system_prompt, "原人设")
        self.assertEqual(req.contexts[0]["content"], [])

    def test_no_origin_cleans_owned_history_without_state_or_injection(self):
        req = self._request_with_owned_blocks()
        asyncio.run(self.core.on_llm_request(FakeEvent(""), req))
        self.assertEqual(req.system_prompt, "原人设")
        self.assertEqual(req.contexts[0]["content"], [])
        self.assertEqual(self.store.sessions, {})
        self.assertEqual(self.core.stats.total_injections, 0)

    def test_runtime_config_off_removes_runtime_but_keeps_stable_rules(self):
        core = HumanChatQualityCore(
            AppConfig.from_config({"inject_runtime_state": False}), self.store, text_part_factory=FakePart
        )
        req = self._request_with_owned_blocks()
        asyncio.run(core.on_llm_request(self.ev, req))
        self.assertEqual(req.contexts[0]["content"], [])
        self.assertIn(STABLE_RULE_MARKER, req.system_prompt)

    def test_stable_config_off_removes_stable_rules_but_runtime_stays_active(self):
        asyncio.run(self.store.record_response(self.ev.unified_msg_origin, "好的，回答一"))
        asyncio.run(self.store.record_response(self.ev.unified_msg_origin, "好的，回答二"))
        asyncio.run(self.store.record_response(self.ev.unified_msg_origin, "好的，回答三"))
        core = HumanChatQualityCore(
            AppConfig.from_config({"inject_stable_rules": False}), self.store, text_part_factory=FakePart
        )
        req = self._request_with_owned_blocks()
        asyncio.run(core.on_llm_request(self.ev, req))
        self.assertEqual(req.system_prompt, "原人设")
        self.assertEqual(req.contexts[0]["content"], [])
        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.assertIn(RUNTIME_HINT_MARKER, req.extra_user_content_parts[0].text)

    def test_missing_text_part_factory_still_cleans_history_without_fake_part(self):
        asyncio.run(self.store.record_response(self.ev.unified_msg_origin, "好的，回答一"))
        asyncio.run(self.store.record_response(self.ev.unified_msg_origin, "好的，回答二"))
        asyncio.run(self.store.record_response(self.ev.unified_msg_origin, "好的，回答三"))
        core = HumanChatQualityCore(AppConfig.from_config(None), self.store, text_part_factory=None)
        req = FakeReq()
        req.contexts = [{"role": "user", "content": [{"type": "text", "text": "在吗"}]}]
        asyncio.run(core.on_llm_request(self.ev, req))
        self.assertEqual(req.contexts[0]["content"], [{"type": "text", "text": "在吗"}])
        self.assertEqual(req.extra_user_content_parts, [])
        self.assertIn(STABLE_RULE_MARKER, req.system_prompt)

    def test_missing_text_part_factory_removes_existing_runtime_hint(self):
        for _ in range(3):
            asyncio.run(self.store.record_response(self.ev.unified_msg_origin, "好的，回答"))
        core = HumanChatQualityCore(AppConfig.from_config(None), self.store, text_part_factory=None)
        old_hint = build_runtime_hint(["旧开头"], MAX_RUNTIME_HINT_CHARS)
        req = FakeReq()
        req.contexts = [{"role": "user", "content": [{"type": "text", "text": old_hint}]}]

        asyncio.run(core.on_llm_request(self.ev, req))

        self.assertEqual(req.contexts[0]["content"], [])
        self.assertEqual(req.extra_user_content_parts, [])

    def test_total_injections_only_real_injections(self):
        req = FakeReq()
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertEqual(self.core.stats.total_injections, 1)
        # 幂等轮：同一 req 已含规则、无 hint → 不注入不计
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertEqual(self.core.stats.total_injections, 1)
        # 仅移除历史旧块 → 不计注入
        req2 = FakeReq()
        req2.system_prompt = req.system_prompt
        req2.contexts = [{"role": "user", "content": [{"type": "text", "text": RUNTIME_HINT_MARKER + "\n旧"}]}]
        asyncio.run(self.core.on_llm_request(self.ev, req2))
        self.assertEqual(self.core.stats.total_injections, 1)

    def test_response_signals_are_detected_once(self):
        core_detect = core_module.detect_cliches
        with mock.patch.object(core_module, "detect_cliches", wraps=core_detect) as core_mock:
            asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("好问题，回答")))

        # 检测只在 core 发生一次；store 只做合并，不再有任何检测调用
        self.assertEqual(core_mock.call_count, 1)

    def test_cleanup_stats_count_all_removed_blocks(self):
        runtime = build_runtime_hint(["旧开头"], MAX_RUNTIME_HINT_CHARS)
        req = FakeReq(system_prompt=f"{build_stable_rules()}\n\n{build_stable_rules()}")
        req.contexts = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": runtime},
                    {"type": "text", "text": runtime},
                ],
            }
        ]

        asyncio.run(self.core.on_llm_request(FakeEvent(""), req))

        self.assertEqual(self.core.stats.legacy_blocks_removed, 2)
        self.assertEqual(self.core.stats.stale_hints_removed, 2)

    def test_runtime_hint_missed_counts_repetition_after_hint(self):
        for _ in range(3):
            req = FakeReq()
            asyncio.run(self.core.on_llm_request(self.ev, req))
            asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("好的，回答")))
        # 第四轮请求注入提醒（avoid_openers=["好的"]），回复仍用同一开头 → 计一次忽略
        req = FakeReq()
        asyncio.run(self.core.on_llm_request(self.ev, req))
        asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("好的，还在重复")))
        self.assertEqual(self.core.stats.runtime_hint_missed, 1)
        # 下一轮换了开头，不再计数
        req = FakeReq()
        asyncio.run(self.core.on_llm_request(self.ev, req))
        asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("换了个自然开头")))
        self.assertEqual(self.core.stats.runtime_hint_missed, 1)

    def test_runtime_hint_missed_not_counted_without_hint(self):
        for _ in range(3):
            req = FakeReq()
            asyncio.run(self.core.on_llm_request(self.ev, req))
            asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("好的，回答")))
        # 无提醒注入的响应（如另一会话）：不计数
        asyncio.run(self.core.on_llm_response(FakeEvent("aiocqhttp:GroupMessage:999"), FakeLLMResp("好的，重复")))
        self.assertEqual(self.core.stats.runtime_hint_missed, 0)

    def test_formal_writing_with_emotion_topic_skips_request_and_response(self):
        event = FakeEvent(self.ev.unified_msg_origin, "写一篇关于抑郁的论文")
        req = FakeReq()

        asyncio.run(self.core.on_llm_request(event, req))
        asyncio.run(self.core.on_llm_response(event, FakeLLMResp("好的，论文草稿")))

        self.assertNotIn(STABLE_RULE_MARKER, req.system_prompt)
        self.assertNotIn(event.unified_msg_origin, self.store.sessions)

    def test_discussing_a_formal_topic_keeps_chat_quality_active(self):
        event = FakeEvent(self.ev.unified_msg_origin, "分析一下关于抑郁论文的观点")
        req = FakeReq()

        asyncio.run(self.core.on_llm_request(event, req))

        self.assertIn(STABLE_RULE_MARKER, req.system_prompt)

    def test_runtime_hint_missed_only_checks_items_that_were_injected(self):
        first = "第一项第一项第一项第一项第一项"
        second = "第二项第二项第二项第二项第二项"
        self.store.sessions[self.ev.unified_msg_origin] = SessionState(avoid_openers=[first, second])
        core = HumanChatQualityCore(
            AppConfig.from_config({"max_runtime_hint_chars": 80}), self.store, text_part_factory=FakePart
        )
        req = FakeReq()

        asyncio.run(core.on_llm_request(self.ev, req))
        asyncio.run(core.on_llm_response(self.ev, FakeLLMResp(second)))

        self.assertIn(first, req.extra_user_content_parts[0].text)
        self.assertNotIn(second, req.extra_user_content_parts[0].text)
        self.assertEqual(core.stats.runtime_hint_missed, 0)

    def test_runtime_hint_tracking_is_fifo_per_session(self):
        self.store.sessions[self.ev.unified_msg_origin] = SessionState(avoid_openers=["第一项"])
        asyncio.run(self.core.on_llm_request(self.ev, FakeReq()))
        self.store.sessions[self.ev.unified_msg_origin] = SessionState(avoid_openers=["第二项"])
        asyncio.run(self.core.on_llm_request(self.ev, FakeReq()))

        asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("第二项")))
        self.assertEqual(self.core.stats.runtime_hint_missed, 0)
        asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("第二项")))
        self.assertEqual(self.core.stats.runtime_hint_missed, 1)

    def test_pending_hint_queue_is_bounded_per_session(self):
        limit = core_module.PENDING_HINT_MAX_PER_SESSION
        for index in range(limit + 3):
            self.store.sessions[self.ev.unified_msg_origin] = SessionState(avoid_openers=[f"第{index}项"])
            asyncio.run(self.core.on_llm_request(self.ev, FakeReq()))

        pending = self.core._pending_hints[self.ev.unified_msg_origin]
        self.assertEqual(pending.maxlen, limit)
        self.assertEqual(len(pending), limit)

    def test_expired_pending_hint_does_not_count_as_missed(self):
        self.store.sessions[self.ev.unified_msg_origin] = SessionState(avoid_openers=["旧项"])
        with (
            mock.patch.object(core_module, "PENDING_HINT_TTL_SECONDS", 10, create=True),
            mock.patch.object(core_module, "time", create=True) as clock,
        ):
            clock.monotonic.side_effect = [0, 11]
            asyncio.run(self.core.on_llm_request(self.ev, FakeReq()))
            asyncio.run(self.core.on_llm_response(self.ev, FakeLLMResp("旧项")))

        self.assertEqual(self.core.stats.runtime_hint_missed, 0)

    def test_no_origin_skips_everything(self):
        ev = FakeEvent("")
        req = FakeReq()
        asyncio.run(self.core.on_llm_request(ev, req))
        self.assertEqual(req.system_prompt, "原人设：你是XX")
        asyncio.run(self.core.on_llm_response(ev, FakeLLMResp("好的，回答")))
        self.assertEqual(self.store.sessions, {})

    def test_unknown_stable_string_is_preserved_across_requests(self):
        req = FakeReq()
        req.contexts = [{"role": "user", "content": "[Human Chat Quality Rules v2]\n旧规则块"}]
        asyncio.run(self.core.on_llm_request(self.ev, req))
        asyncio.run(self.core.on_llm_request(self.ev, req))
        self.assertEqual(req.contexts[0]["content"], "[Human Chat Quality Rules v2]\n旧规则块")

    def test_debug_log_reports_ambiguous_owned_markers_kept(self):
        core = HumanChatQualityCore(AppConfig.from_config({"debug_log": True}), self.store, text_part_factory=FakePart)
        req = FakeReq()
        req.system_prompt = "[Human Chat Quality Rules v3]\n未知规则"
        req.contexts = [{"role": "user", "content": RUNTIME_HINT_MARKER + "\n未知提示"}]
        with self.assertLogs("astrbot_plugin_human_chat_quality.core", level="DEBUG") as logs:
            asyncio.run(core.on_llm_request(self.ev, req))
        self.assertTrue(any("ambiguous" in line for line in logs.output))

    def test_command_result_and_status_expose_pending_persistence(self):
        async def run():
            with mock.patch.object(self.store, "_write_snapshot_sync", side_effect=OSError("disk full")):
                result = await self.core.set_session_enabled(self.ev.unified_msg_origin, False)
            self.assertFalse(result)
            self.assertFalse(self.store.is_enabled(self.ev.unified_msg_origin))
            self.assertIn("待重试", self.core.status_text(self.ev.unified_msg_origin, self.ev))

        from unittest import mock

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
