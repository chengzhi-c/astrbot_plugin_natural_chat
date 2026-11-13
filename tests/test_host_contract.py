import asyncio
import unittest
from pathlib import Path
from unittest import mock

from tests._support import ensure_plugin_package

ensure_plugin_package()

from astrbot.api.event.filter import PermissionType, PermissionTypeFilter
from astrbot.api.star import Star
from astrbot.core.star.star_handler import EventType, star_handlers_registry
from astrbot_plugin_natural_chat import main as main_module
from astrbot_plugin_natural_chat.main import NaturalChatPlugin, _version_from_lines


class TestVersionParse(unittest.TestCase):
    def test_version_from_lines(self):
        self.assertEqual(_version_from_lines(["name: x", "version: 1.2.3", ""]), "1.2.3")
        self.assertEqual(_version_from_lines(['version: "1.0.0"']), "1.0.0")
        self.assertEqual(_version_from_lines(["name: x"]), "0.0.0")
        self.assertEqual(_version_from_lines(["version:"]), "0.0.0")

    def test_undecodable_metadata_falls_back_to_default_version(self):
        from astrbot_plugin_natural_chat.main import _read_metadata_version

        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        with mock.patch("pathlib.Path.read_text", side_effect=error):
            self.assertEqual(_read_metadata_version(), "0.0.0")

    def test_plugin_id_matches_metadata_name(self):
        """PLUGIN_ID（数据目录依据）与 metadata.name（发布包名）必须一致。"""
        from astrbot_plugin_natural_chat.main import PLUGIN_ID

        meta = Path(__file__).resolve().parents[1] / "metadata.yaml"
        for line in meta.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text.startswith("name:"):
                self.assertEqual(PLUGIN_ID, text.split(":", 1)[1].strip().strip("\"'"))
                return
        self.fail("metadata.yaml 缺少 name 字段")


class TestTextPartFactory(unittest.TestCase):
    def test_plugin_hands_core_a_usable_factory(self):
        """回归：4.0.0 把零参数探测函数当工厂传进 core，每轮注入抛 TypeError 被吞，提醒静默失效。

        必须从插件构造处断言：直接测 helper 抓不到接线错误。
        """
        import tempfile
        from pathlib import Path

        from astrbot_plugin_natural_chat.rules import text_part_ready

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            main_module.StarTools, "get_data_dir", return_value=Path(tmp)
        ):
            plugin = main_module.NaturalChatPlugin(object(), {})
        self.assertTrue(text_part_ready(plugin.core.text_part_factory), "工厂必须能产出 part")

    def test_factory_returns_text_part_with_text(self):
        from astrbot_plugin_natural_chat.rules import make_text_part

        probed = main_module._probe_text_part_cls()
        self.assertIsNotNone(probed, "宿主环境应能探测到 TextPart，探测失败说明宿主契约变了")
        part = make_text_part("提示", main_module._make_text_part)
        self.assertEqual(part.text, "提示")

    def test_factory_is_keyword_only(self):
        """core 按 factory(text=...) 调用，签名不匹配就是线上那条 TypeError。"""
        with self.assertRaises(TypeError):
            main_module._make_text_part("提示")


class TestHostRegistration(unittest.TestCase):
    def test_plugin_is_star_subclass_and_handlers_registered(self):
        self.assertTrue(issubclass(NaturalChatPlugin, Star))
        handlers = star_handlers_registry.get_handlers_by_module_name(NaturalChatPlugin.__module__)
        names = {handler.handler_name for handler in handlers}
        self.assertTrue(
            {
                "on_llm_request",
                "on_llm_response",
                "natural_status",
                "natural_on",
                "natural_off",
                "natural_reset",
                "natural_rules",
                "natural_stats",
            }
            <= names
        )
        event_types = {handler.handler_name: handler.event_type for handler in handlers}
        self.assertEqual(event_types["on_llm_request"], EventType.OnLLMRequestEvent)
        self.assertEqual(event_types["on_llm_response"], EventType.OnLLMResponseEvent)
        request_handler = next(handler for handler in handlers if handler.handler_name == "on_llm_request")
        self.assertEqual(request_handler.extras_configs["priority"], 100)

    def test_commands_keep_admin_permission_filter(self):
        handlers = star_handlers_registry.get_handlers_by_module_name(NaturalChatPlugin.__module__)
        commands = {"natural_status", "natural_on", "natural_off", "natural_reset", "natural_rules", "natural_stats"}
        for handler in handlers:
            if handler.handler_name not in commands:
                continue
            permissions = [item for item in handler.event_filters if isinstance(item, PermissionTypeFilter)]
            self.assertEqual(len(permissions), 1, handler.handler_name)
            self.assertEqual(permissions[0].permission_type, PermissionType.ADMIN)

    def test_readme_lists_all_natural_commands(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
        for command in ("status", "on", "off", "reset", "rules", "stats"):
            with self.subTest(command=command):
                self.assertIn(f"| `/natural {command}` |", readme)

    def test_plugin_adapter_swallows_core_errors(self):
        async def fail(event, req):
            raise RuntimeError("boom")

        stub = type("StubCore", (), {"on_llm_request": staticmethod(fail)})()
        plugin = type("StubPlugin", (), {"core": stub})()
        with mock.patch.object(main_module, "logger") as fake_logger:
            asyncio.run(NaturalChatPlugin.on_llm_request(plugin, object(), object()))
            self.assertEqual(fake_logger.error.call_count, 1)

        stub = type("StubCore", (), {"on_llm_response": staticmethod(fail)})()
        plugin = type("StubPlugin", (), {"core": stub})()
        with mock.patch.object(main_module, "logger") as fake_logger:
            asyncio.run(NaturalChatPlugin.on_llm_response(plugin, object(), object()))
            self.assertEqual(fake_logger.error.call_count, 1)

    def test_state_commands_report_pending_persistence(self):
        class Event:
            unified_msg_origin = "session"

            @staticmethod
            def plain_result(text):
                return text

        cases = (
            ("natural_on", "set_session_enabled", "启用"),
            ("natural_off", "set_session_enabled", "关闭"),
            ("natural_reset", "reset_session", "清空"),
        )
        for method_name, core_method, action in cases:
            with self.subTest(command=method_name):
                core = mock.Mock()
                setattr(core, core_method, mock.AsyncMock(return_value=False))
                plugin = type("StubPlugin", (), {"core": core})()

                async def collect():
                    method = getattr(NaturalChatPlugin, method_name)
                    return [item async for item in method(plugin, Event())]

                result = asyncio.run(collect())
                self.assertEqual(len(result), 1)
                self.assertIn(action, result[0])
                self.assertIn("写入失败", result[0])
                self.assertIn("待重试", result[0])

    def test_state_commands_keep_success_messages(self):
        class Event:
            unified_msg_origin = "session"

            @staticmethod
            def plain_result(text):
                return text

        cases = (
            ("natural_on", "set_session_enabled", "已启用"),
            ("natural_off", "set_session_enabled", "已关闭"),
            ("natural_reset", "reset_session", "已清空"),
        )
        for method_name, core_method, expected in cases:
            with self.subTest(command=method_name):
                core = mock.Mock()
                setattr(core, core_method, mock.AsyncMock(return_value=True))
                plugin = type("StubPlugin", (), {"core": core})()

                async def collect():
                    method = getattr(NaturalChatPlugin, method_name)
                    return [item async for item in method(plugin, Event())]

                self.assertIn(expected, asyncio.run(collect())[0])

    def test_stats_command_renders_key_rows(self):
        from astrbot_plugin_natural_chat.core import QualityStats

        class Event:
            @staticmethod
            def plain_result(text):
                return text

        stats = QualityStats()
        stats.record_request(True, True)
        stats.signal_hits["好的"] = 1
        plugin = type("StubPlugin", (), {"core": type("Core", (), {"stats": stats})()})()

        async def collect():
            return [item async for item in NaturalChatPlugin.natural_stats(plugin, Event())]

        text = asyncio.run(collect())[0]
        for row in ("累计注入：1 次", "固定规则：1 次", "动态提醒：1 次", "好的"):
            self.assertIn(row, text)

    def test_terminate_flushes_pending_state(self):
        store = mock.Mock()
        store.terminate = mock.AsyncMock(return_value=True)
        core = type("Core", (), {"stats": type("Stats", (), {"total_injections": 0})()})()
        plugin = type("StubPlugin", (), {"store": store, "core": core})()
        asyncio.run(NaturalChatPlugin.terminate(plugin))
        store.terminate.assert_awaited_once()

    def test_extra_parts_must_support_model_dump(self):
        """extra part 走对象路径：宿主装配时直接调用 part.model_dump()。

        4.23.x 对裸 dict 会抛 AttributeError（4.28 起宿主自行兼容），故插件只注入
        ContentPart 对象，两版宿主都能装配。
        """
        from astrbot.core.agent.message import TextPart
        from astrbot.core.provider.entities import ProviderRequest

        part = TextPart(text="owned")
        dumped = part.model_dump()
        self.assertEqual(dumped["type"], "text")
        self.assertEqual(dumped["text"], "owned")

        req = ProviderRequest(prompt="hi")
        req.extra_user_content_parts = [part]
        assembled = asyncio.run(req.assemble_context())
        self.assertTrue(any(block.get("text") == "owned" for block in assembled["content"]))

    def test_stale_extra_text_part_is_rewritten_then_assembled(self):
        from astrbot.core.agent.message import TextPart
        from astrbot.core.provider.entities import ProviderRequest
        from astrbot_plugin_natural_chat.rules import (
            RUNTIME_HINT_MARKER,
            append_temp_text_part,
            render_runtime_hint,
            rewrite_context_injections,
        )

        old = render_runtime_hint(["旧开头"])
        new = render_runtime_hint(["新开头"])
        req = ProviderRequest(prompt="hi")
        req.extra_user_content_parts = [TextPart(text=old)]

        result = rewrite_context_injections(req, new)
        self.assertEqual(req.extra_user_content_parts, [])
        self.assertEqual(result.runtime_removed, 1)

        self.assertTrue(append_temp_text_part(req, new, TextPart, marker=RUNTIME_HINT_MARKER))
        self.assertTrue(all(hasattr(part, "model_dump") for part in req.extra_user_content_parts))
        self.assertFalse(any(isinstance(part, dict) for part in req.extra_user_content_parts))

        assembled = asyncio.run(req.assemble_context())
        texts = [block.get("text") for block in assembled["content"] if isinstance(block, dict)]
        self.assertIn("hi", texts)
        self.assertIn(new, texts)
        self.assertNotIn(old, texts)

    def test_terminate_logs_failed_final_flush(self):
        store = mock.Mock()
        store.terminate = mock.AsyncMock(return_value=False)
        core = type("Core", (), {"stats": type("Stats", (), {"total_injections": 0})()})()
        plugin = type("StubPlugin", (), {"store": store, "core": core})()
        with mock.patch("astrbot_plugin_natural_chat.main.logger.warning") as warning:
            asyncio.run(NaturalChatPlugin.terminate(plugin))
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
