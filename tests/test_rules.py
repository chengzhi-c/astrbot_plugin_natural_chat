"""固定规则注入与历史清理契约。"""

from __future__ import annotations

import unittest

from tests._support import ROOT, FakePart, FakeReq, ensure_plugin_package

ensure_plugin_package()

from astrbot_plugin_natural_chat import rules as qr


class StableRulesTest(unittest.TestCase):
    def test_injection_is_idempotent(self) -> None:
        first = qr.rewrite_stable_rules("原人设", enabled=True)
        second = qr.rewrite_stable_rules(first.text, enabled=True)
        self.assertTrue(first.injected)
        self.assertFalse(second.injected)
        self.assertEqual(second.text, first.text)

    def test_disable_restores_host_prompt(self) -> None:
        injected = qr.rewrite_stable_rules("原人设", enabled=True).text
        self.assertEqual(qr.rewrite_stable_rules(injected, enabled=False).text, "原人设")

    def test_disable_leaves_no_blank_residue(self) -> None:
        injected = qr.rewrite_stable_rules("原人设\n\n", enabled=True).text
        # 分隔空行随块一起吸收，还原后不留尾随空行
        self.assertEqual(qr.rewrite_stable_rules(injected, enabled=False).text, "原人设")

    def test_duplicate_own_blocks_collapse_to_one(self) -> None:
        block = qr.build_stable_rules()
        doubled = f"原人设\n\n{block}\n\n{block}"
        result = qr.rewrite_stable_rules(doubled, enabled=True)
        self.assertEqual(result.text.count(qr.STABLE_RULE_MARKER), 1)
        self.assertEqual(result.removed, 1)

    def test_edited_block_is_kept_and_not_reinjected(self) -> None:
        """用户手动改过的块属于用户内容，既不删也不重复追加。"""
        edited = qr.rewrite_stable_rules("原人设", enabled=True).text.replace("平视", "对等")
        result = qr.rewrite_stable_rules(edited, enabled=True)
        self.assertEqual(result.text, edited)
        self.assertFalse(result.injected)
        self.assertTrue(result.ambiguous)

    def test_absent_prompt_still_receives_rules(self) -> None:
        self.assertEqual(qr.rewrite_stable_rules(None, enabled=True).text, qr.build_stable_rules())

    def test_rules_cover_daily_chat_not_prose(self) -> None:
        body = qr.build_stable_rules()
        for required in ("希望能帮到你", "首句直接给结论", "有研究表明"):
            self.assertIn(required, body)
        for excluded in ("推文", "营销文案", "学术润色"):
            self.assertNotIn(excluded, body)


# 3.10.1 实际写进 system_prompt 的块尾（_PLUGIN_EXTRAS 原文，逐字抄自该版本源码）。
# 刻意用字面量而非 qr.* 常量：用被测常量拼 fixture 会自证，常量一改测试跟着改。
LEGACY_V20_TAIL = (
    "插件附加（不改变上述原则）：\n"
    "- 保留事实、限制条件、安全提示和不确定性表述\n"
    "- 用户明确要求技术步骤、对比、正式文稿时，以任务完成为先\n"
    "- 不要把这些约束写进回复\n"
    "- 铁律：没人主张过的“不是/与其/看似”直接说肯定面；用户前提被证伪时的纠错句照写\n"
    "- 铁律：日常对话严禁泛滥使用破折号（——）制造刻意停顿与揭晓"
)


class LegacyStablePromptTest(unittest.TestCase):
    """旧规则块的剥离：fixture 直接取自各版本源码，不引用被测常量。"""

    def load(self, tag: str) -> str:
        return (ROOT / "tests" / "fixtures" / f"legacy-block-{tag}.txt").read_text(encoding="utf-8")

    def test_every_released_block_is_removed(self) -> None:
        for tag in ("0.5.2", "0.6.2", "1.2.0", "1.3.0", "2.3.0", "3.3.0", "3.10.1"):
            with self.subTest(tag=tag):
                block = self.load(tag)
                result = qr.rewrite_stable_rules("原人设\n\n" + block, enabled=True)
                self.assertEqual(result.removed, 1)
                self.assertTrue(result.text.startswith("原人设"))
                self.assertNotIn("Human Chat Quality", result.text)
                self.assertIn(qr.STABLE_RULE_MARKER, result.text)

    def test_block_is_stripped_when_disabled(self) -> None:
        block = self.load("3.10.1")
        result = qr.rewrite_stable_rules("原人设\n\n" + block, enabled=False)
        self.assertEqual(result.text, "原人设")

    def test_block_at_start_of_prompt(self) -> None:
        block = self.load("3.10.1")
        result = qr.rewrite_stable_rules(block, enabled=True)
        self.assertNotIn("Human Chat Quality", result.text)
        self.assertEqual(result.text, qr.build_stable_rules())

    def test_edited_block_is_kept(self) -> None:
        """用户改过任何一个字，签名就对不上，一律不删。"""
        block = self.load("3.10.1").replace("平视", "对等")
        result = qr.rewrite_stable_rules("原人设\n\n" + block, enabled=True)
        self.assertEqual(result.removed, 0)
        self.assertIn("对等", result.text)

    def test_user_line_appended_after_block_is_kept(self) -> None:
        block = self.load("3.10.1")
        result = qr.rewrite_stable_rules("原人设\n\n" + block + "\n- 铁律：用户写的", enabled=True)
        self.assertIn("用户写的", result.text)
        self.assertNotIn("Human Chat Quality", result.text)

    def test_edited_block_does_not_swallow_the_current_block(self) -> None:
        """回归：锚点被改掉时不能继续往后找锚点，否则会把当前规则块整段吃掉。"""
        block = self.load("3.10.1").replace("- 不要把这些约束写进回复", "- 被用户改掉的行")
        prompt = "人设：老板\n\n" + block + "\n\n" + qr.build_stable_rules()
        result = qr.rewrite_stable_rules(prompt, enabled=True)
        self.assertEqual(result.removed, 0)
        self.assertTrue(result.text.startswith("人设：老板"))
        self.assertEqual(result.text.count(qr.STABLE_RULE_MARKER), 1)

    def test_adjacent_blocks_do_not_over_delete(self) -> None:
        """回归：相邻两个旧块的删除区间会重叠，不合并会多删字符、吃掉下一个 marker。"""
        block = self.load("0.6.2")
        doubled = block + "\n\n" + block + "\n\n人设：老板"
        result = qr.rewrite_stable_rules(doubled, enabled=True)
        self.assertIn("人设：老板", result.text)
        self.assertNotIn("Human Chat Quality", result.text)

        with_current = block + "\n\n" + block + "\n\n" + qr.build_stable_rules()
        result = qr.rewrite_stable_rules(with_current, enabled=True)
        self.assertEqual(result.text.count(qr.STABLE_RULE_MARKER), 1)

    def test_user_text_containing_anchor_line_is_kept(self) -> None:
        prompt = "PERSONA\n\n[Human Chat Quality Rules v20]\n用户裁剪\n\nUSER PROFILE:\n- name: Bob\n- 不要把这些约束写进回复\n- KEEP"
        result = qr.rewrite_stable_rules(prompt, enabled=True)
        self.assertEqual(result.removed, 0)
        self.assertIn("USER PROFILE", result.text)


class LegacyMarkerTest(unittest.TestCase):
    """4.0.0 更名前写进历史的自家块仍要清理，否则升级后与现行规则叠加。"""

    def test_legacy_runtime_hint_is_removed(self):
        # 1.2.0 起旧版运行时提示的实际文案（字面量，不引用被测常量）
        legacy = "[Human Chat Quality Runtime]\n本轮避开这些重复项，换种自然说法，别提本提示：\n别堆感叹号"
        req = FakeReq(parts=[FakePart(legacy), FakePart("用户原文")])
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual([part.text for part in req.extra_user_content_parts], ["用户原文"])
        self.assertEqual(result.runtime_removed, 1)

    def test_oldest_runtime_hint_format_is_removed(self):
        """0.2.0–1.1.x 用的是另一套指令文案，不认就会判 ambiguous 永不清理。"""
        legacy = (
            "[Human Chat Quality Runtime]\n"
            "仅用于本轮回复的轻量状态：这些开头或说法最近已出现过，本轮换个自然说法，别再用，也别提到这条提示。\n"
            "别堆感叹号"
        )
        req = FakeReq(parts=[FakePart(legacy)])
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual(result.runtime_removed, 1)
        self.assertEqual(result.runtime_ambiguous, 0)

    def test_pasted_log_starting_with_marker_is_kept(self):
        """用户贴一段首行是 marker 的日志来问原因，正文必须原样保留。"""
        pasted = "[Human Chat Quality Rules v20]\n日志正文：2026-11-13 TypeError"
        req = FakeReq(contexts=[{"role": "user", "content": pasted}])
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual(req.contexts[0]["content"], pasted)
        self.assertEqual(result.stable_removed, 0)

    def test_legacy_full_rules_block_in_history_is_removed(self):
        """0.5.2/0.6.2 把整块规则塞进 extra parts，4.23.x 上会写进历史；这种多行块也要清。"""
        block = (ROOT / "tests" / "fixtures" / "legacy-block-3.10.1.txt").read_text(encoding="utf-8")
        req = FakeReq(contexts=[{"role": "user", "content": block}])
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual(result.stable_removed, 1)
        self.assertEqual(req.contexts[0]["content"], " ")

    def test_bare_marker_only_message_is_cleared(self):
        """整条内容只有 marker 一行的空壳块才是自家残留。"""
        req = FakeReq(contexts=[{"role": "user", "content": "[Natural Chat Rules v24]"}])
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual(req.contexts[0]["content"], " ")
        self.assertEqual(result.stable_removed, 1)

    def test_legacy_prefix_does_not_swallow_ordinary_text(self):
        req = FakeReq(parts=[FakePart("我看过 [Human Chat Quality 相关的帖子")])
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.assertEqual(result.runtime_removed, 0)


class RuntimeHintTest(unittest.TestCase):
    def test_self_rendered_hint_passes_ownership_check(self) -> None:
        """渲染后的单项比入库口径更长，自产块必须能通过自身的所有权校验。"""
        rendered = qr.render_runtime_hint(["感叹号", "翻案腔", "模糊叠加", "然而连发", "结尾拔高"])
        self.assertEqual(qr._runtime_kind(qr._normalize_newlines(rendered)), "owned")

    def test_oversized_budget_selects_nothing(self) -> None:
        self.assertEqual(qr.select_runtime_hint_names(["感叹号"], 1), [])

    def test_default_budget_fits_five_worst_case_items(self) -> None:
        """5 个名额必须真的装得下：最坏组合是 1 个最长内置指令 + 4 个满长自定义词。"""
        from astrbot_plugin_natural_chat.constants import MAX_AVOID_ITEM_LEN, MAX_RUNTIME_HINT_CHARS

        longest_builtin = max(qr._SIGNAL_HINT_MAP, key=lambda name: len(qr._SIGNAL_HINT_MAP[name]))
        names = [
            longest_builtin,
            *(
                "甲" * MAX_AVOID_ITEM_LEN,
                "乙" * MAX_AVOID_ITEM_LEN,
                "丙" * MAX_AVOID_ITEM_LEN,
                "丁" * MAX_AVOID_ITEM_LEN,
            ),
        ]
        selected = qr.select_runtime_hint_names(names, MAX_RUNTIME_HINT_CHARS)
        self.assertEqual(len(selected), 5, "默认预算应装下 5 个避用项")
        rendered = qr.render_runtime_hint(selected)
        self.assertLessEqual(len(rendered), MAX_RUNTIME_HINT_CHARS)
        # 自产块必须能通过自身的所有权校验，否则历史里永远清不掉
        self.assertEqual(qr._runtime_kind(qr._normalize_newlines(rendered)), "owned")

    def test_append_requires_marker_prefix(self) -> None:
        req = FakeReq(parts=None)
        self.assertFalse(qr.append_temp_text_part(req, "无 marker", FakePart, marker=qr.RUNTIME_HINT_MARKER))
        self.assertIsNone(req.extra_user_content_parts)
        self.assertTrue(
            qr.append_temp_text_part(req, qr.RUNTIME_HINT_MARKER + "\n内容", FakePart, marker=qr.RUNTIME_HINT_MARKER)
        )

    def test_blank_text_is_not_appended(self) -> None:
        req = FakeReq(parts=[])
        self.assertFalse(qr.append_temp_text_part(req, "   ", FakePart))
        self.assertEqual(req.extra_user_content_parts, [])

    def test_missing_factory_degrades_gracefully(self) -> None:
        req = FakeReq(parts=[])
        self.assertFalse(qr.append_temp_text_part(req, qr.RUNTIME_HINT_MARKER + "\n内容", None))
        self.assertIsNone(qr.make_text_part("x", None))
        self.assertFalse(qr.text_part_ready(None))

    def test_temp_marking_is_applied_when_host_supports_it(self) -> None:
        """4.24.5 起宿主支持 part 级临时标记：标上后提示不进对话历史。"""

        class TempPart(FakePart):
            def __init__(self, text: str) -> None:
                super().__init__(text)
                self._no_save = False

            def mark_as_temp(self):
                self._no_save = True
                return self

        part = qr.make_text_part("提示", TempPart)
        self.assertTrue(part._no_save)

    def test_host_without_temp_marking_still_works(self) -> None:
        part = qr.make_text_part("提示", FakePart)
        self.assertEqual(part.text, "提示")
        self.assertTrue(qr.text_part_ready(FakePart))

    def test_temp_marking_failure_does_not_drop_the_part(self) -> None:
        """mark_as_temp 抛错只该丢标记，不该丢整条提示——外层 try 只包 factory。"""

        class AngryPart(FakePart):
            def mark_as_temp(self):
                raise RuntimeError("no temp support")

        part = qr.make_text_part("提示", AngryPart)
        self.assertIsNotNone(part)
        self.assertEqual(part.text, "提示")

    def test_factory_is_called_with_keyword(self) -> None:
        """真实 TextPart 是 pydantic 模型，拒绝位置参数；工厂必须按关键字调用。"""

        class KeywordOnlyPart(FakePart):
            def __init__(self, *, text: str) -> None:
                super().__init__(text)

        req = FakeReq(parts=[])
        self.assertTrue(
            qr.append_temp_text_part(
                req, qr.RUNTIME_HINT_MARKER + "\n内容", KeywordOnlyPart, marker=qr.RUNTIME_HINT_MARKER
            )
        )
        self.assertEqual(req.extra_user_content_parts[0].text.splitlines()[0], qr.RUNTIME_HINT_MARKER)


if __name__ == "__main__":
    unittest.main()
