"""固定规则注入与历史清理契约。"""

from __future__ import annotations

import unittest

from tests._support import FakePart, FakeReq, ensure_plugin_package

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


class ContextRewriteTest(unittest.TestCase):
    def test_removes_own_blocks_from_history_parts(self) -> None:
        """parts 形态必须就地删除：TextPart 不保证 __eq__，重建列表无法判等。"""
        content = [
            FakePart(qr.render_runtime_hint(["感叹号"])),
            FakePart("真实用户内容"),
            FakePart(qr.build_stable_rules()),
        ]
        req = FakeReq(contexts=[{"role": "user", "content": content}])
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual([part.text for part in content], ["真实用户内容"])
        self.assertEqual(result.stable_removed, 1)
        self.assertEqual(result.runtime_removed, 1)

    def test_removes_own_blocks_from_history_strings(self) -> None:
        req = FakeReq(contexts=[{"role": "user", "content": qr.build_stable_rules()}])
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual(result.stable_removed, 1)
        self.assertEqual(req.contexts[0]["content"], "")

    def test_plain_history_is_untouched(self) -> None:
        contexts = [{"role": "user", "content": "帮我看下这个报错"}, {"role": "assistant", "content": "贴日志"}]
        req = FakeReq(contexts=contexts)
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual(req.contexts, contexts)
        self.assertEqual(result, qr.ContextRewriteResult())

    def test_matching_runtime_hint_is_kept(self) -> None:
        hint = qr.render_runtime_hint(["感叹号"])
        req = FakeReq(parts=[FakePart(hint)])
        result = qr.rewrite_context_injections(req, hint)
        self.assertTrue(result.runtime_satisfied)
        self.assertEqual(len(req.extra_user_content_parts), 1)

    def test_stale_runtime_hints_are_dropped(self) -> None:
        req = FakeReq(parts=[FakePart(qr.render_runtime_hint(["感叹号"])), FakePart("用户原文")])
        result = qr.rewrite_context_injections(req, None)
        self.assertEqual([part.text for part in req.extra_user_content_parts], ["用户原文"])
        self.assertEqual(result.runtime_removed, 1)

    def test_yield_round_does_not_report_itself_as_satisfied(self) -> None:
        """让位轮没有本轮提示，历史里的同款旧块被清掉；不能记成 satisfied，否则下一轮静默。"""
        hint = qr.render_runtime_hint(["感叹号"])
        req = FakeReq(parts=[FakePart("用户本轮正文")], contexts=[{"role": "user", "content": hint}])
        result = qr.rewrite_context_injections(req, None)
        self.assertFalse(result.runtime_satisfied)
        self.assertEqual(result.runtime_removed, 1)

    def test_malformed_owned_block_is_kept_and_flagged(self) -> None:
        broken = qr.RUNTIME_HINT_MARKER + "\n乱写的内容"
        req = FakeReq(parts=[FakePart(broken)])
        result = qr.rewrite_context_injections(req, "随便")
        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.assertEqual(result.runtime_ambiguous, 1)

    def test_absent_parts_container_is_tolerated(self) -> None:
        req = FakeReq(parts=None)
        self.assertEqual(qr.rewrite_context_injections(req, None), qr.ContextRewriteResult())
        self.assertIsNone(req.extra_user_content_parts)


class RuntimeHintTest(unittest.TestCase):
    def test_self_rendered_hint_passes_ownership_check(self) -> None:
        """渲染后的单项比入库口径更长，自产块必须能通过自身的所有权校验。"""
        rendered = qr.render_runtime_hint(["感叹号", "翻案腔", "模糊叠加", "然而连发", "结尾拔高"])
        self.assertEqual(qr._runtime_kind(qr._normalize_newlines(rendered)), "owned")

    def test_oversized_budget_selects_nothing(self) -> None:
        self.assertEqual(qr.select_runtime_hint_names(["感叹号"], 1), [])

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


if __name__ == "__main__":
    unittest.main()
