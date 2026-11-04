"""signal_detectors 与 scene_guard 的判定口径。"""

from __future__ import annotations

import unittest

from tests._support import ensure_plugin_package

ensure_plugin_package()

from astrbot_plugin_natural_chat.scene_guard import is_sticky_followup, is_tone_request
from astrbot_plugin_natural_chat.signal_detectors import detect_cliches


def hits(text: str, **kwargs: object) -> list[str]:
    return detect_cliches(text, **kwargs)  # type: ignore[arg-type]


class OpeningTest(unittest.TestCase):
    def test_service_openings_are_caught(self) -> None:
        self.assertIn("好问题", hits("好问题，附近那家面馆就行。"))
        self.assertIn("说白了", hits("说白了，缓存没命中。"))
        self.assertIn("简单来说", hits("简单来说，先回滚。"))

    def test_real_action_openings_are_not_service_talk(self) -> None:
        self.assertEqual(hits("让我们先来点具体的：把日志打开。"), [])
        self.assertEqual(hits("我看了下日志，是连接池打满了。"), [])


class EndingTest(unittest.TestCase):
    def test_service_ending_is_caught(self) -> None:
        self.assertIn("希望能帮到你", hits("重启一下就好。希望能帮到你。"))

    def test_hedged_ending_is_caught(self) -> None:
        self.assertIn("因人而异", hits("看情况吧，因人而异。"))

    def test_ending_only_matches_the_tail(self) -> None:
        self.assertEqual(hits("随时联系我这种话不该由客服说，改掉。"), [])


class SignalTest(unittest.TestCase):
    def test_identity_disclaimers_are_caught(self) -> None:
        self.assertIn("作为AI", hits("作为AI，我不能替你决定。"))

    def test_unsourced_attribution_is_caught(self) -> None:
        self.assertIn("有研究表明", hits("有研究表明这样更快。"))
        self.assertIn("业内认为", hits("业内认为这是最佳实践。"))

    def test_sycophancy_is_caught(self) -> None:
        self.assertIn("你说得太对了", hits("你说得太对了，这里确实要改。"))

    def test_custom_cliche_is_case_insensitive(self) -> None:
        self.assertIn("great question", hits("Great Question, let me check.", custom_cliches=("great question",)))

    def test_duplicate_names_are_reported_once(self) -> None:
        found = hits("作为AI，作为AI，我不知道。")
        self.assertEqual(found.count("作为AI"), 1)


class IronRuleTest(unittest.TestCase):
    def test_reversal_pattern_is_caught(self) -> None:
        self.assertIn("翻案腔", hits("这不是优化，而是重构。"))
        self.assertIn("结尾拔高", hits("这不仅是工具，更是工程文化。"))
        self.assertIn("空转提示语", hits("核心是：先保证可观测性。"))

    def test_limited_negation_is_information_not_reversal(self) -> None:
        self.assertEqual(hits("不是只看数据库，而是也要查网络超时。"), [])

    def test_complete_reversal_shells_are_caught_regardless_of_limiters(self) -> None:
        """「看似…实则」是完整翻案句式，不享受「不是」那条限量否定豁免。"""
        self.assertIn("翻案腔", hits("看似简单，实则复杂。"))
        self.assertIn("翻案腔", hits("看似只改了一行，实则动了架构。"))

    def test_user_stated_negation_allows_correction(self) -> None:
        self.assertNotIn("翻案腔", hits("这不是缓存的问题，而是连接池打满了。", user_text="不是缓存的问题吗"))

    def test_open_why_question_does_not_buy_an_exemption(self) -> None:
        self.assertIn("翻案腔", hits("这不是妥协，只是算账。", user_text="年轻人为啥不买大件了？"))

    def test_localized_diagnosis_allows_rule_out(self) -> None:
        self.assertNotIn("翻案腔", hits("报错的来源不是应用层，而是网关超时。", user_text="这个报错是哪里来的"))

    def test_quoted_speech_is_left_alone(self) -> None:
        self.assertEqual(hits("角色说：“这不是优化而是重构。”这属于台词。"), [])
        self.assertEqual(hits("代码示例 `不是优化而是重构` 只是字符串。"), [])

    def test_fenced_code_and_urls_are_masked(self) -> None:
        self.assertEqual(hits("```python\nprint('不是优化而是重构')\n```\n见 https://e.com/?q=不是"), [])


class PunctuationTest(unittest.TestCase):
    def test_dash_explanation_is_caught(self) -> None:
        self.assertIn("揭示式破折号", hits("这次降级——就是漏掉的开关没开，别的都不是；先补开关。"))

    def test_dash_reveal_is_caught(self) -> None:
        self.assertIn("揭示式破折号", hits("答案只有一个——那就是缓存。"))

    def test_dash_reveal_variants_are_caught(self) -> None:
        """全角双破折号与 ASCII 双减号是同一种揭晓形态，不能因为字符集漏掉。"""
        for text in ("答案只有一个——原来是这样。", "答案只有一个--原来是这样。", "答案只有一个—原来是这样。"):
            with self.subTest(text=text):
                self.assertIn("揭示式破折号", hits(text))
        self.assertEqual(hits("这个符号 a--b 出现在标识符里。"), [])

    def test_density_scales_with_length(self) -> None:
        short = "太好了！太棒了！真厉害！冲啊！"
        self.assertIn("感叹号", hits(short))
        self.assertNotIn("感叹号", hits(short + "。" * 2000))


class HedgeTest(unittest.TestCase):
    def test_stacked_hedges_are_caught(self) -> None:
        self.assertIn("模糊叠加", hits("可能需要重启，也许还要重装，大概才能好。"))

    def test_single_hedge_is_honest(self) -> None:
        self.assertEqual(hits("可能是网络问题，先看下 PING。"), [])

    def test_paired_alternatives_are_information(self) -> None:
        self.assertEqual(hits("可能是，也可能只是表达直接。"), [])

    def test_hedges_split_by_sentence_do_not_stack(self) -> None:
        self.assertEqual(hits("可能是网络问题。也许要等通知。"), [])


class SceneGuardTest(unittest.TestCase):
    def test_tone_rewrite_requests_yield(self) -> None:
        for text in ("把这段话润色一下", "这段文字太AI了，改改", "帮我改写这封邮件", "换个口吻重写这段"):
            with self.subTest(text=text):
                self.assertEqual(is_tone_request(text), "tone")

    def test_soft_tone_requests_yield(self) -> None:
        """不带「改成」的软性语气要求也算改语气；靠程度副词区分，陈述句不能误伤。"""
        for text in ("写得自然点", "口语化一点", "改得正式一些", "说自然一些", "改得简洁点"):
            with self.subTest(text=text):
                self.assertEqual(is_tone_request(text), "tone")

    def test_daily_work_does_not_yield(self) -> None:
        for text in (
            "帮我写一个 python 脚本",
            "写一份周报总结这周进度",
            "这段代码报错了怎么改",
            "解释下这个函数",
            "这个设计很自然",
            "自然语言处理怎么做",
            "改自然语言处理的代码",
            "口语识别怎么做",
        ):
            with self.subTest(text=text):
                self.assertIsNone(is_tone_request(text))

    def test_pasted_artifact_yields(self) -> None:
        body = "原文如下：" + "公司决定下周把服务迁移到新机房，迁移窗口定在周六凌晨，期间会有短暂不可用。" * 2
        self.assertEqual(is_tone_request(body), "copy")

    def test_paste_with_leading_prefix_still_yields(self) -> None:
        """转手文本前面常带「[图片]」「@某人」，不能因此漏判。"""
        body = "原文如下：" + "公司决定下周把服务迁移到新机房，迁移窗口定在周六凌晨，期间会有短暂不可用。" * 2
        self.assertEqual(is_tone_request(f"[图片] {body} 转给同事"), "copy")

    def test_short_paste_with_task_does_not_yield(self) -> None:
        self.assertIsNone(is_tone_request("原文如下：你好，帮我润色一下"))

    def test_sticky_followup_needs_a_continuation_phrase(self) -> None:
        self.assertTrue(is_sticky_followup("继续改"))
        self.assertTrue(is_sticky_followup("继续改第三段"))
        self.assertTrue(is_sticky_followup("再顺一下"))
        self.assertTrue(is_sticky_followup("同上"))
        self.assertFalse(is_sticky_followup("你好"))
        self.assertFalse(is_sticky_followup("继续"))
        self.assertFalse(is_sticky_followup("继续" + "很长的补充" * 10))


if __name__ == "__main__":
    unittest.main()
