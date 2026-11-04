"""会话场景守卫：判定这一轮是不是「改语气」或「转手成品文本」。

插件管的是自己怎么说话。用户让模型改一段已有文字时，语体由用户指定，不归本插件管；
贴来一大段要转发的成品文本时，也不该被改写。这两类让位。
日常工作里的写代码、写脚本、写通知、写周报都照常注入——那才是这个插件的主场。
"""

from __future__ import annotations

import re

from .constants import PASTE_YIELD_MIN_CHARS, STICKY_FOLLOWUP_MAX_LEN
from .protocols import MessageEventProtocol


def event_text(event: MessageEventProtocol | None) -> str:
    """宿主对象形状随版本变化，逐属性探测是有意的。"""
    if event is not None:
        for attr in ("get_message_str", "message_str", "message", "text"):
            try:
                value = getattr(event, attr, None)
                if callable(value):
                    value = value()
                if isinstance(value, str) and value.strip():
                    return value.strip()
            except Exception:
                continue
    return ""


_TONE_ACTION = re.compile(r"润色|改写|重写|改一下|改改|改得|顺一下|顺顺|理顺|换个说法|换种说法|换个口吻")
# 显式语气要求：既认「改成X语气」，也认「写得自然点」「口语化一点」这类不带「改成」的软性说法。
# 后者必须带程度副词或「点/一些」，避免误伤「这个设计很自然」这类陈述。
# 词尾不得紧跟汉字，否则「改自然语言处理的代码」会被当成改语气。
_TONE_MARKERS = re.compile(
    r"改成[^，。；！？\n]{0,6}(?:语气|口吻|腔调)"
    r"|太\s*AI\s*了?|AI\s*(?:味|腔)"
    r"|(?:写|说|表达|改)得?(?:更)?(?:自然|口语|随和|亲切|正式|委婉|简洁|生动)(?:一?点|一?些|些)?(?![\u4e00-\u9fff])"
    r"|(?:口语化|自然化|拟人化)(?:一?点|一?些|些)?(?![\u4e00-\u9fff])"
)
_DEMO_TARGET = re.compile(r"这段话|这段文字|这段|这篇|这份|这封|这条|这句|这稿|稿子|文字|文档")
_FORMAL_ARTIFACT = re.compile(r"文书|声明|公文|稿件|稿子|邮件|文案|申请|通知|纪要")
# 任意位置匹配：转手文本前面常带「[图片]」「@某人」这类前缀
_PASTE_BODY = re.compile(r"(?:^|[\s\]）)])(?:原文|内容|全文)如下")
_PASTE_TASK = re.compile(r"润色|改写|重写|翻译|总结|摘要|归纳|校对|精简|扩写|改一下|改改|顺一下")


def is_tone_request(text: str) -> str | None:
    """返回让位类别：tone（改语气）/ copy（转手成品文本）/ None（正常对话，照常注入）。"""
    stripped = text.strip()
    if not stripped:
        return None
    if _TONE_MARKERS.search(stripped) or (
        _TONE_ACTION.search(stripped) and (_DEMO_TARGET.search(stripped) or _FORMAL_ARTIFACT.search(stripped))
    ):
        return "tone"
    if len(stripped) >= PASTE_YIELD_MIN_CHARS and _PASTE_BODY.search(stripped) and not _PASTE_TASK.search(stripped):
        return "copy"
    return None


_CONTINUATION_RE = re.compile(
    r"^(?:继续|接着)(?:改|润色|写|顺|来)(?:一下|这|它|第[一二三四五六七八九十]+[段落章节])?[。.!！?？]*$"
    r"|^再(?:改|润色|写|顺)(?:一下)?[。.!！?？]*$"
    r"|^(?:同上|按这个改|继续这样|就这样改)[。.!！?？]*$"
)


def is_sticky_followup(text: str) -> bool:
    """让位窗口内的续写口令（继续／接着改／同上）。普通短消息不算，否则日常问答会被吞。"""
    stripped = text.strip()
    return bool(stripped) and len(stripped) <= STICKY_FOLLOWUP_MAX_LEN and bool(_CONTINUATION_RE.fullmatch(stripped))
