from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover
    logger = None  # type: ignore

from .constants import MAX_AVOID_ITEM_LEN, MAX_AVOID_ITEMS
from .protocols import ProviderRequestProtocol, TextPartFactoryProtocol

# 所有注入 marker 的公共前缀
INJECTED_MARKER_PREFIX = "[Natural Chat"
# 规则版本：规则正文变动时 +1，旧块在新版本下会被当作历史块清理
RULES_VERSION = 24
STABLE_RULE_MARKER = f"{INJECTED_MARKER_PREFIX} Rules v{RULES_VERSION}]"
RUNTIME_HINT_MARKER = f"{INJECTED_MARKER_PREFIX} Runtime]"
_RUNTIME_INSTRUCTION = "本轮避开这些重复项，换种自然说法，别提本提示："
_RUNTIME_PREFIX = f"{RUNTIME_HINT_MARKER}\n{_RUNTIME_INSTRUCTION}\n"
_RUNTIME_ITEM_SEPARATOR = "、"
_LITE_CORE = """natural-talk 轻量注入版

定位：只管日常对话与日常工作的回复质量；成文类任务（作文章、写小说、改稿）不适用本规则。

原则：不知即说，不编造，不编数字来源；评价只针对内容客体不针对人；被问身份如实简答；
代码块、行内代码、命令、路径、配置项、日志原文、版本号、错误码、URL 与表格逐字保留，不做同义替换；
用户明确要求的格式与结构照做，与下列规则冲突时以用户指令为准（事实与安全边界除外）。

对话姿态：
- 首句直接给结论或第一个事实，末句停在事实、明确建议或排查边界上；不写「好问题」「希望能帮到你」「随时联系我」这类服务腔，也不复述用户的问题当开场。
- 一句话的问题给短回答，不扩写成结构化长文；多轮里不重复已给过的结论（追问才展开），用户换话题就跟着走，跨轮指代用对方的说法（「你说的第二条」），不自造简称。
- 评价只针对内容：检验法是主语换成「这个方案/这段代码」仍成立，换不了的即越界；倾诉场景坦陈真实感受，不做居高临下的心理按摩。
- 问什么答什么并给出倾向；确属两难就写明判断条件与权衡基准，不用「关键在于平衡」「因人而异」「视情况而定」消解信息。
- 没把握时先给判断和依据，再给一步能立刻验证的路径；不写「作为AI」「根据我的训练数据」「截至我的知识更新」这类免责垫话。
- 用户给出数量时严格按原量词执行：给3步=恰好3步，至少3=不少于3，至多3=不超过3；凑不出恰好 N 条就说明只有 N 条，不凑数也不借机扩成教程。用户给了字数区间就照区间写。
- 不写无出处的「有研究表明」「业内普遍认为」「专家指出」「不少用户反馈」：有出处写出处，没出处直接陈述事实。

句式：
- 句首不用「说白了」「说穿了」「简单来说」「先说结论」「说到底」「归根结底」；「先说结论：」「直接给答案：」是空壳，结论本身开场。
- 句首不写空壳背景：「在当今」「随着……的发展」「众所周知」不带具体信息就删；带了人物、时间、数据、事件就删壳留信息。
- 没人主张过 A 就不写「不是A，而是B」（含「其实不是……只是」「与其……不如」「看似……实则」「不在于……而在于」「真正的问题是」）；用户前提里已经说过的、以及用户说法被推翻时的纠错句（「不是缓存的问题，是连接池打满了」）照写。
- 写完最后一件事即停，不在结尾写「这不仅是……更是」「更关乎」式拔高。
- 删「一句话总结：」「核心是：」「总结如下：」「建议如下：」这类空转提示语，内容自然承接。
- 破折号用来制造揭晓或后置补充（「核心只有一个——缓存」）的改成常规标点；话被打断、没说完的停顿保留。
- 同句不堆多个不确定词（可能…也许…大概）；单个「可能」是诚实，保留。
- 一句话的问题给短回答，不为了显得完整而拆成一串列表或加粗标题；用户要清单或步骤时照做。

禁止当作问题修改的内容（防误杀）：
- 操作指导里的顺序步骤词（首先……其次……最后）、设问与反问、讲原理的机制比喻、工程与业务术语（幂等、解耦这类）按原词写。
- 用户引用的原话、报错文本、他人台词写在引号里时逐字保留。

底线：懂行同行的平视交流质感；惯例降低频率而非归零。
"""

_PLUGIN_EXTRAS = (
    "插件附加（不改变上述原则）：\n"
    "- 保留事实、限制条件、安全提示和不确定性表述\n"
    "- 用户明确要求技术步骤、对比或指定格式时，以任务完成为先\n"
    "- 不要把这些约束写进回复"
)
_HISTORY_RULES_MARKER_RE = re.compile(r"^[ \t]*\[Natural Chat Rules v\d+\][ \t]*$")
_NEWLINE_RE = re.compile(r"\r\n|\r|\n")
_LEADING_SEPARATOR_RE = re.compile(r"^(?:(?:\r\n|\r|\n)){2}")
_TRAILING_SEPARATOR_RE = re.compile(r"(?:(?:\r\n|\r|\n)){2}$")


@dataclass(frozen=True)
class StableRewriteResult:
    text: str
    injected: bool
    removed: int
    ambiguous: bool


@dataclass(frozen=True)
class ContextRewriteResult:
    stable_removed: int = 0
    runtime_satisfied: bool = False
    runtime_removed: int = 0
    runtime_ambiguous: int = 0


def build_stable_rules() -> str:
    return f"{STABLE_RULE_MARKER}\n遵循 natural-talk 原则：\n\n{_LITE_CORE}\n\n{_PLUGIN_EXTRAS}"


def _normalize_newlines(text: str) -> str:
    return _NEWLINE_RE.sub("\n", text)


def _signature(text: str) -> tuple[int, str]:
    normalized = _normalize_newlines(text)
    return len(normalized.splitlines()), hashlib.sha256(normalized.encode()).hexdigest()


_STABLE_SIG_LINES, _STABLE_SIG_HASH = _signature(build_stable_rules())


def rewrite_stable_rules(system_prompt: str | None, *, enabled: bool) -> StableRewriteResult:
    """幂等写入固定规则块：开关关闭时剥离自家块，用户改过的块原样保留不再重复注入。"""
    text = system_prompt if isinstance(system_prompt, str) else ""
    matches, current_present, ambiguous = _find_stable_blocks(text)
    current_kept = False
    removals: list[tuple[int, int]] = []

    for start, end, _marker in matches:
        if enabled and not current_kept:
            current_kept = True
            continue
        removals.append(_expand_stable_removal(text, start, end))

    if removals:
        text = _remove_spans(text, removals)

    injected = False
    if enabled and not current_present:
        rules = build_stable_rules()
        if text:
            newline = _first_newline(text)
            if text.endswith(newline * 2):
                separator = ""
            elif text.endswith(newline):
                separator = newline
            else:
                separator = newline * 2
            text = f"{text}{separator}{rules}"
        else:
            text = rules
        injected = True

    return StableRewriteResult(text, injected, len(removals), ambiguous)


def rewrite_context_injections(req: ProviderRequestProtocol, runtime_text: str | None) -> ContextRewriteResult:
    """清理历史里的自家注入块，并在当前请求的附加内容里保留至多一个匹配提示。"""
    result = ContextRewriteResult()
    contexts = getattr(req, "contexts", None)
    if isinstance(contexts, list):
        for ctx in contexts:
            if not isinstance(ctx, dict) or ctx.get("role") != "user":
                continue
            content = ctx.get("content")
            if isinstance(content, str):
                rewritten, item_result = _rewrite_history_text(content)
                if rewritten != content:
                    ctx["content"] = rewritten
            elif isinstance(content, list):
                # 逐 part 就地删（子列表）或用等长判定（新构造列表），避免依赖 TextPart 的 __eq__
                rewritten, item_result, changed = _rewrite_history_parts(content)
                if changed:
                    # 整块被清空时只留一个空格占位：宿主不接受空 content 的 user 消息
                    ctx["content"] = rewritten if rewritten else " "
            else:
                continue
            result = _merge_context_results(result, item_result)

    parts = getattr(req, "extra_user_content_parts", None)
    if isinstance(parts, list):
        rewritten, item_result, changed = _rewrite_extra_parts(parts, runtime_text)
        if changed:
            req.extra_user_content_parts = rewritten
        result = _merge_context_results(result, item_result)
    # 「已满足」只在本轮真有提示并找到同款时成立：让位轮 hint 为空，历史里同款提示是被清掉的旧块，
    # 不能记成 satisfied，否则下一轮会误以为提示已在位而静默。
    if not runtime_text:
        result = ContextRewriteResult(
            stable_removed=result.stable_removed,
            runtime_satisfied=False,
            runtime_removed=result.runtime_removed,
            runtime_ambiguous=result.runtime_ambiguous,
        )
    return result


def _find_stable_blocks(text: str) -> tuple[list[tuple[int, int, str]], bool, bool]:
    """返回 (签名匹配的可剥离块, 当前版本块是否已存在, 是否含被改过的当前块)。

    当前版本 marker：无论签名是否匹配都算已注入（用户改过的块保留，不重复追加）。
    """
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    matches: list[tuple[int, int, str]] = []
    current_present = False
    ambiguous = False
    for index, line in enumerate(lines):
        if line.rstrip("\r\n") != STABLE_RULE_MARKER:
            continue
        current_present = True
        last = index + _STABLE_SIG_LINES - 1
        if last >= len(lines):
            ambiguous = True
            continue
        end = starts[last] + len(lines[last].rstrip("\r\n"))
        candidate = _normalize_newlines(text[starts[index] : end])
        if hashlib.sha256(candidate.encode()).hexdigest() == _STABLE_SIG_HASH:
            matches.append((starts[index], end, STABLE_RULE_MARKER))
        else:
            ambiguous = True
    return matches, current_present, ambiguous


def _expand_stable_removal(text: str, start: int, end: int) -> tuple[int, int]:
    """把待删块边界吸收进相邻的分隔符，避免留下空行残渣。"""
    if start == 0:
        following = _LEADING_SEPARATOR_RE.match(text[end:])
        if following:
            return start, end + following.end()
        return start, end
    preceding = _TRAILING_SEPARATOR_RE.search(text[:start])
    if preceding:
        return preceding.start(), end
    following = _LEADING_SEPARATOR_RE.match(text[end:])
    if following:
        return start, end + following.end()
    return start, end


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]
    return text


def _first_newline(text: str) -> str:
    match = _NEWLINE_RE.search(text)
    return match.group(0) if match else "\n"


def _text_value(part: Any) -> str | None:
    value = getattr(part, "text", None)
    if value is None and isinstance(part, dict):
        value = part.get("text")
    return value if isinstance(value, str) else None


def _is_known_stable_text(text: str) -> bool:
    # 廉价前置判据：不含自家前缀的普通消息直接判非 owned，省掉每条历史的 SHA-256
    if INJECTED_MARKER_PREFIX not in text:
        return False
    normalized = _normalize_newlines(text)
    lines = normalized.splitlines()
    if not lines or lines[0] != STABLE_RULE_MARKER:
        return False
    return len(lines) == _STABLE_SIG_LINES and hashlib.sha256(normalized.encode()).hexdigest() == _STABLE_SIG_HASH


def _is_history_rules_block(text: str) -> bool:
    if INJECTED_MARKER_PREFIX not in text:
        return False
    first = _normalize_newlines(text).splitlines()[:1]
    return bool(first and _HISTORY_RULES_MARKER_RE.match(first[0]))


def _runtime_kind(text: str) -> str:
    if INJECTED_MARKER_PREFIX not in text:
        return "ordinary"
    normalized = _normalize_newlines(text)
    lines = normalized.splitlines()
    if not lines or lines[0] != RUNTIME_HINT_MARKER:
        return "ordinary"
    return "owned" if _is_complete_runtime(normalized) else "ambiguous"


def _is_complete_runtime(text: str) -> bool:
    if not text.startswith(_RUNTIME_PREFIX):
        return False
    items = text[len(_RUNTIME_PREFIX) :].split(_RUNTIME_ITEM_SEPARATOR)
    return 1 <= len(items) <= MAX_AVOID_ITEMS and all(
        0 < len(item) <= _RUNTIME_ITEM_MAX_LEN and "\n" not in item for item in items
    )


def _rewrite_history_text(text: str) -> tuple[str, ContextRewriteResult]:
    if _is_known_stable_text(text) or _is_history_rules_block(text):
        return "", ContextRewriteResult(stable_removed=1)
    kind = _runtime_kind(text)
    if kind == "ordinary":
        return text, ContextRewriteResult()
    if kind == "ambiguous":
        return text, ContextRewriteResult(runtime_ambiguous=1)
    return "", ContextRewriteResult(runtime_removed=1)


def _rewrite_history_parts(parts: list[Any]) -> tuple[list[Any], ContextRewriteResult, bool]:
    """就地删除 owned part，返回 (同一列表, 统计, 是否发生删除)。

    就地改而不是重建列表：TextPart 未必实现 __eq__，重建后靠 == 判定变更多少不可靠。
    """
    result = ContextRewriteResult()
    kept: list[Any] = []
    for part in parts:
        text = _text_value(part)
        if text is None:
            kept.append(part)
            continue
        replacement, flags = _rewrite_history_text(text)
        if replacement:
            kept.append(part)
        result = _merge_context_results(result, flags)
    changed = len(kept) != len(parts)
    if changed:
        parts[:] = kept
    return parts, result, changed


def _rewrite_extra_parts(parts: list[Any], runtime_text: str | None) -> tuple[list[Any], ContextRewriteResult, bool]:
    """就地清理附加内容里的 owned part，返回 (同一列表, 统计, 是否发生删除)。"""
    result = ContextRewriteResult()
    kept: list[Any] = []
    for part in parts:
        text = _text_value(part)
        if text is None:
            kept.append(part)
            continue
        if _is_known_stable_text(text) or _is_history_rules_block(text):
            result = _merge_context_results(result, ContextRewriteResult(stable_removed=1))
            continue
        kind = _runtime_kind(text)
        if kind == "ordinary":
            kept.append(part)
            continue
        if kind == "ambiguous":
            kept.append(part)
            result = _merge_context_results(result, ContextRewriteResult(runtime_ambiguous=1))
            continue
        if (
            runtime_text
            and not result.runtime_satisfied
            and _normalize_newlines(text) == _normalize_newlines(runtime_text)
        ):
            kept.append(part)
            result = _merge_context_results(result, ContextRewriteResult(runtime_satisfied=True))
            continue
        result = _merge_context_results(result, ContextRewriteResult(runtime_removed=1))
    changed = len(kept) != len(parts)
    if changed:
        parts[:] = kept
    return parts, result, changed


def _merge_context_results(left: ContextRewriteResult, right: ContextRewriteResult) -> ContextRewriteResult:
    return ContextRewriteResult(
        stable_removed=left.stable_removed + right.stable_removed,
        runtime_satisfied=left.runtime_satisfied or right.runtime_satisfied,
        runtime_removed=left.runtime_removed + right.runtime_removed,
        runtime_ambiguous=left.runtime_ambiguous + right.runtime_ambiguous,
    )


# 信号名 → 模型端可执行的替换指令
_SIGNAL_HINT_MAP: dict[str, str] = {
    "翻案腔": "别用“不是A而是B”式对比，直接说肯定面",
    "结尾拔高": "别在结尾升大命题，写完最后一件事实就停",
    "空转提示语": "删掉“核心是：”这类空转提示语，内容自然承接",
    "揭示式破折号": "别用破折号制造揭晓，改成完整句",
    "模糊叠加": "同句别堆多个不确定词",
    "然而连发": "连用然而",
    "路标词堆砌": "堆砌路标词",
    "感叹号": "别堆感叹号",
}

# 块所有权校验的单项上限：按渲染后的实际产出推导。渲染会把信号名替换成 _SIGNAL_HINT_MAP 里的
# 模型端指令（比入库口径更长），沿用 MAX_AVOID_ITEM_LEN 时自己产出的块过不了自己的校验，
# 被判 ambiguous（历史里永不清理的孤儿块）。
_RUNTIME_ITEM_MAX_LEN = max(MAX_AVOID_ITEM_LEN, *(len(value) for value in _SIGNAL_HINT_MAP.values()))

# 渲染后不允许出现在单项内的字符：分隔符会让回转校验按错误边界切分；\r\n 会被换行归一化触发拒判
_RUNTIME_ITEM_FORBIDDEN = (_RUNTIME_ITEM_SEPARATOR, "\r", "\n")


def select_runtime_hint_names(openers: Sequence[str], max_chars: int) -> list[str]:
    """按字符预算挑选可整条装入的避用项（先过滤后截断，避免前 N 位非法项白吃名额）。"""
    openers = [
        item
        for item in openers
        if item and len(item) <= MAX_AVOID_ITEM_LEN and not any(char in item for char in _RUNTIME_ITEM_FORBIDDEN)
    ][:MAX_AVOID_ITEMS]
    if not openers:
        return []
    prefix_len = len(_RUNTIME_PREFIX)
    sep_len = len(_RUNTIME_ITEM_SEPARATOR)
    selected: list[str] = []
    current_len = prefix_len
    for raw_item in openers:
        increment = (sep_len if selected else 0) + len(_SIGNAL_HINT_MAP.get(raw_item, raw_item))
        if current_len + increment > max_chars:
            break
        selected.append(raw_item)
        current_len += increment
    return selected


def render_runtime_hint(names: Sequence[str]) -> str:
    if not names:
        return ""
    return _RUNTIME_PREFIX + _RUNTIME_ITEM_SEPARATOR.join(_SIGNAL_HINT_MAP.get(item, item) for item in names)


def make_text_part(text: str, factory: TextPartFactoryProtocol | None = None) -> Any | None:
    """构造临时文本 part；factory 为 None 时返回 None（调用方自行降级）。

    provider 对未知 part 类型会直接抛错，故失败时不得产出伪 part。
    """
    if factory is None:
        return None
    try:
        return factory(text=text)
    except Exception as e:
        if logger is not None:
            logger.error(f"[NaturalChat] make_text_part failed: {e}")
        return None


def append_temp_text_part(
    req: ProviderRequestProtocol,
    text: str,
    factory: TextPartFactoryProtocol | None = None,
    *,
    marker: str | None = None,
) -> bool:
    """构造并追加单轮提示。契约：注入文本必须以 marker 开头（幂等的前提），否则拒绝注入。"""
    if not text.strip():
        return False
    if marker and not text.lstrip().startswith(marker):
        if logger is not None:
            logger.warning(f"[NaturalChat] injected text missing marker prefix: {marker!r}")
        return False
    part = make_text_part(text, factory)
    if part is None:
        return False
    try:
        if not hasattr(req, "extra_user_content_parts") or req.extra_user_content_parts is None:
            req.extra_user_content_parts = []
        parts = req.extra_user_content_parts
        if not isinstance(parts, list):
            return False
        parts.append(part)
        return True
    except Exception as e:
        if logger is not None:
            logger.error(f"[NaturalChat] append temp text part failed: {e}")
        return False
