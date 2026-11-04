"""AI 腔信号检测：词表面 + 固定句式 + 篇幅折算的密度项。

检测口径对齐 natural-talk 生成模式：只收无需读语境即可判定的高置信形态。
需要读上下文的语义判断（如「评价是否越界」）不在这里机械匹配，交给固定规则约束。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from .constants import CONSECUTIVE_THRESHOLD, DENSITY_BASE, OPENER_DELIM

# D5 模型身份与训练数据垫话：任意位置命中即报
DEFAULT_AI_CLICHES: tuple[str, ...] = (
    "作为AI",
    "作为人工智能",
    "作为一个语言模型",
    "作为一个AI助手",
    "作为AI语言模型",
    "根据我的训练",
    "基于我的训练数据",
    "训练数据截至",
    "截至我的知识",
    "基于我所掌握的信息",
    "截至我的知识更新",
)

# D1 服务腔开场 / D3 元话语空预告 / 核心规则 1 的句首壳：仅回复首部（首个标点前）命中
OPENING_CLICHES: tuple[str, ...] = (
    "好问题",
    "问得好",
    "这是个好问题",
    "这真是个好问题",
    "让我来",
    "感谢你的提问",
    "Great question",
    "关于你提到的",
    "关于您提到的",
    "你问到了核心",
    "在当今快速发展的时代",
    "随着AI不断进步",
    "说白了",
    "说穿了",
    "简单来说",
    "简单说",
    "先说结论",
    "说到底",
    "归根结底",
    "直接给答案",
    "让我们先来",
    "下面我将",
    "接下来我将",
    "众所周知",
)

# D3 负例：这些首部开头是真实动作，不是空预告
_OPENING_NEGATIVES: tuple[str, ...] = (
    "让我们先来点",
    "让我们先来听",
    "让我们先来看",
    "让我们先来试",
)

# D1 谄媚越界：整句级高置信触发，任意位置命中
DEFAULT_SYMPATHY_CLICHES: tuple[str, ...] = (
    "我完全理解你的感受",
    "你说得太对了",
)

# D4 模糊假归因：无出处的背书，任意位置命中
DEFAULT_VAGUE_ATTRIBUTIONS: tuple[str, ...] = (
    "有研究表明",
    "业内普遍认为",
    "业内认为",
    "专家指出",
    "不少用户反馈",
)

# D1 客服收尾 + D2 万能收尾 + 空泛打气：仅结尾命中
DEFAULT_ENDINGS: tuple[str, ...] = (
    "希望能帮到你",
    "希望这能帮到你",
    "希望对你有帮助",
    "希望对您有帮助",
    "希望对你有所帮助",
    "如果还有问题",
    "如果还有其他问题",
    "如有疑问",
    "如有任何问题",
    "有任何问题随时",
    "随时联系我",
    "欢迎继续交流",
    "欢迎随时",
    "未来可期",
    "一起加油",
    "共同努力",
    "砥砺前行",
    "不忘初心",
    "关键在于找到平衡",
    "关键在于平衡",
    "要结合实际情况",
    "需要综合考虑",
    "因人而异",
    "视情况而定",
    "没有标准答案",
    "看具体情况",
    "没有绝对的对错",
    "综上所述",
    "由此可见",
    "I hope this helps",
)

_TRAILING_PUNCT = "。．.!！?？~～…‥、,，;； \t\r\n"

# 密度项：阈值按篇幅折算（每 DENSITY_BASE 字一档），基准档的允许次数见第三列
_DENSITY_CHECKS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("感叹号", re.compile(r"[！!]"), 3),
    (
        "路标词堆砌",
        re.compile(r"值得注意的是|需要强调的是|更关键的是|事实上|实际上|换句话说|说白了|本质上|归根结底|与此同时"),
        2,
    ),
)

# 固定句式族：标签即模型端修改指令的索引；[^。\n] 限长避免回溯与跨句
# 翻案腔的判据：先否定一个**确定性结论**再给出正解。所以「不是A，而是B」里的 A 不能带
# 「只/仅/光/单」这类限量词——限量否定的意思是「不止于此」，是本轮真正的补全，不是翻案。
# 只作用于以「不是」起首的两条模式；「看似…实则」「与其…不如」是完整翻案句式，不加此豁免。
_NOT_ONLY = r"(?![^。\n？?！!]{0,2}[只仅光单])"
_TIER3_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 核心规则 5 翻案腔：没人主张过的先否定再肯定
    ("翻案腔", re.compile(rf"不是{_NOT_ONLY}[^。\n？?！!]{{0,30}}而是")),
    ("翻案腔", re.compile(rf"不是{_NOT_ONLY}[^。\n？?！!]{{0,16}}(?:[，,]\s*|——+)只是")),
    ("翻案腔", re.compile(r"不是[^。\n？?！!]{0,16}——+是")),
    ("翻案腔", re.compile(r"不在于[^。\n？?！!]{0,16}[，,]\s*而在于")),
    ("翻案腔", re.compile(r"表面(?:上)?[^。\n？?！!]{0,16}[，,]\s*实则")),
    ("翻案腔", re.compile(rf"其实不是{_NOT_ONLY}[^。\n？?！!]{{0,30}}只是")),
    ("翻案腔", re.compile(r"与其[^。\n？?！!]{0,16}不如")),
    ("翻案腔", re.compile(r"与其说[^。\n？?！!]{0,16}不如说")),
    ("翻案腔", re.compile(r"看似[^。\n？?！!]{0,12}实则")),
    ("翻案腔", re.compile(r"真正的问题是")),
    # C2 结尾拔高
    ("结尾拔高", re.compile(r"不仅是[^。\n「」“”]{0,20}更(?:是|关乎)")),
    # 核心规则 9 空转提示语加冒号引列表
    (
        "空转提示语",
        re.compile(r"(?:一句话总结|核心是|原因如下|总结如下|具体分析如下|建议如下|分析如下)\s*[:：]"),
    ),
    # 核心规则 7 揭晓式破折号：右侧补上前句没说出的原因或答案（含 ASCII 双减号写法）
    ("揭示式破折号", re.compile(r"(?:[—–]{1,2}|--)(?:那就是|原来|真正的原因|正因如此)")),
    # 核心规则 7 后置解释：破折号夹一个分句再落回原句（话被打断时不接内容）。
    # 只认破折号**紧跟**解释且解释后回到原句；「他停顿——先等等——再看一眼」式并列停顿不算。
    ("揭示式破折号", re.compile(r"[—–]{1,2}[^—–，,。\n]{1,60}，[^，,。\n]{1,20}[；;]")),
)
_HEDGE_WORDS = r"(?:可能|也许|或许|大概|大致|恐怕|估计)"
# C3 犹豫词：同句出现 ≥2 个不确定标记才算，单个「可能是」是诚实。
# 「可能是，也可能只是」是在给两个并列分叉，属于有效信息，不按堆砌处理。
_HEDGE_SENTENCE_RE = re.compile(rf"{_HEDGE_WORDS}[^。\n]{{0,40}}?{_HEDGE_WORDS}")
_HEDGE_PAIRED_RE = re.compile(rf"{_HEDGE_WORDS}[^。\n]{{0,6}}?[，,]\s*也{_HEDGE_WORDS}")
_FENCED_CODE_RE = re.compile(r"^[ \t]*(```|~~~).*?(?:^[ \t]*\1[^\n]*$)", re.DOTALL | re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_QUOTED_SPAN_RE = re.compile(r"[\"“][^\"”\n]{0,2000}[\"”]|「[^」\n]{0,2000}」|『[^』\n]{0,2000}』")
_NEGATION_RE = re.compile(r"不是|没有|并非|与其|看似|不在于|不算")
# 诊断型提问：针对某一处具体现象要定位（「变量污染的问题在哪」「缓存怎么还没生效」）。
# 这类问法的答案是先排除再定位，「不是 A 而是 B」是排查信息；开放式的「为什么」不算豁免。
_DIAGNOSIS_RE = re.compile(r"哪里|哪儿|怎么(?:会|还|没|不|样)|是否|是不是|会不会|能不能|有没有|原因|的问题")


def _blank(match: re.Match[str]) -> str:
    return re.sub(r"[^\r\n]", " ", match.group(0))


def _mask_code(text: str) -> str:
    """等长遮罩代码块、行内代码与 URL，保持长度与换行不变。"""
    masked = _FENCED_CODE_RE.sub(_blank, text or "")
    masked = _INLINE_CODE_RE.sub(_blank, masked)
    return _URL_RE.sub(_blank, masked)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", _mask_code(text)).strip()


def _find_contained_phrases(text: str, phrases: Sequence[str], *, casefold: bool = False) -> list[str]:
    """保序、精确子串检索。"""
    if not text or not phrases:
        return []
    target = text.casefold() if casefold else text
    return [phrase for phrase in phrases if phrase and (phrase.casefold() if casefold else phrase) in target]


def detect_ending_cliches(text: str) -> list[str]:
    """收尾模板：仅结尾命中，多词命中只报首个。"""
    tail = text.rstrip(_TRAILING_PUNCT).casefold()
    for phrase in DEFAULT_ENDINGS:
        if tail.endswith(phrase.casefold()):
            return [phrase]
    return []


def detect_opening_cliches(text: str) -> list[str]:
    """开场套话：仅首部命中，真实动作开场豁免。"""
    first_clause = OPENER_DELIM.split(text, maxsplit=1)[0].casefold()
    if any(first_clause.startswith(negative) for negative in _OPENING_NEGATIVES):
        return []
    return [phrase for phrase in OPENING_CLICHES if first_clause.startswith(phrase.casefold())]


def detect_density_signals(text: str) -> list[str]:
    """高频标点与路标词：按 300 字基准折算容差。"""
    density_cap = max(1, math.ceil(len(text) / DENSITY_BASE))
    return [
        label for label, pattern, allowance in _DENSITY_CHECKS if len(pattern.findall(text)) > density_cap * allowance
    ]


def detect_fixed_pattern_signals(text: str) -> list[str]:
    return ["然而连发"] if text.count("然而") >= CONSECUTIVE_THRESHOLD else []


def detect_hedge(text: str) -> list[str]:
    """同句 ≥2 个不确定标记，且不是「可能是，也可能」这种并列分叉。"""
    if _HEDGE_PAIRED_RE.search(text):
        return []
    return ["模糊叠加"] if _HEDGE_SENTENCE_RE.search(text) else []


def _user_rationed_patterns(user_text: str) -> frozenset[str]:
    """用户这句话里已经给出的前提，构成允许的纠错语境。

    两种「不是 A 而是 B」是有效信息，不算翻案腔：
    用户自己给了否定/对比前提；或用户指名道姓地要诊断某一处，答案是先排除再定位。
    「为什么」这种开放提问不算：模型借它照样可以拿起翻案腔。
    """
    if not user_text:
        return frozenset()
    if _NEGATION_RE.search(user_text) or _DIAGNOSIS_RE.search(user_text):
        return frozenset({"翻案腔"})
    return frozenset()


def detect_iron_rule(text: str, *, allow: frozenset[str] = frozenset()) -> list[str]:
    """固定句式族：同族多模式只报一个，异族全部报出。allow 里的族被跳过（用户已给出对应前提）。"""
    hits: list[str] = []
    for label, pattern in _TIER3_PATTERNS:
        if label not in hits and label not in allow and pattern.search(text):
            hits.append(label)
    return hits


def signal_names() -> frozenset[str]:
    """检测器可能吐出的全部内置命中名（词表面 + 聚合标签），供评测覆盖度校验。"""
    return frozenset(
        (
            *DEFAULT_AI_CLICHES,
            *OPENING_CLICHES,
            *DEFAULT_SYMPATHY_CLICHES,
            *DEFAULT_ENDINGS,
            *DEFAULT_VAGUE_ATTRIBUTIONS,
            "然而连发",
            *(label for label, _ in _TIER3_PATTERNS),
            "模糊叠加",
            "感叹号",
            "路标词堆砌",
        )
    )


def detect_cliches(text: str, custom_cliches: tuple[str, ...] = (), user_text: str = "") -> list[str]:
    """检测高置信度 AI 腔信号（去重、保序）。

    用户原话里已陈述的否定前提算「主张过」，这一轮的「不是A而是B」是纠错答案，不报翻案腔。
    引号里的他人原话按上游 N13 逐字保留，不参与句式判定。
    """
    normalized = _normalize_text(text)
    if not normalized:
        return []
    allow = _user_rationed_patterns(user_text)
    unquoted = _QUOTED_SPAN_RE.sub(" ", normalized)

    groups = (
        detect_ending_cliches(normalized),
        _find_contained_phrases(normalized, DEFAULT_AI_CLICHES),
        detect_opening_cliches(normalized),
        _find_contained_phrases(normalized, DEFAULT_SYMPATHY_CLICHES),
        _find_contained_phrases(normalized, DEFAULT_VAGUE_ATTRIBUTIONS),
        _find_contained_phrases(normalized, custom_cliches, casefold=True),
        detect_fixed_pattern_signals(normalized),
        detect_iron_rule(unquoted, allow=allow),
        detect_hedge(unquoted),
        detect_density_signals(normalized),
    )
    hits: list[str] = []
    for signals in groups:
        for signal in signals:
            if signal not in hits:
                hits.append(signal)
    return hits


# 危害档位：1 = 损害回答可靠性，2 = 仅影响观感（默认）
_PRIORITY_1_SIGNALS: frozenset[str] = frozenset(
    (*DEFAULT_AI_CLICHES, *DEFAULT_SYMPATHY_CLICHES, *DEFAULT_VAGUE_ATTRIBUTIONS, "翻案腔")
)


def signal_priority(name: str) -> int:
    return 1 if name in _PRIORITY_1_SIGNALS else 2
