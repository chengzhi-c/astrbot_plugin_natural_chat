from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .constants import (
    DEFAULT_RECENT_REPLY_WINDOW,
    DEFAULT_STATE_RETENTION_DAYS,
    MAX_RECENT_REPLY_WINDOW,
    MAX_RUNTIME_HINT_CHARS,
    MAX_STATE_RETENTION_DAYS,
    MIN_RECENT_REPLY_WINDOW,
    MIN_RUNTIME_HINT_CHARS,
    MIN_STATE_RETENTION_DAYS,
    PENDING_SESSION_CAP,
    YIELD_STICKY_TTL_SECONDS,
)
from .protocols import LLMResponseProtocol, MessageEventProtocol, ProviderRequestProtocol
from .rules import (
    RUNTIME_HINT_MARKER,
    STABLE_RULE_MARKER,
    ContextRewriteResult,
    StableRewriteResult,
    append_temp_text_part,
    render_runtime_hint,
    rewrite_context_injections,
    rewrite_stable_rules,
    select_runtime_hint_names,
)
from .runtime_state import RuntimeStateStore, is_session_disabled, unified_origin
from .scene_guard import event_text, is_sticky_followup, is_tone_request
from .signal_detectors import detect_cliches, signal_priority

logger = logging.getLogger(__name__)


@dataclass
class QualityStats:
    """质量层累计统计（进程内，不持久化）。"""

    total_injections: int = 0
    stable_rules_injected: int = 0
    runtime_hints_injected: int = 0
    avoid_openers_seen: int = 0
    signal_hits: dict[str, int] = field(default_factory=dict)
    stale_blocks_removed: int = 0

    def record_request(self, stable_injected: bool, hint_injected: bool) -> None:
        if stable_injected:
            self.stable_rules_injected += 1
        if hint_injected:
            self.runtime_hints_injected += 1
        if stable_injected or hint_injected:
            self.total_injections += 1

    def top_signals(self, limit: int = 5) -> list[tuple[str, int]]:
        return sorted(self.signal_hits.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):  # YAML 里把 int 项写成 true/false 时退回默认，避免被钳成边界值
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.splitlines()
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


@dataclass(frozen=True)
class AppConfig:
    """Construct-once configuration snapshot."""

    enabled: bool = True
    inject_stable_rules: bool = True
    inject_runtime_state: bool = True
    debug_log: bool = False
    max_runtime_hint_chars: int = MAX_RUNTIME_HINT_CHARS
    state_retention_days: int = DEFAULT_STATE_RETENTION_DAYS
    recent_reply_window: int = DEFAULT_RECENT_REPLY_WINDOW
    custom_cliches: tuple[str, ...] = ()
    disabled_sessions: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, config: Any) -> AppConfig:
        get = (config if config is not None else {}).get

        return cls(
            enabled=_parse_bool(get("enabled", True), True),
            inject_stable_rules=_parse_bool(get("inject_stable_rules", True), True),
            inject_runtime_state=_parse_bool(get("inject_runtime_state", True), True),
            debug_log=_parse_bool(get("debug_log", False), False),
            max_runtime_hint_chars=_parse_int(
                get("max_runtime_hint_chars", MAX_RUNTIME_HINT_CHARS),
                MAX_RUNTIME_HINT_CHARS,
                MIN_RUNTIME_HINT_CHARS,
                MAX_RUNTIME_HINT_CHARS,
            ),
            state_retention_days=_parse_int(
                get("state_retention_days", DEFAULT_STATE_RETENTION_DAYS),
                DEFAULT_STATE_RETENTION_DAYS,
                MIN_STATE_RETENTION_DAYS,
                MAX_STATE_RETENTION_DAYS,
            ),
            recent_reply_window=_parse_int(
                get("recent_reply_window", DEFAULT_RECENT_REPLY_WINDOW),
                DEFAULT_RECENT_REPLY_WINDOW,
                MIN_RECENT_REPLY_WINDOW,
                MAX_RECENT_REPLY_WINDOW,
            ),
            custom_cliches=tuple(_parse_list(get("custom_cliches", []))),
            disabled_sessions=frozenset(item.lower() for item in _parse_list(get("disabled_sessions", []))),
        )


def extract_response_text(resp: LLMResponseProtocol) -> str:
    """取模型回复正文。completion_text 是宿主聚合后的纯文本，其余形态按部件兜底。"""
    completion = getattr(resp, "completion_text", None)
    if isinstance(completion, str) and completion.strip():
        return completion.strip()
    chain = getattr(resp, "result_chain", None)
    if isinstance(chain, str):
        return chain.strip()
    items = getattr(chain, "chain", chain)
    if isinstance(items, (list, tuple)):
        texts = [getattr(item, "text", None) for item in items]
        return " ".join(item for item in texts if isinstance(item, str)).strip()
    return ""


_YIELD_REASONS = {
    "copy": "- 复制粘贴场景让位（成品文本不改写）",
    "tone": "- 语气改写请求让位（这一轮改的不是对话腔）",
}


class NaturalChatCore:
    """Host-independent request, response, and session behavior."""

    def __init__(
        self,
        config: AppConfig,
        store: RuntimeStateStore,
        text_part_factory: Any | None = None,
    ) -> None:
        self.cfg = config
        self.store = store
        self.text_part_factory = text_part_factory
        self.stats = QualityStats()
        self._pending_yield: dict[str, tuple[float, str]] = {}

    async def on_llm_request(self, event: MessageEventProtocol, req: ProviderRequestProtocol) -> None:
        session_id = unified_origin(event)
        yield_reason = self._yield_reason(session_id, event, update=True)
        effective_active = not yield_reason and self._is_effectively_active(session_id, event)

        hint = ""
        selected_names: tuple[str, ...] = ()
        if effective_active and self.cfg.inject_runtime_state and self.text_part_factory is not None:
            # 危害排序与入库截断同口径：档 1（可靠性损害）先占名额。
            # 落盘的自定义词是序号标签，注入前还原成原文；还原不了（配置已改）就跳过。
            avoid_sorted = sorted(
                (name for name in map(self.store.resolve_name, self.store.get(session_id).avoid_openers) if name),
                key=signal_priority,
            )
            selected_names = tuple(select_runtime_hint_names(avoid_sorted, self.cfg.max_runtime_hint_chars))
            hint = render_runtime_hint(selected_names)

        context_result = rewrite_context_injections(req, hint or None)
        # 孤儿块（ambiguous）只影响"是否清理"，不影响"是否注入"：历史里存在无法核验的自家旧块时，
        # 仍必须照常注入新提醒，否则该会话动态提醒永久静默（且该块按设计永不被清理，无法自愈）。
        injected_hint = (
            hint
            if hint
            and not context_result.runtime_satisfied
            and append_temp_text_part(req, hint, self.text_part_factory, marker=RUNTIME_HINT_MARKER)
            else ""
        )

        # 只写回宿主原始 prompt：开关关闭时若把空串写回，会连宿主自己（含人设）的 prompt 一起清空。
        before = getattr(req, "system_prompt", None)
        if isinstance(before, str):
            stable_result = rewrite_stable_rules(before, enabled=effective_active and self.cfg.inject_stable_rules)
            if stable_result.text != before:
                req.system_prompt = stable_result.text
        else:
            stable_result = StableRewriteResult(text="", injected=False, removed=0, ambiguous=False)

        self.stats.record_request(stable_result.injected, bool(injected_hint))
        self.stats.stale_blocks_removed += (
            stable_result.removed + context_result.stable_removed + context_result.runtime_removed
        )

        if self.cfg.debug_log:
            self._log_injection(session_id or "<unknown>", stable_result, context_result, injected_hint)

    def _log_injection(
        self,
        session_id: str,
        stable_result: StableRewriteResult,
        context_result: ContextRewriteResult,
        injected_hint: str,
    ) -> None:
        # 用 info 而非 debug：debug_log 开关已是显式开启，再打 debug 会被日志级别二次过滤成不可观察。
        # 不打印提示正文与避用项：其中可能含用户配置的自定义词与模型回复片段。
        logger.info("injection rewrite for %s", session_id)
        if stable_result.injected:
            logger.info("stable rules injected into system_prompt (marker=%s)", STABLE_RULE_MARKER)
        if injected_hint:
            logger.info("runtime hint injected for %s", session_id)
        if stable_result.removed or context_result.stable_removed or context_result.runtime_removed:
            logger.info("stale own injection removed for %s", session_id)
        if stable_result.ambiguous or context_result.runtime_ambiguous:
            logger.info("ambiguous own marker kept for %s", session_id)

    async def on_llm_response(self, event: MessageEventProtocol, resp: LLMResponseProtocol) -> None:
        session_id = unified_origin(event)
        if (
            not session_id
            # 回复正文不参与让位分类，避免模型复述体裁词改写本轮请求写入的粘性原因。
            or self._yield_reason(session_id, event, update=False)
            or not self._is_effectively_active(session_id, event)
        ):
            return
        text = extract_response_text(resp)
        if not text:
            return

        cliches = detect_cliches(text, self.store.custom_cliches, user_text=event_text(event))
        for cliche in cliches:
            self.stats.signal_hits[cliche] = self.stats.signal_hits.get(cliche, 0) + 1

        # 入库截断在 store 内按 MAX_AVOID_ITEMS 执行，故可靠性信号须先排序再传入，
        # 否则档 1 会被同轮的观感项挤出名额（注入侧排序发生在截断之后，救不回来）。
        ordered = tuple(sorted(cliches, key=signal_priority))
        self.stats.avoid_openers_seen += await self.store.record_response(session_id, text, ordered)

        if self.cfg.debug_log:
            # 只记数量：命中项里可能有用户配置的自定义词原文
            logger.info("response recorded for %s: %d signals", session_id, len(ordered))

    async def set_session_enabled(self, session_id: str, enabled: bool) -> bool:
        return await self.store.set_enabled(session_id, enabled)

    async def reset_session(self, session_id: str) -> bool:
        return await self.store.reset(session_id)

    def status_text(self, session_id: str, event: MessageEventProtocol | None = None) -> str:
        persistence = "待重试" if self.store.has_pending_save else "正常"
        reasons: list[str] = []
        if not session_id:
            reasons.append("- 无法识别当前会话标识")
        if not self.cfg.enabled:
            reasons.append("- 全局配置：关闭")
        if session_id and not self.store.is_enabled(session_id):
            reasons.append("- 当前会话：已通过 /natural off 关闭")
        if session_id and is_session_disabled(self.cfg.disabled_sessions, session_id, event):
            reasons.append("- 配置静态禁用：当前会话命中禁用列表")
        yield_reason = self._yield_reason(session_id, event)
        if yield_reason:
            reasons.append(yield_reason)
        if reasons:
            return "\n".join(["Natural Chat 状态：", *reasons, "- 无运行时状态", f"- 状态持久化：{persistence}"])

        state = self.store.get(session_id)
        resolved = [name for name in map(self.store.resolve_name, state.avoid_openers) if name]
        avoid = "、".join(resolved) if resolved else "无（尚未形成重复或套话信号）"
        if not self.cfg.inject_runtime_state:
            runtime_line = "- 运行时提示：配置关闭"
        elif self.text_part_factory is None:
            runtime_line = "- 运行时提示：已配置，但宿主临时文本部件不可用"
        else:
            runtime_line = "- 运行时提示：启用"
        lines = [
            "Natural Chat 状态：",
            "- 当前会话：启用",
            f"- 固定规则：{'启用' if self.cfg.inject_stable_rules else '配置关闭'}（system_prompt）",
            runtime_line,
            f"- 下一轮避用：{avoid}",
        ]
        ignored = dict(self.store.custom_cliches_ignored_reasons)
        if ignored:
            details = "、".join(
                f"{label} {ignored[reason]} 项"
                for reason, label in (("empty", "空值"), ("duplicate", "重复"), ("too_long", "过长"))
                if ignored.get(reason)
            )
            lines.append(f"- 配置忽略：{sum(ignored.values())} 项（{details}）")
        if resolved and self.cfg.inject_runtime_state and self.text_part_factory is not None:
            lines.append("- 下一轮请求会带上动态提醒")
        lines.append(f"- 自启动以来累计注入：{self.stats.total_injections} 次")
        lines.append(f"- 状态持久化：{persistence}")
        return "\n".join(lines)

    def _yield_reason(self, session_id: str, event: MessageEventProtocol | None, *, update: bool = False) -> str | None:
        """让位判定。粘性只认「接着改」这类续写口令，窗口从首次让位算起，不因口令刷新而无限延长。"""
        text = event_text(event)
        kind = is_tone_request(text)
        now = time.monotonic()
        sticky = self._pending_yield.get(session_id) if session_id else None
        # 空文本（纯图片、适配器取不到正文）不能算续写口令，否则会在窗口内吞掉一轮正常请求
        if (
            kind is None
            and text
            and sticky
            and now - sticky[0] <= YIELD_STICKY_TTL_SECONDS
            and is_sticky_followup(text)
        ):
            kind = sticky[1]
        elif kind is None and update and session_id:
            self._pending_yield.pop(session_id, None)
        if kind and update and session_id and session_id not in self._pending_yield:
            self._pending_yield[session_id] = (now, kind)
            self._evict_yield_if_needed()
        return _YIELD_REASONS[kind] if kind else None

    def _evict_yield_if_needed(self) -> None:
        # 有界淘汰：超 cap 逐出最久未刷新的会话（dict 保插入序，原地更新不移动位置，不能直接 pop 首个）
        while len(self._pending_yield) > PENDING_SESSION_CAP:
            oldest = min(self._pending_yield, key=lambda key: self._pending_yield[key][0])
            del self._pending_yield[oldest]

    def _is_effectively_active(self, session_id: str, event: MessageEventProtocol | None = None) -> bool:
        return (
            bool(session_id)
            and self.cfg.enabled
            and self.store.is_enabled(session_id)
            and not is_session_disabled(self.cfg.disabled_sessions, session_id, event)
        )
