from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools

from .core import AppConfig, NaturalChatCore
from .rules import STABLE_RULE_MARKER, build_stable_rules
from .runtime_state import RuntimeStateStore

PLUGIN_ID = "astrbot_plugin_natural_chat"


def _version_from_lines(lines: list[str]) -> str:
    """纯函数：从 metadata.yaml 行文本提取 version（可测）。"""
    for line in lines:
        text = line.strip()
        if text.startswith("version:"):
            value = text.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value
    return "0.0.0"


def _read_metadata_version() -> str:
    try:
        return _version_from_lines(Path(__file__).with_name("metadata.yaml").read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return "0.0.0"


PLUGIN_VERSION = _read_metadata_version()


_TEXT_PART_CLS: Any | None = None
_TEXT_PART_WARNED = False


def _probe_text_part_cls() -> Any | None:
    """探测 TextPart 类，成功即缓存。

    TextPart 未出现在 astrbot.api 的公开导出里，只能走内部路径，因此探测失败要能重试，
    不能用 lru_cache 把失败结果钉死到进程结束。失败告警只打一次：工厂每轮都会被调用，
    不复用告警会把日志刷满。
    """
    global _TEXT_PART_CLS, _TEXT_PART_WARNED
    if _TEXT_PART_CLS is None:
        try:
            from astrbot.core.agent.message import TextPart

            _TEXT_PART_CLS = TextPart
        except Exception as e:
            if not _TEXT_PART_WARNED:
                _TEXT_PART_WARNED = True
                logger.warning(f"[NaturalChat] TextPart unavailable, runtime hint injection disabled: {e}")
    return _TEXT_PART_CLS


def _make_text_part(*, text: str) -> Any | None:
    """把探测到的 TextPart 类包成工厂。

    core 侧按 `factory(text=...)` 调用；探测失败返回 None，由调用方降级。
    每次调用都重新探测，宿主升级或热重载后不必重启进程。
    """
    cls = _probe_text_part_cls()
    return cls(text=text) if cls is not None else None


LEGACY_PLUGIN_ID = "astrbot_plugin_human_chat_quality"
_UNSAVED_SUFFIX = "，但状态文件写入失败；已保留待重试状态，请再次执行命令或检查数据目录。"


def _reply(saved: bool, success: str, done: str) -> str:
    return success if saved else f"{done}{_UNSAVED_SUFFIX}"


def _migrate_legacy_data_dir(data_dir: Path) -> None:
    """把改名前数据目录里的状态文件搬过来，避免升级丢会话状态。

    只在目标位置还没有状态文件时搬运，且源文件保留：迁移失败或用户回滚旧版本时两边都还在。
    """
    target = data_dir / "runtime_state.json"
    if target.exists():
        return
    # 宿主 get_data_dir 只抛 RuntimeError/ValueError；宽捕获会把拼错的变量名一起吞掉
    try:
        legacy_dir = Path(StarTools.get_data_dir(LEGACY_PLUGIN_ID))
    except (RuntimeError, ValueError) as e:
        logger.debug(f"[NaturalChat] legacy data dir unavailable: {e}")
        return
    legacy = legacy_dir / "runtime_state.json"
    if not legacy.is_file():
        return
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, target)
    except OSError as e:
        logger.warning(f"[NaturalChat] legacy state migration failed: {e}")
        return
    logger.info(f"[NaturalChat] migrated legacy state from {legacy}")


class NaturalChatPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None):
        super().__init__(context)
        cfg = AppConfig.from_config(config)
        data_dir = Path(StarTools.get_data_dir(PLUGIN_ID))
        _migrate_legacy_data_dir(data_dir)
        self.store = RuntimeStateStore(
            data_dir / "runtime_state.json",
            retention_days=cfg.state_retention_days,
            recent_reply_window=cfg.recent_reply_window,
            custom_cliches=cfg.custom_cliches,
        )
        # 传工厂而非探测结果：探测失败要在下一轮还能重试，不能把 None 钉死在 core 里。
        self.core = NaturalChatCore(cfg, self.store, text_part_factory=_make_text_part)
        logger.info(
            f"[NaturalChat] plugin loaded, version={PLUGIN_VERSION}, "
            f"sessions={len(self.store.sessions)}, custom_cliches={len(self.store.custom_cliches)}, "
            f"custom_cliches_ignored={self.store.custom_cliches_ignored}"
        )

    @filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        # 质量层是增强功能：任何内部异常都不应阻断消息主链
        try:
            await self.core.on_llm_request(event, req)
        except Exception as e:
            logger.error(f"[NaturalChat] on_llm_request failed: {e}", exc_info=True)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse) -> None:
        try:
            await self.core.on_llm_response(event, resp)
        except Exception as e:
            logger.error(f"[NaturalChat] on_llm_response failed: {e}", exc_info=True)

    @filter.command_group("natural")
    def natural(self):
        """指令组 natural：会话控制。全部子命令仅管理员可用；除 rules 外均作用于当前会话。"""

    @permission_type(PermissionType.ADMIN)
    @natural.command("status")
    async def natural_status(self, event: AstrMessageEvent):
        """查看当前会话的质量层状态与累计注入次数"""
        yield event.plain_result(self.core.status_text(event.unified_msg_origin, event))

    @permission_type(PermissionType.ADMIN)
    @natural.command("on")
    async def natural_on(self, event: AstrMessageEvent):
        """启用当前会话的质量层"""
        saved = await self.core.set_session_enabled(event.unified_msg_origin, True)
        yield event.plain_result(_reply(saved, "Natural Chat 已启用当前会话。", "Natural Chat 当前进程内已启用"))

    @permission_type(PermissionType.ADMIN)
    @natural.command("off")
    async def natural_off(self, event: AstrMessageEvent):
        """关闭当前会话的质量层，直到再次执行 on"""
        saved = await self.core.set_session_enabled(event.unified_msg_origin, False)
        yield event.plain_result(_reply(saved, "Natural Chat 已关闭当前会话。", "Natural Chat 当前进程内已关闭"))

    @permission_type(PermissionType.ADMIN)
    @natural.command("reset")
    async def natural_reset(self, event: AstrMessageEvent):
        """清空当前会话的提醒记录（重复开头与避用词）"""
        saved = await self.core.reset_session(event.unified_msg_origin)
        yield event.plain_result(
            _reply(
                saved,
                "Natural Chat 已清空当前会话的提醒记录；不会删除人设或历史里的旧规则字。",
                "Natural Chat 当前进程内已清空",
            )
        )

    @permission_type(PermissionType.ADMIN)
    @natural.command("rules")
    async def natural_rules(self, event: AstrMessageEvent):
        """查看固定规则原文"""
        yield event.plain_result(build_stable_rules())

    @permission_type(PermissionType.ADMIN)
    @natural.command("stats")
    async def natural_stats(self, event: AstrMessageEvent):
        """查看质量层累计统计"""
        stats = self.core.stats
        lines = [
            "Natural Chat 统计（本次运行）：",
            f"- 累计注入：{stats.total_injections} 次",
            f"  └ 固定规则：{stats.stable_rules_injected} 次",
            f"  └ 动态提醒：{stats.runtime_hints_injected} 次",
            f"- 记录避用项：{stats.avoid_openers_seen} 项次",
            f"- 清理旧注入块：{stats.stale_blocks_removed} 次（宿主每轮重建旧块时会重复计入）",
        ]
        top_signals = stats.top_signals(5)
        if top_signals:
            lines.append("- 高频信号 TOP 5：")
            lines.extend(f"  {count:>3} 次 {signal}" for signal, count in top_signals)

        yield event.plain_result("\n".join(lines))

    async def terminate(self) -> None:
        if not await self.store.terminate():
            logger.warning("[NaturalChat] pending runtime state could not be persisted during terminate")
        logger.info(
            "[NaturalChat] terminated, total injections this run: "
            f"{self.core.stats.total_injections}; marker={STABLE_RULE_MARKER}"
        )
