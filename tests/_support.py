"""共享测试替身：只模拟插件实际用到的宿主接口形状。"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "astrbot_plugin_natural_chat"


def ensure_plugin_package() -> None:
    """按规范包名加载源码树。"""
    expected = (ROOT / "__init__.py").resolve()
    current = sys.modules.get(PACKAGE)
    current_file = getattr(current, "__file__", None)
    if current_file and Path(current_file).resolve() == expected:
        return
    spec = importlib.util.spec_from_file_location(PACKAGE, expected, submodule_search_locations=[str(ROOT)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin package from {ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)


def stub_astrbot_logger() -> None:
    """无宿主环境时补一个 astrbot.api.logger，供模块导入。"""
    try:
        import astrbot.api  # noqa: F401
    except ImportError:
        package = types.ModuleType("astrbot")
        api = types.ModuleType("astrbot.api")
        import logging

        api.logger = logging.getLogger("astrbot")
        package.api = api
        sys.modules["astrbot"] = package
        sys.modules["astrbot.api"] = api


class FakePart:
    """TextPart 替身：故意不实现 __eq__，与宿主 dataclass 行为保持一致。"""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeEvent:
    def __init__(self, text: str = "", origin: str = "test:GroupMessage:1", group_id: str = "1") -> None:
        self.message_str = text
        self.unified_msg_origin = origin
        self._group_id = group_id

    def get_group_id(self) -> str:
        return self._group_id


class FakeResp:
    def __init__(self, text: str = "", *, completion: bool = True) -> None:
        self.completion_text = text if completion else ""
        self.result_chain = None if completion else text


class FakeReq:
    """ProviderRequest 替身：每轮新建，字段可缺省。"""

    def __init__(
        self,
        system_prompt: str | None = "原人设",
        contexts: list[dict[str, object]] | None = None,
        parts: list[object] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.contexts = contexts if contexts is not None else []
        self.extra_user_content_parts = parts
