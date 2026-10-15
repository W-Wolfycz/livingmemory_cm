"""Tests for plugin LLM tool registration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from livingmemory_cm.core.base.config_manager import ConfigManager
from livingmemory_cm.core.tools import MemorySearchTool
from livingmemory_cm.main import (
    LivingMemoryCMPlugin,
    _parse_version,
    _version_lt,
)
from livingmemory_cm.version import PLUGIN_REPOSITORY


ROOT = Path(__file__).resolve().parents[1]


def test_register_llm_tools_is_idempotent():
    """Agent 检索工具恒开；重复调用只注册一次。"""
    plugin = LivingMemoryCMPlugin.__new__(LivingMemoryCMPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager()
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()
    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_called_once()
    tools = plugin.context.add_llm_tools.call_args.args
    assert [tool.name for tool in tools] == ["recall_long_term_memory"]
    assert isinstance(tools[0], MemorySearchTool)
    assert plugin._llm_tools_registered is True


@pytest.mark.asyncio
async def test_handle_session_reset_accepts_new_and_legacy_signals():
    """/reset、/new：接受 AstrBot 4.27.x 新信号与旧信号，无信号时跳过。"""
    plugin = LivingMemoryCMPlugin.__new__(LivingMemoryCMPlugin)
    plugin._ensure_plugin_ready = AsyncMock(return_value=(True, None))
    handler = SimpleNamespace(handle_session_reset=AsyncMock())
    plugin.event_handler = handler

    def _event(extra: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            get_extra=lambda key, default=False: key == extra
        )

    await plugin.handle_session_reset(_event("_clean_group_context_session"))
    await plugin.handle_session_reset(_event("_clean_ltm_session"))
    assert handler.handle_session_reset.await_count == 2

    await plugin.handle_session_reset(_event(None))
    assert handler.handle_session_reset.await_count == 2
