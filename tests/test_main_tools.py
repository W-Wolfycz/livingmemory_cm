"""Tests for plugin LLM tool registration."""

from pathlib import Path
from unittest.mock import Mock

from livingmemory_cm.core.base.config_manager import ConfigManager
from livingmemory_cm.core.tools import MemoryMemorizeTool, MemorySearchTool
from livingmemory_cm.main import (
    LivingMemoryCMPlugin,
    _parse_version,
    _version_lt,
)
from livingmemory_cm.version import PLUGIN_REPOSITORY


ROOT = Path(__file__).resolve().parents[1]


def test_parse_version_accepts_source_and_prerelease_versions():
    assert _parse_version("4.25.2") == (4, 25, 2)
    assert _parse_version("v4.25.2-beta.1") == (4, 25, 2)
    assert _parse_version("not-a-version") == ()


def test_version_lt_pads_version_segments():
    assert _version_lt("4.24.1", "4.24.2") is True
    assert _version_lt("4.25", "4.25.0") is False
    assert _version_lt("4.25.2", "4.24.2") is False
    assert _version_lt("unknown", "4.24.2") is False


def test_distribution_metadata_and_required_notices_are_consistent():
    expected_repository = "https://github.com/W-Wolfycz/livingmemory_cm"
    assert PLUGIN_REPOSITORY == expected_repository

    metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    help_catalog = (ROOT / "core" / "i18n" / "zh.json").read_text(
        encoding="utf-8"
    )
    dashboard = (ROOT / "pages" / "dashboard" / "index.html").read_text(
        encoding="utf-8"
    )
    sync_script = (ROOT / "sync.sh").read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for content in (metadata, readme, notice, help_catalog, dashboard):
        assert expected_repository in content
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (
        ROOT / "LICENSE"
    ).read_text(encoding="utf-8")
    assert "ISC License" in (
        ROOT / "pages" / "dashboard" / "vendor" / "LUCIDE_LICENSE"
    ).read_text(encoding="utf-8")
    assert "## [2.5.7-cm]" in changelog
    for required in (
        "LICENSE",
        "NOTICE.md",
        "README.md",
        "CHANGELOG.md",
        "pages/dashboard/vendor/LUCIDE_LICENSE",
    ):
        assert f'"{required}"' in sync_script
    for ignored in (
        ".venv-test/",
        "__pycache__/",
        "/data/",
        "/core.zip",
        "*.db",
        "*.index",
    ):
        assert ignored in gitignore


def test_register_llm_tools_is_idempotent():
    plugin = LivingMemoryCMPlugin.__new__(LivingMemoryCMPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager(
        {"agent_tools": {"enable_recall_tool": True, "enable_memorize_tool": True}}
    )
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()
    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_called_once()
    tools = plugin.context.add_llm_tools.call_args.args
    tools_by_name = {tool.name: tool for tool in tools}
    assert set(tools_by_name) == {
        "recall_long_term_memory",
        "memorize_long_term_memory",
    }
    assert isinstance(tools_by_name["recall_long_term_memory"], MemorySearchTool)
    assert isinstance(tools_by_name["memorize_long_term_memory"], MemoryMemorizeTool)
    assert plugin._llm_tools_registered is True


def test_register_llm_tools_defaults_only_recall():
    plugin = LivingMemoryCMPlugin.__new__(LivingMemoryCMPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager()
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_called_once()
    tools = plugin.context.add_llm_tools.call_args.args
    assert [tool.name for tool in tools] == ["recall_long_term_memory"]
    assert isinstance(tools[0], MemorySearchTool)
    assert plugin._llm_tools_registered is True


def test_register_llm_tools_no_memory_engine():
    plugin = LivingMemoryCMPlugin.__new__(LivingMemoryCMPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager()
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = None
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_not_called()
    assert plugin._llm_tools_registered is False


def test_register_llm_tools_no_memory_processor():
    plugin = LivingMemoryCMPlugin.__new__(LivingMemoryCMPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager()
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = None
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_not_called()
    assert plugin._llm_tools_registered is False


def test_register_llm_tools_respects_recall_tool_disabled():
    plugin = LivingMemoryCMPlugin.__new__(LivingMemoryCMPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager(
        {"agent_tools": {"enable_recall_tool": False, "enable_memorize_tool": True}}
    )
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_called_once()
    tools = plugin.context.add_llm_tools.call_args.args
    assert [tool.name for tool in tools] == ["memorize_long_term_memory"]
    assert isinstance(tools[0], MemoryMemorizeTool)
    assert plugin._llm_tools_registered is True


def test_register_llm_tools_respects_memorize_tool_disabled():
    plugin = LivingMemoryCMPlugin.__new__(LivingMemoryCMPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager(
        {"agent_tools": {"enable_recall_tool": True, "enable_memorize_tool": False}}
    )
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_called_once()
    tools = plugin.context.add_llm_tools.call_args.args
    assert [tool.name for tool in tools] == ["recall_long_term_memory"]
    assert isinstance(tools[0], MemorySearchTool)
    assert plugin._llm_tools_registered is True


def test_register_llm_tools_respects_all_tools_disabled():
    plugin = LivingMemoryCMPlugin.__new__(LivingMemoryCMPlugin)
    plugin.context = Mock()
    plugin.config_manager = ConfigManager(
        {"agent_tools": {"enable_recall_tool": False, "enable_memorize_tool": False}}
    )
    plugin.initializer = Mock()
    plugin.initializer.memory_engine = Mock()
    plugin.initializer.memory_processor = Mock()
    plugin._llm_tools_registered = False

    plugin._register_agent_tools_if_needed()

    plugin.context.add_llm_tools.assert_not_called()
    assert plugin._llm_tools_registered is True
