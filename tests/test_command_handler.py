"""
Tests for CommandHandler.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from livingmemory_cm.core.base.config_manager import ConfigManager
from livingmemory_cm.core.command_handler import CommandHandler
from livingmemory_cm.core.i18n_backend import init as init_i18n


@pytest.fixture(autouse=True)
def initialize_i18n():
    init_i18n()


@pytest.fixture
def config_manager():
    return ConfigManager()


@pytest.fixture
def memory_engine():
    engine = Mock()
    engine.db_path = "/tmp/livingmemory-test.db"
    engine.get_statistics = AsyncMock(
        return_value={
            "total_memories": 2,
            "sessions": {"s1": 1, "s2": 1},
            "newest_memory": 1_700_000_000.0,
        }
    )
    engine.search_memories = AsyncMock(return_value=[])
    engine.delete_memory = AsyncMock(return_value=True)
    engine.rebuild_graph_index = AsyncMock(return_value={"rebuilt": 0, "skipped": 0})
    return engine


@pytest.fixture
def conversation_manager():
    manager = Mock()
    manager.clear_session = AsyncMock()
    return manager


@pytest.fixture
def mock_event():
    event = Mock()
    event.unified_msg_origin = "platform:private:10001"
    event.plain_result = Mock(side_effect=lambda message: message)
    return event


@pytest.fixture
def handler(config_manager, memory_engine, conversation_manager):
    context = Mock()
    return CommandHandler(
        context=context,
        config_manager=config_manager,
        memory_engine=memory_engine,
        conversation_manager=conversation_manager,
        initialization_status_callback=lambda: "ready",
    )


@pytest.mark.asyncio
async def test_handle_status_returns_report(handler, mock_event):
    messages = [msg async for msg in handler.handle_status(mock_event)]
    assert len(messages) == 1
    assert "LivingMemory" in messages[0]
    assert "总记忆数" in messages[0]


@pytest.mark.asyncio
async def test_handle_status_reports_legacy_graph_migration(
    handler, mock_event, memory_engine
):
    memory_engine.get_statistics = AsyncMock(
        return_value={
            "total_memories": 2,
            "sessions": {"s1": 2},
            "newest_memory": 1_700_000_000.0,
            "graph_index": {
                "state": "rebuild_required",
                "total_vectors": 9,
                "memory_vectors": 2,
                "legacy_vectors": 7,
                "orphan_vectors": 0,
            },
        }
    )

    messages = [msg async for msg in handler.handle_status(mock_event)]

    assert "需要迁移" in messages[0]
    assert "旧 entry 级向量: 7" in messages[0]
    assert "/lmem rebuild-graph" in messages[0]


@pytest.mark.asyncio
async def test_handle_status_without_engine_returns_actionable_message(
    config_manager, mock_event
):
    handler = CommandHandler(
        context=Mock(),
        config_manager=config_manager,
        memory_engine=None,
        conversation_manager=None,
    )

    messages = [msg async for msg in handler.handle_status(mock_event)]
    assert len(messages) == 1
    assert "/lmem status 执行失败" in messages[0]
    assert "检查插件状态" in messages[0]


@pytest.mark.asyncio
async def test_handle_status_error_contains_suggestions(
    handler, mock_event, memory_engine
):
    memory_engine.get_statistics = AsyncMock(side_effect=RuntimeError("db unavailable"))

    messages = [msg async for msg in handler.handle_status(mock_event)]
    assert len(messages) == 1
    assert "获取状态失败" in messages[0]
    assert "建议排查" in messages[0]
    assert "数据库文件可读写" in messages[0]


@pytest.mark.asyncio
async def test_handle_search_validates_inputs_and_calls_engine(handler, mock_event):
    empty = [msg async for msg in handler.handle_search(mock_event, "", 3)]
    assert "不能为空" in empty[0]

    _ = [msg async for msg in handler.handle_search(mock_event, "hello", 200)]
    # k should be clamped to 100.
    handler.memory_engine.search_memories.assert_awaited_with(
        query="hello", k=100, session_id=mock_event.unified_msg_origin
    )


@pytest.mark.asyncio
async def test_handle_search_renders_results(handler, mock_event, memory_engine):
    result = Mock(doc_id=7, final_score=0.88, content="hello memory")
    memory_engine.search_memories = AsyncMock(return_value=[result])

    messages = [msg async for msg in handler.handle_search(mock_event, "hello", 5)]
    assert len(messages) == 1
    assert "找到 1 条相关记忆" in messages[0]
    assert "ID: 7" in messages[0]


@pytest.mark.asyncio
async def test_handle_forget_success_and_not_found(handler, mock_event, memory_engine):
    success = [msg async for msg in handler.handle_forget(mock_event, 10)]
    assert "已删除记忆 #10" in success[0]

    memory_engine.delete_memory = AsyncMock(return_value=False)
    failed = [msg async for msg in handler.handle_forget(mock_event, 11)]
    assert "删除失败" in failed[0]


@pytest.mark.asyncio
async def test_handle_reset_and_help(handler, mock_event, conversation_manager):
    reset = [msg async for msg in handler.handle_reset(mock_event)]
    assert "已重置" in reset[0]
    conversation_manager.clear_session.assert_awaited_once()

    help_msg = [msg async for msg in handler.handle_help(mock_event)]
    assert "/lmem status" in help_msg[0]
    assert (
        "https://github.com/W-Wolfycz/livingmemory_cm"
        in help_msg[0]
    )
    assert "AGPL-3.0" in help_msg[0]


@pytest.mark.asyncio
async def test_handle_webui_shows_guide(handler, mock_event):
    messages = [msg async for msg in handler.handle_webui(mock_event)]
    assert len(messages) == 1
    assert "AstrBot" in messages[0]
    assert "Plugins" in messages[0] or "插件" in messages[0]
    assert "Pages -> dashboard" in messages[0]


@pytest.mark.asyncio
async def test_handle_cleanup_invalid_history_json_returns_clear_error(
    config_manager, memory_engine, conversation_manager, mock_event
):
    context = Mock()
    context.conversation_manager = Mock()
    context.conversation_manager.get_curr_conversation_id = AsyncMock(
        return_value="cid-1"
    )
    context.conversation_manager.get_conversation = AsyncMock(
        return_value=Mock(history="{bad json")
    )
    context.conversation_manager.update_conversation = AsyncMock()

    handler = CommandHandler(
        context=context,
        config_manager=config_manager,
        memory_engine=memory_engine,
        conversation_manager=conversation_manager,
    )

    messages = [msg async for msg in handler.handle_cleanup(mock_event, dry_run=True)]
    assert any("解析对话历史失败" in msg for msg in messages)
    assert any("有效 JSON" in msg for msg in messages)


@pytest.mark.asyncio
async def test_handle_search_renders_dual_route_breakdown(
    handler, mock_event, memory_engine
):
    result = Mock(
        doc_id=8,
        final_score=0.91,
        content="graph memory",
        score_breakdown={
            "document_vector_score": 0.22,
            "graph_keyword_score": 0.33,
            "graph_vector_score": 0.44,
        },
    )
    memory_engine.search_memories = AsyncMock(return_value=[result])

    messages = [msg async for msg in handler.handle_search(mock_event, "graph", 5)]
    assert len(messages) == 1
    assert "0.22" in messages[0]
    assert "0.33" in messages[0]
    assert "0.44" in messages[0]


@pytest.mark.asyncio
async def test_handle_rebuild_graph_reports_progress_and_summary(
    handler, mock_event, memory_engine
):
    memory_engine.rebuild_graph_index = AsyncMock(
        return_value={"rebuilt": 3, "skipped": 1}
    )

    messages = [msg async for msg in handler.handle_rebuild_graph(mock_event)]
    assert len(messages) == 2
    memory_engine.rebuild_graph_index.assert_awaited_once()
    assert messages[0].endswith("...")
    assert [
        part for part in messages[1].split() if any(ch.isdigit() for ch in part)
    ] == ["3", "1"]
