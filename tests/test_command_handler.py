"""
Tests for CommandHandler.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from livingmemory_cm.core.base.config_manager import ConfigManager
from livingmemory_cm.core.command_handler import CommandHandler
from livingmemory_cm.core.i18n_backend import init as init_i18n
from livingmemory_cm.storage.freeze_store import FreezeStore


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
async def test_handle_search_validates_inputs_and_calls_engine(handler, mock_event):
    empty = [msg async for msg in handler.handle_search(mock_event, "", 3)]
    assert "不能为空" in empty[0]

    _ = [msg async for msg in handler.handle_search(mock_event, "hello", 200)]
    # k should be clamped to 100.
    handler.memory_engine.search_memories.assert_awaited_with(
        query="hello", k=100, session_id=mock_event.unified_msg_origin
    )


@pytest.mark.asyncio
async def test_handle_forget_success_and_not_found(handler, mock_event, memory_engine):
    success = [msg async for msg in handler.handle_forget(mock_event, 10)]
    assert "已删除记忆 #10" in success[0]

    memory_engine.delete_memory = AsyncMock(return_value=False)
    failed = [msg async for msg in handler.handle_forget(mock_event, 11)]
    assert "删除失败" in failed[0]


# ── /lmem freeze 与 /lmem unfreeze ──────────────────────────────────────────


@pytest.fixture
def freeze_handler(tmp_path, config_manager):
    store = FreezeStore(tmp_path / "frozen_personas.json")
    engine = Mock()
    engine.freeze_store = store
    engine.refresh_persona_access_time = AsyncMock(return_value=3)
    return CommandHandler(
        context=Mock(),
        config_manager=config_manager,
        memory_engine=engine,
        conversation_manager=None,
    )


@pytest.mark.asyncio
async def test_handle_freeze_lists_and_freezes_persona(
    freeze_handler, mock_event, tmp_path
) -> None:
    empty = [msg async for msg in freeze_handler.handle_freeze(mock_event, "list")]
    assert len(empty) == 1
    assert "没有冻结的 persona" in empty[0]

    frozen = [msg async for msg in freeze_handler.handle_freeze(mock_event, "persona_demo")]
    assert len(frozen) == 1
    assert "已冻结" in frozen[0]

    store = freeze_handler.memory_engine.freeze_store
    assert await store.frozen_personas() == {"persona_demo"}

    listed = [msg async for msg in freeze_handler.handle_freeze(mock_event, "")]
    assert len(listed) == 1
    assert "已冻结" in listed[0]
    assert "persona_demo" in listed[0]


@pytest.mark.asyncio
async def test_handle_unfreeze_never_frozen_persona_writes_nothing(
    freeze_handler, mock_event, tmp_path
) -> None:
    """从未冻结的 persona 不能凭空获得豁免记录，也不能被刷新访问时间。"""
    messages = [
        msg async for msg in freeze_handler.handle_unfreeze(mock_event, "persona_never")
    ]

    assert len(messages) == 1
    assert "不在冻结列表中" in messages[0]
    store = freeze_handler.memory_engine.freeze_store
    assert await store.protected_personas() == set()
    assert await store.entries() == []
    assert not (tmp_path / "frozen_personas.json").exists()
    freeze_handler.memory_engine.refresh_persona_access_time.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_unfreeze_frozen_persona_enters_grace(
    freeze_handler, mock_event
) -> None:
    store = freeze_handler.memory_engine.freeze_store
    await store.freeze("persona_demo")

    messages = [msg async for msg in freeze_handler.handle_unfreeze(mock_event, "persona_demo")]

    assert len(messages) == 1
    assert "已解冻" in messages[0]
    assert "3 条记忆的访问时间" in messages[0]
    # 默认 freeze_grace_days=7：解冻后仍在保护名单内，但不再算"冻结"
    assert await store.frozen_personas() == set()
    assert await store.protected_personas() == {"persona_demo"}
    freeze_handler.memory_engine.refresh_persona_access_time.assert_awaited_once_with(
        "persona_demo"
    )


