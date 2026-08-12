"""ConversationManager 单元测试（CM-only 单路径版）。

ConversationManager 在 CM-only fork 中仅维护会话元数据
（如 last_summarized_timestamp），不再做消息写入/读取/缓存。

测试覆盖：
- create_or_get_session：存在则返回，不存在则创建
- get_session_info：返回 Session 或 None
- get_recent_sessions：透传 store
- clear_session：删除消息 + 重置元数据
- update_session_metadata：session 不存在时 warning + no-op；存在则合并写入
- get_session_metadata：session 不存在返回 default；存在返回值或 default
- reset_session_metadata：清空 metadata dict 并持久化

测试通过 fake store（不依赖 aiosqlite），隔离 SQL 层。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livingmemory_cm.core.managers.conversation_manager import ConversationManager
from livingmemory_cm.core.models.conversation_models import (
    Message,
    Session,
    deserialize_from_json,
    serialize_to_json,
)


class FakeStore:
    """内存版 ConversationStore，仅实现 ConversationManager 依赖的方法。"""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.deleted_messages: list[str] = []

    async def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def create_session(self, session_id: str, platform: str) -> Session:
        session = Session(
            id=len(self.sessions) + 1,
            session_id=session_id,
            platform=platform,
            created_at=0.0,
            last_active_at=0.0,
            message_count=0,
            participants=[],
            metadata={},
        )
        self.sessions[session_id] = session
        return session

    async def update_session_activity(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].last_active_at = 999.0

    async def get_recent_sessions(self, limit: int = 10) -> list[Session]:
        return list(self.sessions.values())[:limit]

    async def delete_session_messages(self, session_id: str) -> int:
        self.deleted_messages.append(session_id)
        return 0


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def manager(store: FakeStore) -> ConversationManager:
    return ConversationManager(store=store)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_create_or_get_session_creates_new(store: FakeStore, manager: ConversationManager) -> None:
    session = _run(manager.create_or_get_session("s1", platform="aiocqhttp"))
    assert session.session_id == "s1"
    assert session.platform == "aiocqhttp"
    assert "s1" in store.sessions


def test_create_or_get_session_returns_existing_and_updates_activity(
    store: FakeStore, manager: ConversationManager
) -> None:
    _run(manager.create_or_get_session("s1", platform="aiocqhttp"))
    original_active = store.sessions["s1"].last_active_at

    result = _run(manager.create_or_get_session("s1", platform="aiocqhttp"))
    assert result.session_id == "s1"
    assert store.sessions["s1"].last_active_at != original_active


def test_get_session_info_returns_none_when_missing(manager: ConversationManager) -> None:
    result = _run(manager.get_session_info("nonexistent"))
    assert result is None


def test_get_session_info_returns_session_when_exists(
    store: FakeStore, manager: ConversationManager
) -> None:
    _run(manager.create_or_get_session("s1", platform="aiocqhttp"))
    result = _run(manager.get_session_info("s1"))
    assert result is not None
    assert result.session_id == "s1"


def test_get_recent_sessions_passes_limit(store: FakeStore, manager: ConversationManager) -> None:
    for i in range(3):
        _run(manager.create_or_get_session(f"s{i}", platform="p"))
    result = _run(manager.get_recent_sessions(limit=2))
    assert len(result) == 2


def test_clear_session_deletes_messages_and_resets_metadata(
    store: FakeStore, manager: ConversationManager
) -> None:
    _run(manager.create_or_get_session("s1", platform="p"))
    _run(manager.update_session_metadata("s1", "last_summarized_timestamp", "2026-01-01 00:00:00"))
    assert store.sessions["s1"].metadata != {}

    _run(manager.clear_session("s1"))

    assert "s1" in store.deleted_messages
    assert store.sessions["s1"].metadata == {}


def test_update_session_metadata_no_op_when_session_missing(
    store: FakeStore, manager: ConversationManager
) -> None:
    _run(manager.update_session_metadata("nonexistent", "key", "value"))
    assert store.sessions == {}


def test_update_session_metadata_merges_existing_keys(
    store: FakeStore, manager: ConversationManager
) -> None:
    _run(manager.create_or_get_session("s1", platform="p"))
    _run(manager.update_session_metadata("s1", "last_summarized_timestamp", "T1"))
    _run(manager.update_session_metadata("s1", "other_key", "V2"))

    md = store.sessions["s1"].metadata
    assert md["last_summarized_timestamp"] == "T1"
    assert md["other_key"] == "V2"


def test_get_session_metadata_returns_default_when_session_missing(
    manager: ConversationManager
) -> None:
    result = _run(manager.get_session_metadata("nonexistent", "key", "fallback"))
    assert result == "fallback"


def test_get_session_metadata_returns_default_when_key_missing(
    store: FakeStore, manager: ConversationManager
) -> None:
    _run(manager.create_or_get_session("s1", platform="p"))
    result = _run(manager.get_session_metadata("s1", "missing", "default"))
    assert result == "default"


def test_get_session_metadata_returns_value_when_present(
    store: FakeStore, manager: ConversationManager
) -> None:
    _run(manager.create_or_get_session("s1", platform="p"))
    _run(manager.update_session_metadata("s1", "last_summarized_timestamp", "T1"))
    result = _run(manager.get_session_metadata("s1", "last_summarized_timestamp"))
    assert result == "T1"


def test_reset_session_metadata_clears_dict(
    store: FakeStore, manager: ConversationManager
) -> None:
    _run(manager.create_or_get_session("s1", platform="p"))
    _run(manager.update_session_metadata("s1", "k1", "v1"))
    _run(manager.update_session_metadata("s1", "k2", "v2"))

    _run(manager.reset_session_metadata("s1"))

    assert store.sessions["s1"].metadata == {}


def test_reset_session_metadata_no_op_when_missing(
    store: FakeStore, manager: ConversationManager
) -> None:
    _run(manager.reset_session_metadata("nonexistent"))
    assert store.sessions == {}


def test_message_roundtrip_and_llm_format() -> None:
    message = Message(
        id=1,
        session_id="s1",
        role="assistant",
        content="hello",
        sender_id="bot",
        sender_name="Bot",
        group_id="g1",
        platform="test",
        metadata={"is_bot_message": True},
    )

    restored = Message.from_dict(message.to_dict())
    formatted = restored.format_for_llm(include_sender_name=True)

    assert restored.content == "hello"
    assert formatted["role"] == "assistant"
    assert "[Bot:" in formatted["content"]


def test_message_multimodal_content_is_normalized_for_llm() -> None:
    message = Message(
        id=1,
        session_id="s1",
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            {"type": "text", "text": "图片里的日程是下午三点"},
        ],
        sender_id="u1",
        sender_name="Alice",
        group_id="g1",
        platform="test",
        metadata={},
    )

    formatted = message.format_for_llm(include_sender_name=True)["content"]

    assert "图片里的日程是下午三点" in formatted
    assert "image_url" not in formatted
    assert "example.test" not in formatted


def test_message_image_only_content_uses_placeholder() -> None:
    message = Message(
        id=1,
        session_id="s1",
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
        ],
        sender_id="u1",
        sender_name="Alice",
        group_id=None,
        platform="test",
        metadata={},
    )

    assert message.format_for_llm(include_sender_name=True)["content"] == "[图片消息]"
    assert message.to_dict()["content"] == "[图片消息]"


def test_conversation_json_helpers() -> None:
    raw = serialize_to_json({"a": 1})

    assert isinstance(raw, str)
    assert deserialize_from_json(raw)["a"] == 1
    assert deserialize_from_json(None, default={}) == {}
