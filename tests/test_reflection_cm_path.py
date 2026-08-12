"""CM-only 反思路径回归测试。"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace

import pytest

from livingmemory_cm.core.event_handler_modules.memory_reflection import (
    MemoryReflection,
)
from livingmemory_cm.core.event_handler_modules.memory_recall import MemoryRecall
from livingmemory_cm.core.event_handler_modules import (
    memory_reflection as reflection_hook_module,
)
from livingmemory_cm.log import log_ref
from livingmemory_cm.core.processors.memory_processor import MemoryProcessor
from livingmemory_cm.core.reflection import (
    CMHistoryReader,
    ReflectionBatchWriter,
    ReflectionCursor,
    ReflectionCursorService,
    ReflectionExtractionService,
    ReflectionService,
)
from livingmemory_cm.core.reflection import reflection_service as reflection_module


def _record(turn: int, role: str, second: int, user_id: str = "10001") -> dict:
    return {
        "role": role,
        "content": f"message-{turn}-{role}",
        "user_id": user_id,
        "self_id": "10000",
        "sender_nickname": "Alice" if role == "user" else None,
        "group_id": "group_demo",
        "created_at_utc": f"2026-07-19T00:00:{second:02d}Z",
        "record_id": turn * 10 + (1 if role == "assistant" else 0),
        "turn_id": f"turn-{turn}",
        "content_kind": ["text"],
    }


def _round(turn: int, second: int) -> list[dict]:
    return [_record(turn, "user", second), _record(turn, "assistant", second + 1)]


def test_log_ref_is_stable_without_exposing_identifier() -> None:
    first = log_ref("10001", "user")
    second = log_ref("10001", "user")

    assert first == second
    assert first.startswith("user:")
    assert "10001" not in first


@pytest.mark.asyncio
async def test_query_rounds_fetches_oldest_filtered_batch() -> None:
    reader = CMHistoryReader()
    first_page = [_round(1, 10), _round(2, 20)]
    second_page = [_round(3, 30), _round(4, 40)]
    cm = SimpleNamespace(
        ct_include_kinds={"text"},
        ct_include_all_match=True,
        query_rounds=AsyncMock(side_effect=[first_page, second_page]),
    )

    rounds = await reader.query_rounds_paginated(
        cm_plugin=cm,
        umo="demo:GroupMessage:group_demo",
        conversation_id="conversation_demo",
        user_id=None,
        persona_id="persona_demo",
        llm_status=["llm_success"],
        cursor=ReflectionCursor("2026-07-18T00:00:00.000000Z", 0),
        page_size=2,
        target_rounds=4,
    )

    assert [rnd[0]["turn_id"] for rnd in rounds] == [
        "turn-1",
        "turn-2",
        "turn-3",
        "turn-4",
    ]
    assert cm.query_rounds.await_count == 2
    assert all(
        call.kwargs["from_oldest"] is True
        for call in cm.query_rounds.await_args_list
    )
    assert cm.query_rounds.await_args_list[0].kwargs["content_kind"] == ["text"]
    assert cm.query_rounds.await_args_list[0].kwargs["content_kind_all_match"] is True
    assert cm.query_rounds.await_args_list[0].kwargs["after_id"] == 0


@pytest.mark.asyncio
async def test_query_messages_fetches_oldest_filtered_batch() -> None:
    reader = CMHistoryReader()
    first_page = [_record(1, "user", 10), _record(2, "assistant", 20)]
    second_page = [_record(3, "user", 30), _record(4, "assistant", 40)]
    cm = SimpleNamespace(
        ct_include_kinds={"text"},
        ct_include_all_match=True,
        query_history=AsyncMock(side_effect=[first_page, second_page]),
    )

    messages = await reader.query_messages_paginated(
        cm_plugin=cm,
        umo="demo:GroupMessage:group_demo",
        conversation_id="conversation_demo",
        user_id=None,
        persona_id="persona_demo",
        llm_status=["llm_success", "proactive"],
        cursor=ReflectionCursor("2026-07-18T00:00:00.000000Z", 0),
        page_size=2,
        target_messages=4,
    )

    assert [message["created_at_utc"] for message in messages] == [
        "2026-07-19T00:00:10Z",
        "2026-07-19T00:00:20Z",
        "2026-07-19T00:00:30Z",
        "2026-07-19T00:00:40Z",
    ]
    assert cm.query_history.await_count == 2
    assert all(
        call.kwargs["from_oldest"] is True
        for call in cm.query_history.await_args_list
    )
    assert cm.query_history.await_args_list[0].kwargs["content_kind"] == ["text"]
    assert cm.query_history.await_args_list[0].kwargs["content_kind_all_match"] is True
    assert cm.query_history.await_args_list[0].kwargs["after_id"] == 0


@pytest.mark.asyncio
async def test_complete_batch_at_safety_limit_does_not_raise() -> None:
    reader = CMHistoryReader()
    page = [_record(i, "user", i) for i in range(1, 5)]
    cm = SimpleNamespace(
        ct_include_kinds=set(),
        ct_include_all_match=False,
        query_history=AsyncMock(return_value=page),
    )

    messages = await reader.query_messages_paginated(
        cm_plugin=cm,
        umo="demo:GroupMessage:group_demo",
        conversation_id="conversation_demo",
        user_id=None,
        persona_id="persona_demo",
        llm_status=["llm_success", "proactive"],
        cursor=ReflectionCursor("2026-07-18T00:00:00.000000Z", 0),
        page_size=4,
        target_messages=4,
        max_messages=4,
    )

    assert len(messages) == 4


@pytest.mark.asyncio
async def test_same_timestamp_pagination_uses_record_id_without_skipping() -> None:
    reader = CMHistoryReader()
    first_page = [_record(1, "user", 10), _record(2, "user", 10)]
    second_page = [_record(3, "user", 10), _record(4, "user", 10)]
    for index, record in enumerate(first_page + second_page, 101):
        record["record_id"] = index
    cm = SimpleNamespace(
        ct_include_kinds={"text"},
        ct_include_all_match=False,
        query_history=AsyncMock(side_effect=[first_page, second_page]),
    )

    messages = await reader.query_messages_paginated(
        cm_plugin=cm,
        umo="demo:GroupMessage:group_demo",
        conversation_id="conversation_demo",
        user_id=None,
        persona_id="persona_demo",
        llm_status=["llm_success"],
        cursor=ReflectionCursor("2026-07-19T00:00:09.000000Z", 0),
        page_size=2,
        target_messages=4,
    )

    assert [message["record_id"] for message in messages] == [101, 102, 103, 104]
    second_call = cm.query_history.await_args_list[1].kwargs
    assert second_call["since"].isoformat() == "2026-07-19T00:00:10"
    assert second_call["after_id"] == 102


@pytest.mark.asyncio
async def test_new_partition_baseline_queries_only_latest_record() -> None:
    reader = CMHistoryReader()
    cm = AsyncMock()
    cm.query_history = AsyncMock(return_value=[_record(9, "assistant", 50)])

    cursor = await reader.query_latest_cursor(
        cm_plugin=cm,
        extraction_mode="messages",
        umo="demo:GroupMessage:group_demo",
        conversation_id="conversation_demo",
        user_id=None,
        persona_id="persona_demo",
        llm_status=["llm_success", "proactive"],
    )

    assert cursor == ReflectionCursor("2026-07-19T00:00:50.000000Z", 91)
    assert cm.query_history.await_args.kwargs["limit"] == 1
    assert "from_oldest" not in cm.query_history.await_args.kwargs


@pytest.mark.asyncio
async def test_new_partition_bypasses_backlog_pagination(monkeypatch) -> None:
    cm = SimpleNamespace(
        ct_full_group=False,
        ct_llm_status_filter=["llm_success", "proactive"],
        ct_include_kinds={"text"},
        ct_include_all_match=False,
        _is_group_umo=lambda _umo: True,
    )
    context = SimpleNamespace(
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation_demo")
        )
    )
    conversation_manager = SimpleNamespace(
        create_or_get_session=AsyncMock(),
        get_session_metadata=AsyncMock(return_value={}),
        update_session_metadata=AsyncMock(),
    )
    history_reader = SimpleNamespace(
        query_latest_cursor=AsyncMock(
            return_value=ReflectionCursor("2026-07-19T00:00:50.000000Z", 91)
        ),
        query_messages_paginated=AsyncMock(),
    )
    service = ReflectionService(
        context=context,
        conversation_manager=conversation_manager,
        cursor_service=ReflectionCursorService(conversation_manager),
        history_reader=history_reader,
        batch_writer=SimpleNamespace(write=AsyncMock()),
        storage_tasks=set(),
        storage_sessions_inflight=set(),
        storage_state_lock=asyncio.Lock(),
    )

    async def _persona_id(_context, _event) -> str:
        return "persona_demo"

    monkeypatch.setattr(reflection_module, "get_cm_plugin", lambda _context: cm)
    monkeypatch.setattr(reflection_module, "get_persona_id", _persona_id)

    event = SimpleNamespace(get_sender_id=lambda: "10001")
    await service.dispatch(
        event=event,
        session_id="demo:GroupMessage:group_demo",
        trigger_count=120,
        cm_limit=120,
        extraction_mode="messages",
    )

    history_reader.query_latest_cursor.assert_awaited_once()
    history_reader.query_messages_paginated.assert_not_awaited()
    assert conversation_manager.update_session_metadata.await_count == 2


def test_extraction_mode_follows_cm_status_semantics() -> None:
    assert (
        ReflectionService.get_extraction_mode(
            SimpleNamespace(ct_llm_status_filter={"llm_success"})
        )
        == "rounds"
    )
    assert (
        ReflectionService.get_extraction_mode(
            SimpleNamespace(ct_llm_status_filter={"llm_success", "proactive"})
        )
        == "messages"
    )
    assert (
        ReflectionService.get_extraction_mode(
            SimpleNamespace(ct_llm_status_filter=set())
        )
        == "messages"
    )


@pytest.mark.asyncio
async def test_memory_reflection_hook_delegates_to_service(monkeypatch) -> None:
    reflection = MemoryReflection.__new__(MemoryReflection)
    reflection.context = SimpleNamespace()
    reflection.config_manager = SimpleNamespace(
        get=lambda _key, _default=0: 5
    )
    reflection._reflection_service = SimpleNamespace(dispatch=AsyncMock())
    cm = SimpleNamespace(ct_llm_status_filter={"llm_success"})
    monkeypatch.setattr(
        reflection_hook_module,
        "get_cm_status",
        lambda _context: (True, 120),
    )
    monkeypatch.setattr(
        reflection_hook_module,
        "get_cm_plugin",
        lambda _context: cm,
    )
    event = SimpleNamespace(unified_msg_origin="demo:FriendMessage:10001")

    await reflection.handle_memory_reflection(event)

    reflection._reflection_service.dispatch.assert_awaited_once_with(
        event=event,
        session_id="demo:FriendMessage:10001",
        trigger_count=5,
        cm_limit=120,
        extraction_mode="rounds",
    )


def test_content_kind_filter_matches_cm_any_and_all_semantics() -> None:
    record = {"content_kind": ["text", "image"]}
    any_cm = SimpleNamespace(
        ct_include_kinds={"text"}, ct_include_all_match=False
    )
    all_cm = SimpleNamespace(
        ct_include_kinds={"text"}, ct_include_all_match=True
    )

    assert CMHistoryReader.record_matches_content_filter(record, any_cm)
    assert not CMHistoryReader.record_matches_content_filter(record, all_cm)


@pytest.mark.asyncio
async def test_batch_writer_writes_all_generated_topic_memories() -> None:
    conversation_manager = SimpleNamespace(
        get_session_metadata=AsyncMock(return_value={}),
        update_session_metadata=AsyncMock(),
    )
    memory_processor = SimpleNamespace(
        process_conversation_batch=AsyncMock(
            return_value=[
                (
                    "事实：项目会议",
                    {"topics": ["会议"], "key_facts": ["Alice 安排项目会议"]},
                    0.8,
                ),
                (
                    "事实：咖啡偏好",
                    {"topics": ["偏好"], "key_facts": ["Alice 喜欢黑咖啡"]},
                    0.6,
                ),
            ]
        ),
        classify_atoms_from_metadata=lambda **_kwargs: [],
    )
    memory_engine = SimpleNamespace(add_memory=AsyncMock())
    writer = ReflectionBatchWriter(
        memory_engine,
        conversation_manager,
        ReflectionCursorService(conversation_manager),
        ReflectionExtractionService(memory_processor),
    )

    await writer.write(
        session_id="demo:GroupMessage:group_demo",
        cm_messages=_round(1, 10),
        persona_id="persona_demo",
        start_cursor=ReflectionCursor("2026-07-19T00:00:00.000000Z", 0),
        end_cursor=ReflectionCursor("2026-07-19T00:00:11.000000Z", 11),
        cursor_key="cursor_demo",
        current_user_id="10001",
    )

    assert memory_engine.add_memory.await_count == 2
    first_metadata = memory_engine.add_memory.await_args_list[0].kwargs[
        "metadata"
    ]
    second_metadata = memory_engine.add_memory.await_args_list[1].kwargs[
        "metadata"
    ]
    assert first_metadata["source_window"]["batch_index"] == 1
    assert second_metadata["source_window"]["batch_index"] == 2
    assert second_metadata["source_window"]["batch_size"] == 2
    assert second_metadata["source_window"]["end_record_id"] == 11
    first_key = memory_engine.add_memory.await_args_list[0].kwargs[
        "idempotency_key"
    ]
    second_key = memory_engine.add_memory.await_args_list[1].kwargs[
        "idempotency_key"
    ]
    assert first_key.startswith("reflection:")
    assert first_key.endswith(":1")
    assert second_key.endswith(":2")
    assert conversation_manager.update_session_metadata.await_count == 2


@pytest.mark.asyncio
async def test_empty_extraction_advances_cursor_without_writing_memory() -> None:
    conversation_manager = SimpleNamespace(
        get_session_metadata=AsyncMock(return_value={}),
        update_session_metadata=AsyncMock(),
    )
    memory_processor = SimpleNamespace(
        process_conversation_batch=AsyncMock(return_value=[]),
        classify_atoms_from_metadata=lambda **_kwargs: [],
    )
    memory_engine = SimpleNamespace(add_memory=AsyncMock())
    writer = ReflectionBatchWriter(
        memory_engine,
        conversation_manager,
        ReflectionCursorService(conversation_manager),
        ReflectionExtractionService(memory_processor),
    )

    await writer.write(
        session_id="demo:FriendMessage:10001",
        cm_messages=_round(1, 10),
        persona_id="persona_demo",
        start_cursor=ReflectionCursor("2026-07-19T00:00:00.000000Z", 0),
        end_cursor=ReflectionCursor("2026-07-19T00:00:11.000000Z", 11),
        cursor_key="cursor_demo",
        current_user_id="10001",
    )

    memory_engine.add_memory.assert_not_awaited()
    assert conversation_manager.update_session_metadata.await_count == 2


@pytest.mark.asyncio
async def test_failed_extraction_does_not_advance_cursor() -> None:
    conversation_manager = SimpleNamespace(
        get_session_metadata=AsyncMock(return_value={}),
        update_session_metadata=AsyncMock(),
    )
    memory_processor = SimpleNamespace(
        process_conversation_batch=AsyncMock(side_effect=RuntimeError("provider")),
        classify_atoms_from_metadata=lambda **_kwargs: [],
    )
    memory_engine = SimpleNamespace(add_memory=AsyncMock())
    writer = ReflectionBatchWriter(
        memory_engine,
        conversation_manager,
        ReflectionCursorService(conversation_manager),
        ReflectionExtractionService(memory_processor),
    )

    await writer.write(
        session_id="demo:FriendMessage:10001",
        cm_messages=_round(1, 10),
        persona_id="persona_demo",
        start_cursor=ReflectionCursor("2026-07-19T00:00:00.000000Z", 0),
        end_cursor=ReflectionCursor("2026-07-19T00:00:11.000000Z", 11),
        cursor_key="cursor_demo",
        current_user_id="10001",
    )

    memory_engine.add_memory.assert_not_awaited()
    conversation_manager.update_session_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_reflection_service_delegates_to_batch_writer() -> None:
    writer = SimpleNamespace(write=AsyncMock())
    service = ReflectionService(
        context=SimpleNamespace(),
        conversation_manager=SimpleNamespace(),
        cursor_service=SimpleNamespace(),
        history_reader=SimpleNamespace(),
        batch_writer=writer,
        storage_tasks=set(),
        storage_sessions_inflight=set(),
        storage_state_lock=asyncio.Lock(),
    )
    start_cursor = ReflectionCursor("2026-07-19T00:00:00.000000Z", 0)
    end_cursor = ReflectionCursor("2026-07-19T00:00:11.000000Z", 11)
    messages = _round(1, 10)

    await service._write_batch(
        session_id="demo:GroupMessage:group_demo",
        cm_messages=messages,
        persona_id="persona_demo",
        start_cursor=start_cursor,
        end_cursor=end_cursor,
        cursor_key="cursor_demo",
        current_user_id="10001",
    )

    writer.write.assert_awaited_once_with(
        session_id="demo:GroupMessage:group_demo",
        cm_messages=messages,
        persona_id="persona_demo",
        start_cursor=start_cursor,
        end_cursor=end_cursor,
        cursor_key="cursor_demo",
        current_user_id="10001",
    )


def test_cursor_is_partitioned_by_conversation_persona_and_scope() -> None:
    base = ReflectionCursorService.build_partition_key(
        "cid-a", "persona-a", "user:10001"
    )
    assert base != ReflectionCursorService.build_partition_key(
        "cid-b", "persona-a", "user:10001"
    )
    assert base != ReflectionCursorService.build_partition_key(
        "cid-a", "persona-b", "user:10001"
    )
    assert base != ReflectionCursorService.build_partition_key(
        "cid-a", "persona-a", "full_group"
    )


def test_composite_cursor_orders_same_timestamp_by_record_id() -> None:
    first = _record(1, "user", 10)
    second = _record(2, "user", 10)
    first["record_id"] = 100
    second["record_id"] = 101

    cursor = ReflectionCursorService.latest_from_records([first, second])

    assert cursor == ReflectionCursor("2026-07-19T00:00:10.000000Z", 101)
    assert ReflectionCursorService.sort_key(cursor) > ReflectionCursorService.sort_key(
        ReflectionCursor("2026-07-19T00:00:10.000000Z", 100)
    )


@pytest.mark.asyncio
async def test_legacy_v2_cursor_loads_as_exclusive_composite_cursor() -> None:
    conversation_manager = SimpleNamespace(
        get_session_metadata=AsyncMock(
            side_effect=[
                {},
                {"cursor_demo": "2026-07-19T00:00:10Z"},
            ]
        )
    )

    cursor, migrated = await ReflectionCursorService(
        conversation_manager
    ).load(
        "demo:FriendMessage:10001", "cursor_demo"
    )

    assert migrated is True
    assert cursor == ReflectionCursor(
        "2026-07-19T00:00:10.000000Z", 2**63 - 1
    )


@pytest.mark.asyncio
async def test_legacy_cursor_resolves_exact_record_id_from_cm() -> None:
    reader = CMHistoryReader()
    record = _record(5, "assistant", 10)
    record["record_id"] = 77
    cm = SimpleNamespace(query_history=AsyncMock(return_value=[record]))

    resolved = await reader.resolve_legacy_cursor_record_id(
        cm_plugin=cm,
        umo="demo:FriendMessage:10001",
        conversation_id="conversation_demo",
        user_id="10001",
        persona_id="persona_demo",
        cursor=ReflectionCursor(
            "2026-07-19T00:00:10.000000Z", 2**63 - 1
        ),
    )

    assert resolved == ReflectionCursor("2026-07-19T00:00:10.000000Z", 77)
    assert cm.query_history.await_args.kwargs["since"] == cm.query_history.await_args.kwargs[
        "until"
    ]


def test_cm_message_conversion_preserves_real_speaker_identity() -> None:
    user = ReflectionExtractionService.convert_cm_dict_to_message(
        _record(1, "user", 10), "demo:GroupMessage:group_demo", "10001"
    )
    bot = ReflectionExtractionService.convert_cm_dict_to_message(
        _record(1, "assistant", 11), "demo:GroupMessage:group_demo", "10001"
    )

    assert user.sender_id == "10001"
    assert user.sender_name == "Alice"
    assert user.metadata["speaker_relation"] == "current_user"
    assert bot.sender_id == "10000"
    assert bot.sender_name == "Bot"
    assert bot.metadata["speaker_relation"] == "bot"

    processor = MemoryProcessor.__new__(MemoryProcessor)
    formatted = processor._format_conversation([user, bot])
    assert "[当前发言者: Alice | ID: 10001" in formatted
    assert "[Bot: Bot | ID: 10000" in formatted


def _make_recall(
    rounds: int = 2,
    max_chars: int = 800,
    max_age_seconds: int = 0,
) -> MemoryRecall:
    recall = MemoryRecall.__new__(MemoryRecall)
    values = {
        "recall_engine.query_context_rounds": rounds,
        "recall_engine.query_context_max_chars": max_chars,
        "recall_engine.query_context_max_age_seconds": max_age_seconds,
    }
    recall.config_manager = Mock()
    recall.config_manager.get.side_effect = lambda key, default=None: values.get(
        key, default
    )
    return recall


def _recall_event() -> SimpleNamespace:
    return SimpleNamespace(
        unified_msg_origin="demo:GroupMessage:group_demo",
        get_sender_id=lambda: "10001",
    )


@pytest.mark.asyncio
async def test_recall_query_rounds_zero_uses_only_current_message() -> None:
    recall = _make_recall(rounds=0)
    recall.context = Mock()

    query = await recall._build_recall_query(
        event=_recall_event(),
        actual_query="继续说那个计划",
        persona_id="persona_demo",
    )

    assert query == "继续说那个计划"
    recall.context.get_registered_star.assert_not_called()


@pytest.mark.asyncio
async def test_recall_query_only_reads_current_user_paired_rounds() -> None:
    recall = _make_recall(rounds=2)
    cm = SimpleNamespace(
        ct_full_group=True,
        ct_cross_session=True,
        query_rounds=AsyncMock(
            return_value=[
                [
                    {"role": "user", "content": "张三讨论项目 A"},
                    {"role": "assistant", "content": "Bot 回复项目 A"},
                ],
                [
                    {"role": "user", "content": "张三又问了截止日期"},
                    {"role": "assistant", "content": "Bot 回答周五截止"},
                ],
            ]
        ),
    )
    recall.context = SimpleNamespace(
        get_registered_star=lambda _name: cm,
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation_demo")
        ),
    )

    query = await recall._build_recall_query(
        event=_recall_event(),
        actual_query="那个什么时候截止？",
        persona_id="persona_demo",
    )

    assert query.startswith("当前用户发言：那个什么时候截止？")
    assert query.endswith("需要检索的当前发言：那个什么时候截止？")
    assert "张三讨论项目 A" in query
    assert "Bot 回答周五截止" in query
    call = cm.query_rounds.await_args.kwargs
    assert call["umo"] == "demo:GroupMessage:group_demo"
    assert call["conversation_id"] == "conversation_demo"
    assert call["user_id"] == "10001"
    assert call["llm_status"] == "llm_success"
    assert call["limit_rounds"] == 2
    assert call["since"] is None


@pytest.mark.asyncio
async def test_recall_query_pushes_age_window_into_chat_memory() -> None:
    recall = _make_recall(max_age_seconds=7200)
    cm = SimpleNamespace(query_rounds=AsyncMock(return_value=[]))
    recall.context = SimpleNamespace(
        get_registered_star=lambda _name: cm,
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation_demo")
        ),
    )

    await recall._build_recall_query(
        event=_recall_event(),
        actual_query="继续",
        persona_id="persona_demo",
    )

    since = cm.query_rounds.await_args.kwargs["since"]
    assert since.tzinfo is not None
    age_seconds = (datetime.now(timezone.utc) - since).total_seconds()
    assert 7190 <= age_seconds <= 7210


@pytest.mark.asyncio
async def test_recall_query_respects_history_character_budget() -> None:
    recall = _make_recall(rounds=3, max_chars=90)
    cm = SimpleNamespace(
        query_rounds=AsyncMock(
            return_value=[
                [
                    {"role": "user", "content": "较早主题" * 20},
                    {"role": "assistant", "content": "较早回复" * 20},
                ],
                [
                    {"role": "user", "content": "最近主题"},
                    {"role": "assistant", "content": "最近回复"},
                ],
            ]
        )
    )
    recall.context = SimpleNamespace(
        get_registered_star=lambda _name: cm,
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation_demo")
        ),
    )

    query = await recall._build_recall_query(
        event=_recall_event(),
        actual_query="继续",
        persona_id="persona_demo",
    )

    history = query.split("最近相关问答（仅用于指代消歧）：\n", 1)[1].split(
        "\n需要检索的当前发言：", 1
    )[0]
    assert len(history) <= 90
    assert "最近主题" in history
    assert "最近回复" in history
