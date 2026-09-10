"""冻结（冷存储）逻辑测试：状态持久化、召回隔离、清理保护。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from livingmemory_cm.core.memory.memory_lifecycle_service import (
    MemoryLifecycleContext,
    MemoryLifecycleService,
)
from livingmemory_cm.core.memory.memory_search_service import MemorySearchService
from livingmemory_cm.storage.freeze_store import FreezeStateUnreadable, FreezeStore


def _freeze_store(tmp_path: Path) -> FreezeStore:
    return FreezeStore(tmp_path / "frozen_personas.json")


@pytest.mark.asyncio
async def test_freeze_and_unfreeze_lifecycle(tmp_path: Path) -> None:
    """冻结 → 保护名单；解冻 → 宽限期内仍受保护，之后自动清理记录。"""
    store = _freeze_store(tmp_path)

    await store.freeze("persona_demo")
    assert await store.frozen_personas() == {"persona_demo"}
    assert await store.protected_personas() == {"persona_demo"}

    # 重复冻结幂等
    await store.freeze("persona_demo")
    assert await store.frozen_personas() == {"persona_demo"}

    await store.unfreeze("persona_demo", grace_days=7)
    assert await store.frozen_personas() == set()
    assert await store.protected_personas() == {"persona_demo"}

    entries = await store.entries()
    assert entries[0]["state"] == "grace"

    # grace_days=0 → 直接移除状态记录
    await store.unfreeze("persona_demo", grace_days=0)
    assert await store.protected_personas() == set()
    assert await store.entries() == []


@pytest.mark.asyncio
async def test_expired_grace_is_reported_and_cleaned(tmp_path: Path) -> None:
    """宽限期已过的记录标记为 expired，并可被清理出状态文件。"""
    path = tmp_path / "frozen_personas.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "personas": {
                    "persona_demo": {
                        "frozen_at": "2026-01-01T00:00:00+00:00",
                        "grace_until": "2026-01-02T00:00:00+00:00",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = FreezeStore(path)

    entries = await store.entries()
    assert entries[0]["state"] == "expired"
    assert await store.protected_personas() == set()
    assert await store.grace_expired_personas() == ["persona_demo"]
    assert await store.entries() == []


@pytest.mark.asyncio
async def test_unfreeze_without_grace_removes_record(tmp_path: Path) -> None:
    store = _freeze_store(tmp_path)
    await store.freeze("persona_demo")

    await store.unfreeze("persona_demo", grace_days=0)

    assert await store.entries() == []
    assert await store.protected_personas() == set()


@pytest.mark.asyncio
async def test_freeze_store_fails_closed_when_state_file_corrupt(tmp_path: Path) -> None:
    """状态文件损坏/为空/不可解析时抛错，绝不按"未冻结"继续。"""
    path = tmp_path / "frozen_personas.json"
    path.write_text("{ not json", encoding="utf-8")
    store = FreezeStore(path)

    with pytest.raises(FreezeStateUnreadable):
        await store.frozen_personas()
    with pytest.raises(FreezeStateUnreadable):
        await store.protected_personas()
    with pytest.raises(FreezeStateUnreadable):
        await store.entries()

    # 写路径同样拒绝：否则会把空视图覆盖落盘，永久丢光已有冻结记录
    with pytest.raises(FreezeStateUnreadable):
        await store.freeze("persona_demo")
    assert path.read_text(encoding="utf-8") == "{ not json"

    # 空文件与结构非法同样按不可读处理
    path.write_text("", encoding="utf-8")
    with pytest.raises(FreezeStateUnreadable):
        await store.load()
    path.write_text('[1, 2]', encoding="utf-8")
    with pytest.raises(FreezeStateUnreadable):
        await store.load()
    path.write_text('{"schema": 1}', encoding="utf-8")
    with pytest.raises(FreezeStateUnreadable):
        await store.load()
    path.write_text('{"schema": 1, "personas": {"p": 1}}', encoding="utf-8")
    with pytest.raises(FreezeStateUnreadable):
        await store.load()

    # 文件根本不存在（从未冻结）是正常情况，返回空表
    store_missing = FreezeStore(tmp_path / "nope.json")
    assert await store_missing.frozen_personas() == set()
    assert await store_missing.entries() == []


@pytest.mark.asyncio
async def test_unparsable_grace_until_keeps_protection(tmp_path: Path) -> None:
    """grace_until 是坏值时按"仍冻结/仍受保护"处理，且不会被自动清出状态文件。"""
    path = tmp_path / "frozen_personas.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "personas": {
                    "persona_demo": {
                        "frozen_at": "2026-01-01T00:00:00+00:00",
                        "grace_until": "not-a-timestamp",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = FreezeStore(path)

    assert await store.frozen_personas() == {"persona_demo"}
    assert await store.protected_personas() == {"persona_demo"}
    assert (await store.entries())[0]["state"] == "grace"
    assert await store.grace_expired_personas() == []
    assert await store.protected_personas() == {"persona_demo"}


def test_search_service_excludes_frozen_personas() -> None:
    """冻结 persona 的结果在召回阶段被剔除。"""
    from livingmemory_cm.core.memory.memory_search_service import HybridResult

    service = MemorySearchService({})
    results = [
        HybridResult(1, 0.9, 0.8, "冻结人格记忆", {"persona_id": "persona_frozen"}),
        HybridResult(2, 0.9, 0.8, "正常人格记忆", {"persona_id": "persona_live"}),
        HybridResult(3, 0.9, 0.8, "无人格记忆", {"importance": 0.8}),
    ]

    filtered = service._filter_frozen_personas(results, {"persona_frozen"})
    assert [item.doc_id for item in filtered] == [2, 3]

    # 空集合不过滤
    assert len(service._filter_frozen_personas(results, set())) == 3


def _normalize_metadata(batch: list[dict]) -> list[dict]:
    """模拟 DocumentRepository：把 JSON 文本 metadata 解析为 dict。"""
    normalized = []
    for doc in batch:
        metadata = doc.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        normalized.append({**doc, "metadata": metadata if isinstance(metadata, dict) else {}})
    return normalized


def _make_cleanup_context(db_path: Path, store: FreezeStore) -> MemoryLifecycleContext:
    documents = [
        (1, "frozen-old", {"persona_id": "persona_frozen", "importance": 0.05,
                           "create_time": time.time() - 90 * 86400}),
        (2, "grace-old", {"persona_id": "persona_grace", "importance": 0.05,
                          "create_time": time.time() - 90 * 86400}),
        (3, "normal-old", {"persona_id": "persona_live", "importance": 0.05,
                           "create_time": time.time() - 90 * 86400}),
    ]
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
    )
    connection.executemany(
        "INSERT INTO documents(id, text, metadata) VALUES (?, ?, ?)",
        [(doc_id, text, json.dumps(meta)) for doc_id, text, meta in documents],
    )
    connection.commit()

    async def _count_documents(metadata_filters=None) -> int:
        return len(documents)

    async def _get_documents(metadata_filters=None, limit=None, offset=0):
        rows = connection.execute(
            "SELECT id, text, metadata FROM documents"
        ).fetchall()
        return [
            {"id": row[0], "text": row[1], "metadata": row[2]} for row in rows
        ]

    return MemoryLifecycleContext(
        db_connection=None,
        faiss_db=SimpleNamespace(
            document_storage=SimpleNamespace(
                count_documents=_count_documents,
                get_documents=_get_documents,
            )
        ),
        graph_memory_manager=None,
        document_repository=SimpleNamespace(
            normalize_batch_metadata=_normalize_metadata
        ),
        config={"cleanup_days_threshold": 30, "cleanup_importance_threshold": 0.3},
        batch_delete_memories=lambda ids: _record_deletion(ids),
        invalidate_search_cache=lambda: None,
        freeze_store=store,
    )


@pytest.mark.asyncio
async def test_cleanup_skips_frozen_and_grace_personas(tmp_path: Path) -> None:
    """自动清理跳过冻结中与解冻宽限期内的 persona，但正常 persona 照删。"""
    store = _freeze_store(tmp_path)
    await store.freeze("persona_frozen")
    await store.freeze("persona_grace")
    await store.unfreeze("persona_grace", grace_days=7)

    deleted: list[list[int]] = []

    async def _record_deletion(ids: list[int]) -> int:
        deleted.append(list(ids))
        return len(ids)

    context = _make_cleanup_context(tmp_path / "documents.db", store)
    context.batch_delete_memories = _record_deletion

    count = await MemoryLifecycleService().cleanup_old_memories(context)

    assert deleted == [[3]]
    assert count == 1


@pytest.mark.asyncio
async def test_recent_slot_does_not_leak_frozen_persona() -> None:
    """近期槽位直接查库，必须同样过滤冻结 persona（回归防护）。"""
    from livingmemory_cm.core.memory.memory_search_service import HybridResult

    session = "demo:GroupMessage:group_demo"
    now = time.time()
    connection = await aiosqlite.connect(":memory:")
    connection.row_factory = aiosqlite.Row
    await connection.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
    )
    await connection.executemany(
        "INSERT INTO documents(id, text, metadata) VALUES (?, ?, ?)",
        [
            (1, "frozen-recent", json.dumps({"persona_id": "persona_frozen",
                                             "session_id": session, "importance": 0.6,
                                             "create_time": now - 60,
                                             "last_access_time": now - 60})),
            (2, "live-recent", json.dumps({"persona_id": "persona_live",
                                           "session_id": session, "importance": 0.6,
                                           "create_time": now - 120,
                                           "last_access_time": now - 120})),
        ],
    )
    await connection.commit()

    class _EmptyHybrid:
        async def search(self, query, k, session_id=None, persona_id=None):
            return []

    async def _noop(*_args, **_kwargs):
        return None

    service = MemorySearchService(
        {"recent_memory_count": 2, "recent_memory_max_age_hours": 72}
    )
    results = await service.search(
        query="测试查询",
        k=5,
        session_id=session,
        persona_id=None,
        exclude_personas={"persona_frozen"},
        hybrid_retriever=_EmptyHybrid(),
        dual_route_retriever=None,
        schedule_task=lambda coro: coro.close(),
        update_access_time=_noop,
        migrate_session=_noop,
        db_connection=connection,
    )

    assert [item.doc_id for item in results] == [2]
    await connection.close()


@pytest.mark.asyncio
async def test_cleanup_skips_round_when_state_file_unreadable(tmp_path: Path) -> None:
    """状态文件损坏时整轮清理跳过：宁可不清理，也不误删冻结/豁免期记忆。"""
    path = tmp_path / "frozen_personas.json"
    path.write_text("{ broken", encoding="utf-8")
    store = FreezeStore(path)

    deleted: list[list[int]] = []

    async def _record_deletion(ids: list[int]) -> int:
        deleted.append(list(ids))
        return len(ids)

    context = _make_cleanup_context(tmp_path / "documents.db", store)
    context.batch_delete_memories = _record_deletion

    count = await MemoryLifecycleService().cleanup_old_memories(context)

    assert count == 0
    assert deleted == []


@pytest.mark.asyncio
async def test_cleanup_rechecks_protection_before_delete(tmp_path: Path) -> None:
    """扫描期间被冻结的 persona，同一轮不能被删（快照式保护会漏掉它）。"""
    store = _freeze_store(tmp_path)
    documents = [
        (1, "old-low", {"persona_id": "persona_demo", "importance": 0.05,
                        "create_time": time.time() - 90 * 86400}),
    ]
    connection = sqlite3.connect(tmp_path / "documents.db")
    connection.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
    )
    connection.executemany(
        "INSERT INTO documents(id, text, metadata) VALUES (?, ?, ?)",
        [(doc_id, text, json.dumps(meta)) for doc_id, text, meta in documents],
    )
    connection.commit()

    scanned = False

    async def _count_documents(metadata_filters=None) -> int:
        return len(documents)

    async def _get_documents(metadata_filters=None, limit=None, offset=0):
        nonlocal scanned
        if not scanned:
            scanned = True
            # 模拟"扫描过程中管理员冻结了该 persona"
            await store.freeze("persona_demo")
        rows = connection.execute(
            "SELECT id, text, metadata FROM documents"
        ).fetchall()
        return [{"id": row[0], "text": row[1], "metadata": row[2]} for row in rows]

    deleted: list[list[int]] = []

    async def _record_deletion(ids: list[int]) -> int:
        deleted.append(list(ids))
        return len(ids)

    context = MemoryLifecycleContext(
        db_connection=None,
        faiss_db=SimpleNamespace(
            document_storage=SimpleNamespace(
                count_documents=_count_documents,
                get_documents=_get_documents,
            )
        ),
        graph_memory_manager=None,
        document_repository=SimpleNamespace(
            normalize_batch_metadata=_normalize_metadata
        ),
        config={"cleanup_days_threshold": 30, "cleanup_importance_threshold": 0.3},
        batch_delete_memories=_record_deletion,
        invalidate_search_cache=lambda: None,
        freeze_store=store,
    )

    count = await MemoryLifecycleService().cleanup_old_memories(context)

    assert count == 0
    assert deleted == []
    assert await store.frozen_personas() == {"persona_demo"}


@pytest.mark.asyncio
async def test_refresh_persona_access_time_handles_escaped_metadata(
    tmp_path: Path,
) -> None:
    """AstrBot 用 json.dumps(metadata) 落库（default ensure_ascii=True）→
    中文 persona_id 在库里是 \\uXXXX 转义，解冻唤醒必须仍能命中。"""
    connection = await aiosqlite.connect(tmp_path / "documents.db")
    connection.row_factory = aiosqlite.Row
    await connection.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
    )
    now = time.time()
    persona = "人格助手"
    other = "其他助手"
    await connection.executemany(
        "INSERT INTO documents(id, text, metadata) VALUES (?, ?, ?)",
        [
            # 刻意用 json.dumps 默认参数，复刻 AstrBot document_storage 的写法
            (1, "目标人格记忆", json.dumps(
                {"persona_id": persona, "importance": 0.2,
                 "create_time": now - 400 * 86400, "last_access_time": 1.0})),
            (2, "其他人格记忆", json.dumps(
                {"persona_id": other, "importance": 0.2,
                 "create_time": now - 400 * 86400, "last_access_time": 1.0})),
        ],
    )
    await connection.commit()

    invalidated: list[bool] = []
    context = MemoryLifecycleContext(
        db_connection=connection,
        faiss_db=None,
        graph_memory_manager=None,
        document_repository=None,
        config={},
        batch_delete_memories=None,
        invalidate_search_cache=lambda: invalidated.append(True),
    )

    refreshed = await MemoryLifecycleService().refresh_persona_access_time(
        context, persona
    )

    assert refreshed == 1
    assert invalidated == [True]
    cursor = await connection.execute(
        "SELECT id, metadata FROM documents ORDER BY id"
    )
    rows = await cursor.fetchall()
    target = json.loads(rows[0]["metadata"])
    untouched = json.loads(rows[1]["metadata"])
    assert target["last_access_time"] > 1.0
    assert target["create_time"] == now - 400 * 86400
    assert target["importance"] == 0.2
    assert untouched["last_access_time"] == 1.0
    await connection.close()
