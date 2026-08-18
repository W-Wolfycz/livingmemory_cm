"""MemoryEngine 拆分服务的独立回归测试。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiosqlite
import pytest

from livingmemory_cm.core.memory import (
    DocumentRepository,
    MemoryLifecycleContext,
    MemoryLifecycleService,
    MemoryRepairContext,
    MemoryRepairService,
    MemorySchemaContext,
    MemorySchemaService,
    MemorySearchService,
    MemoryStatisticsService,
    MemoryWriteContext,
    MemoryWriteCoordinator,
)
from livingmemory_cm.core.models.memory_atom import (
    AtomStatus,
    AtomType,
    DecayType,
    MemoryAtom,
)
from livingmemory_cm.core.retrieval.hybrid_retriever import HybridResult


class _FakeDocumentStorage:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs

    async def get_documents(
        self,
        metadata_filters,
        ids=None,
        limit=50,
        offset=0,
    ):
        docs = list(self.docs)
        if ids is not None:
            wanted = set(ids)
            docs = [doc for doc in docs if doc["id"] in wanted]
        for key, value in (metadata_filters or {}).items():
            docs = [
                doc
                for doc in docs
                if isinstance(doc.get("metadata"), dict)
                and doc["metadata"].get(key) == value
            ]
        return [dict(doc) for doc in docs[offset : offset + limit]]

    async def count_documents(self, metadata_filters):
        docs = await self.get_documents(metadata_filters, limit=10000)
        return len(docs)


def _faiss_with_docs(docs: list[dict]):
    return SimpleNamespace(document_storage=_FakeDocumentStorage(docs))


def _doc(doc_id: int, text: str, **metadata) -> dict:
    return {
        "id": doc_id,
        "doc_id": f"uuid-{doc_id}",
        "text": text,
        "metadata": metadata,
    }


def _repair_context(**overrides) -> MemoryRepairContext:
    values = {
        "db_connection": None,
        "faiss_db": SimpleNamespace(delete=AsyncMock()),
        "atom_store": None,
        "graph_memory_manager": None,
        "atom_enabled": True,
        "max_retries": 3,
        "get_memory": AsyncMock(return_value=None),
        "advance_write_op": AsyncMock(),
        "invalidate_search_cache": Mock(),
    }
    values.update(overrides)
    return MemoryRepairContext(**values)


def _write_context(**overrides) -> MemoryWriteContext:
    hybrid_retriever = SimpleNamespace(
        add_memory=AsyncMock(return_value=7),
        update_metadata=AsyncMock(return_value=True),
        delete_memory=AsyncMock(return_value=True),
    )
    values = {
        "db_connection": None,
        "faiss_db": SimpleNamespace(delete=AsyncMock()),
        "hybrid_retriever": hybrid_retriever,
        "atom_store": None,
        "graph_memory_manager": None,
        "atom_enabled": True,
        "get_memory": AsyncMock(return_value=None),
        "find_memory_by_idempotency_key": AsyncMock(return_value=None),
        "add_memory": AsyncMock(return_value=8),
        "delete_memory": AsyncMock(return_value=True),
        "start_write_op": AsyncMock(return_value=11),
        "advance_write_op": AsyncMock(),
        "serialize_atom_for_repair": Mock(return_value={}),
        "delete_graph_and_atoms_for_batch": AsyncMock(),
        "invalidate_search_cache": Mock(),
    }
    values.update(overrides)
    return MemoryWriteContext(**values)


def _lifecycle_context(docs: list[dict] | None = None, **overrides):
    faiss_db = _faiss_with_docs(docs or [])
    values = {
        "db_connection": None,
        "faiss_db": faiss_db,
        "graph_memory_manager": None,
        "document_repository": DocumentRepository(faiss_db),
        "config": {},
        "batch_delete_memories": AsyncMock(return_value=0),
        "invalidate_search_cache": Mock(),
    }
    values.update(overrides)
    return MemoryLifecycleContext(**values)


def test_document_repository_normalizes_metadata() -> None:
    docs = [
        {"metadata": '{"importance": 0.8}'},
        {"metadata": "{invalid"},
        {"metadata": None},
    ]

    normalized = DocumentRepository.normalize_batch_metadata(docs)

    assert normalized[0]["metadata"] == {"importance": 0.8}
    assert normalized[1]["metadata"] == {}
    assert normalized[2]["metadata"] == {}


@pytest.mark.asyncio
async def test_document_repository_reads_by_id_and_idempotency_key() -> None:
    repository = DocumentRepository(
        _faiss_with_docs(
            [
                _doc(
                    7,
                    "测试记忆",
                    idempotency_key="batch-demo:0",
                    session_id="demo:private:10001",
                )
            ]
        )
    )

    assert await repository.find_by_idempotency_key("batch-demo:0") == 7
    assert await repository.find_by_idempotency_key("") is None
    memory = await repository.get_memory(7)
    assert memory is not None
    assert memory["text"] == "测试记忆"
    assert await repository.get_memory(99) is None


@pytest.mark.asyncio
async def test_document_repository_sorts_session_memories() -> None:
    session_id = "demo:private:10001"
    repository = DocumentRepository(
        _faiss_with_docs(
            [
                _doc(1, "较早", session_id=session_id, create_time=10.0),
                _doc(2, "最新", session_id=session_id, create_time=30.0),
                _doc(3, "中间", session_id=session_id, create_time=20.0),
                _doc(4, "其他会话", session_id="demo:private:10002"),
            ]
        )
    )

    memories = await repository.get_session_memories(session_id, limit=2)

    assert [memory["id"] for memory in memories] == [2, 3]


@pytest.mark.asyncio
async def test_memory_search_service_caches_and_invalidates() -> None:
    result = HybridResult(
        doc_id=1,
        final_score=0.9,
        vector_score=0.8,
        content="缓存测试",
        metadata={},
    )
    retriever = SimpleNamespace(search=AsyncMock(return_value=[result]))
    service = MemorySearchService(
        {"search_cache_ttl_seconds": 60, "search_cache_max_size": 8}
    )
    access_ids: list[int] = []
    migrated_sessions: list[str] = []
    tasks: list[asyncio.Task] = []

    async def update_access(memory_id: int) -> bool:
        access_ids.append(memory_id)
        return True

    async def migrate_session(session_id: str) -> None:
        migrated_sessions.append(session_id)

    def schedule(coro) -> None:
        tasks.append(asyncio.create_task(coro))

    kwargs = {
        "k": 3,
        "session_id": "demo:private:10001",
        "persona_id": "persona_demo",
        "hybrid_retriever": retriever,
        "dual_route_retriever": None,
        "schedule_task": schedule,
        "update_access_time": update_access,
        "migrate_session": migrate_session,
    }
    first = await service.search(query="苹果", **kwargs)
    second = await service.search(query="  苹果  ", **kwargs)
    service.invalidate()
    third = await service.search(query="苹果", **kwargs)
    await asyncio.gather(*tasks)

    assert [item.doc_id for item in first] == [1]
    assert [item.doc_id for item in second] == [1]
    assert [item.doc_id for item in third] == [1]
    assert retriever.search.await_count == 2
    assert access_ids == [1, 1, 1]
    assert migrated_sessions == ["demo:private:10001", "demo:private:10001"]


@pytest.mark.asyncio
async def test_memory_search_service_empty_query_skips_dependencies() -> None:
    service = MemorySearchService({})

    result = await service.search(
        query="   ",
        k=5,
        session_id=None,
        persona_id=None,
        hybrid_retriever=None,
        dual_route_retriever=None,
        schedule_task=lambda coro: None,
        update_access_time=AsyncMock(),
        migrate_session=AsyncMock(),
    )

    assert result == []


def test_memory_search_service_applies_optional_retrieval_policies() -> None:
    service = MemorySearchService(
        {
            "min_importance_for_retrieval": 0.6,
            "min_similarity_for_retrieval": 0.5,
            "memory_type_filter": "event_only",
        }
    )
    results = [
        HybridResult(1, 0.9, 0.8, "保留", {"importance": 0.8}),
        HybridResult(
            2,
            0.9,
            0.8,
            "已归档",
            {"importance": 0.8, "status": "archived"},
        ),
        HybridResult(3, 0.9, 0.8, "低重要性", {"importance": 0.2}),
        HybridResult(4, 0.9, 0.2, "低相似度", {"importance": 0.8}),
        HybridResult(
            5,
            0.9,
            0.8,
            "纯偏好",
            {"importance": 0.8, "atom_types": ["preference"]},
        ),
    ]

    filtered = service._filter_by_retrieval_policy(results)

    assert [result.doc_id for result in filtered] == [1]


def test_memory_search_service_event_only_excludes_preference_keeps_events() -> None:
    service = MemorySearchService({"memory_type_filter": "event_only"})
    results = [
        HybridResult(
            1,
            0.9,
            0.8,
            "纯偏好",
            {"importance": 0.8, "atom_types": ["preference"]},
        ),
        HybridResult(
            2,
            0.9,
            0.8,
            "计划与事实",
            {"importance": 0.8, "atom_types": ["planned", "factual"]},
        ),
        HybridResult(
            3,
            0.9,
            0.8,
            "混合含事件",
            {"importance": 0.8, "atom_types": ["preference", "planned"]},
        ),
        HybridResult(
            4,
            0.9,
            0.8,
            "空类型列表兼容",
            {"importance": 0.8, "atom_types": []},
        ),
        HybridResult(5, 0.9, 0.8, "无类型元数据兼容", {"importance": 0.8}),
    ]

    filtered = service._filter_by_retrieval_policy(results)

    assert [result.doc_id for result in filtered] == [2, 3, 4, 5]


@pytest.mark.asyncio
async def test_memory_search_recent_slot_keeps_session_and_persona_scope() -> None:
    connection = await aiosqlite.connect(":memory:")
    await connection.execute(
        "CREATE TABLE documents(id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
    )
    now = 1_900_000_000.0
    rows = [
        (
            1,
            "当前范围近期记忆",
            json.dumps(
                {
                    "session_id": "demo:private:10001",
                    "persona_id": "persona_demo",
                    "create_time": now,
                    "status": "active",
                }
            ),
        ),
        (
            2,
            "其他人格近期记忆",
            json.dumps(
                {
                    "session_id": "demo:private:10001",
                    "persona_id": "persona_other",
                    "create_time": now + 10,
                    "status": "active",
                }
            ),
        ),
    ]
    await connection.executemany(
        "INSERT INTO documents(id, text, metadata) VALUES (?, ?, ?)",
        rows,
    )
    await connection.commit()

    vector_result = HybridResult(9, 0.9, 0.8, "相关向量记忆", {})
    retriever = SimpleNamespace(search=AsyncMock(return_value=[vector_result]))
    service = MemorySearchService(
        {
            "recent_memory_count": 1,
            "recent_memory_max_age_hours": 0,
        }
    )

    def close_task(coro) -> None:
        coro.close()

    results = await service.search(
        query="查询",
        k=2,
        session_id="demo:private:10001",
        persona_id="persona_demo",
        hybrid_retriever=retriever,
        dual_route_retriever=None,
        schedule_task=close_task,
        update_access_time=AsyncMock(),
        migrate_session=AsyncMock(),
        db_connection=connection,
    )

    assert [result.doc_id for result in results] == [9, 1]
    await connection.close()


@pytest.mark.asyncio
async def test_memory_statistics_service_filters_persona_and_aggregates() -> None:
    faiss_db = _faiss_with_docs(
        [
            _doc(
                1,
                "第一条",
                persona_id="persona_demo",
                session_id="demo:private:10001",
                importance=0.8,
                status="active",
                create_time=10.0,
            ),
            _doc(
                2,
                "第二条",
                persona_id="persona_demo",
                session_id="demo:private:10001",
                importance=0.4,
                status="archived",
                create_time=20.0,
            ),
            _doc(
                3,
                "其他人格",
                persona_id="persona_other",
                session_id="demo:private:10002",
                importance=0.9,
            ),
        ]
    )
    service = MemoryStatisticsService("unused.db", faiss_db)

    stats = await service.get_statistics("persona_demo")

    assert stats["total_memories"] == 2
    assert stats["sessions"] == {"demo:private:10001": 2}
    assert stats["status_breakdown"] == {
        "active": 1,
        "archived": 1,
        "deleted": 0,
    }
    assert stats["avg_importance"] == pytest.approx(0.6)
    assert stats["oldest_memory"] == 10.0
    assert stats["newest_memory"] == 20.0
    assert stats["graph_memory_enabled"] is False


@pytest.mark.asyncio
async def test_memory_statistics_service_maintains_sqlite(tmp_path) -> None:
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"")
    connection = SimpleNamespace(
        execute=AsyncMock(),
        commit=AsyncMock(),
    )
    service = MemoryStatisticsService(str(db_path), _faiss_with_docs([]))

    result = await service.maintain_storage(connection, vacuum=True)

    assert result["success"] is True
    assert result["vacuum"] is True
    executed = [call.args[0] for call in connection.execute.await_args_list]
    assert "PRAGMA wal_checkpoint(TRUNCATE)" in executed
    assert "VACUUM" in executed
    assert any("livingmemory_graph_entries_fts" in sql for sql in executed)
    assert any("memory_atoms_fts" in sql for sql in executed)


def test_memory_repair_service_round_trips_atom_payload() -> None:
    service = MemoryRepairService()
    atom = MemoryAtom(
        parent_memory_id=7,
        atom_type=AtomType.FACTUAL,
        content="用户偏好清淡口味",
        entities=["清淡口味"],
        importance=0.8,
        status=AtomStatus.DORMANT,
        decay_type=DecayType.LINEAR,
        session_id="demo:private:10001",
        persona_id="persona_demo",
        metadata={"source": "reflection"},
    )

    payload = service.serialize_atom(atom)
    restored = service.deserialize_atom(payload, 9, None, None)

    assert restored is not None
    assert restored.parent_memory_id == 9
    assert restored.atom_type is AtomType.FACTUAL
    assert restored.status is AtomStatus.DORMANT
    assert restored.decay_type is DecayType.LINEAR
    assert restored.session_id == "demo:private:10001"
    assert restored.persona_id == "persona_demo"
    assert restored.metadata == {"source": "reflection"}


@pytest.mark.asyncio
async def test_memory_repair_service_repairs_delete_and_invalidates_cache() -> None:
    cursor = SimpleNamespace(
        fetchall=AsyncMock(
            return_value=[
                {
                    "id": 11,
                    "op_type": "delete",
                    "memory_id": 7,
                    "status": "needs_repair",
                    "step": "document_deleted",
                    "payload": "{}",
                    "retry_count": 1,
                }
            ]
        )
    )
    db_connection = SimpleNamespace(execute=AsyncMock(return_value=cursor))
    graph_manager = SimpleNamespace(delete_memory=AsyncMock())
    atom_store = SimpleNamespace(delete_by_parent=AsyncMock())
    context = _repair_context(
        db_connection=db_connection,
        graph_memory_manager=graph_manager,
        atom_store=atom_store,
    )

    repaired = await MemoryRepairService().repair_incomplete_write_ops(context)

    assert repaired == 1
    graph_manager.delete_memory.assert_awaited_once_with(7)
    atom_store.delete_by_parent.assert_awaited_once_with(7)
    context.advance_write_op.assert_awaited_once_with(
        11,
        "completed",
        status="completed",
        memory_id=7,
    )
    context.invalidate_search_cache.assert_called_once_with()


@pytest.mark.asyncio
async def test_memory_repair_service_batch_delete_filters_invalid_ids() -> None:
    graph_manager = SimpleNamespace(batch_delete_memories=AsyncMock())
    atom_store = SimpleNamespace(batch_delete_by_parent=AsyncMock())
    context = _repair_context(
        graph_memory_manager=graph_manager,
        atom_store=atom_store,
    )

    repaired = await MemoryRepairService().repair_batch_delete_write_op(
        context,
        12,
        {"memory_ids": [1, "bad", "2", None]},
    )

    assert repaired is True
    graph_manager.batch_delete_memories.assert_awaited_once_with([1, 2])
    atom_store.batch_delete_by_parent.assert_awaited_once_with([1, 2])
    context.advance_write_op.assert_awaited_once_with(
        12,
        "completed",
        status="completed",
        payload_patch={"deleted_count": 2},
    )


@pytest.mark.asyncio
async def test_memory_write_coordinator_idempotency_skips_new_write() -> None:
    context = _write_context(
        find_memory_by_idempotency_key=AsyncMock(return_value=7)
    )

    memory_id = await MemoryWriteCoordinator().add_memory(
        context,
        "已存在的记忆",
        idempotency_key="batch-demo:0",
    )

    assert memory_id == 7
    context.start_write_op.assert_not_awaited()
    context.hybrid_retriever.add_memory.assert_not_awaited()
    context.invalidate_search_cache.assert_not_called()


@pytest.mark.asyncio
async def test_memory_write_coordinator_add_completes_write_log() -> None:
    context = _write_context()

    memory_id = await MemoryWriteCoordinator().add_memory(
        context,
        "新增长期记忆",
        session_id="demo:private:10001",
        persona_id="persona_demo",
        importance=0.8,
    )

    assert memory_id == 7
    content, metadata = context.hybrid_retriever.add_memory.await_args.args
    assert content == "新增长期记忆"
    assert metadata["session_id"] == "demo:private:10001"
    assert metadata["persona_id"] == "persona_demo"
    assert metadata["importance"] == pytest.approx(0.8)
    assert context.advance_write_op.await_args_list[-1].args == (11, "completed")
    assert context.advance_write_op.await_args_list[-1].kwargs == {
        "status": "completed",
        "memory_id": 7,
    }
    context.invalidate_search_cache.assert_called_once_with()


@pytest.mark.asyncio
async def test_memory_write_coordinator_marks_secondary_delete_for_repair() -> None:
    graph_manager = SimpleNamespace(
        delete_memory=AsyncMock(side_effect=RuntimeError("graph unavailable"))
    )
    atom_store = SimpleNamespace(delete_by_parent=AsyncMock())
    context = _write_context(
        graph_memory_manager=graph_manager,
        atom_store=atom_store,
    )

    deleted = await MemoryWriteCoordinator().delete_memory(context, 7)

    assert deleted is True
    atom_store.delete_by_parent.assert_awaited_once_with(7)
    failed_calls = [
        call
        for call in context.advance_write_op.await_args_list
        if call.args[1] == "graph_delete_failed"
    ]
    assert len(failed_calls) == 1
    assert failed_calls[0].kwargs["status"] == "needs_repair"
    assert all(
        call.args[1] != "completed"
        for call in context.advance_write_op.await_args_list
    )


@pytest.mark.asyncio
async def test_memory_lifecycle_service_rebuilds_graph_and_skips_empty_docs() -> None:
    graph_manager = SimpleNamespace(index_memory=AsyncMock())
    context = _lifecycle_context(
        [
            _doc(1, "有效记忆", importance=0.8),
            _doc(2, "   ", importance=0.5),
        ],
        graph_memory_manager=graph_manager,
    )

    result = await MemoryLifecycleService().rebuild_graph_index(context)

    assert result == {"rebuilt": 1, "skipped": 1}
    graph_manager.index_memory.assert_awaited_once_with(
        1,
        "有效记忆",
        {"importance": 0.8},
    )
    context.invalidate_search_cache.assert_called_once_with()


@pytest.mark.asyncio
async def test_memory_lifecycle_service_protects_high_importance_from_decay() -> None:
    connection = await aiosqlite.connect(":memory:")
    connection.row_factory = aiosqlite.Row
    await connection.execute(
        "CREATE TABLE documents(id INTEGER PRIMARY KEY, metadata TEXT)"
    )
    await connection.executemany(
        "INSERT INTO documents(id, metadata) VALUES (?, ?)",
        [
            (1, json.dumps({"importance": 1.0})),
            (2, json.dumps({"importance": 0.8})),
        ],
    )
    await connection.commit()
    context = _lifecycle_context(
        db_connection=connection,
        config={"protected_importance_threshold": 1.0},
    )

    affected = await MemoryLifecycleService().apply_daily_decay(
        context,
        decay_rate=0.1,
    )

    rows = await (
        await connection.execute("SELECT id, metadata FROM documents ORDER BY id")
    ).fetchall()
    metadata_by_id = {row["id"]: json.loads(row["metadata"]) for row in rows}
    assert affected == 1
    assert metadata_by_id[1]["importance"] == 1.0
    assert metadata_by_id[2]["importance"] < 0.8
    await connection.close()


@pytest.mark.asyncio
async def test_memory_lifecycle_service_cleanup_delegates_candidate_ids() -> None:
    batch_delete = AsyncMock(return_value=1)
    context = _lifecycle_context(
        [
            _doc(1, "低重要性旧记忆", importance=0.1, create_time=1.0),
            _doc(2, "高重要性旧记忆", importance=0.9, create_time=1.0),
        ],
        batch_delete_memories=batch_delete,
    )

    deleted = await MemoryLifecycleService().cleanup_old_memories(
        context,
        days_threshold=1,
        importance_threshold=0.3,
    )

    assert deleted == 1
    batch_delete.assert_awaited_once_with([1])


@pytest.mark.asyncio
async def test_memory_lifecycle_service_recovers_invalid_access_count() -> None:
    read_cursor = SimpleNamespace(
        fetchone=AsyncMock(return_value=('{"access_count": "bad"}',))
    )
    write_cursor = SimpleNamespace()
    connection = SimpleNamespace(
        execute=AsyncMock(side_effect=[read_cursor, write_cursor]),
        commit=AsyncMock(),
    )
    context = _lifecycle_context(db_connection=connection)

    updated = await MemoryLifecycleService().update_access_time(context, 7)

    assert updated is True
    update_args = connection.execute.await_args_list[1].args
    stored_metadata = json.loads(update_args[1][0])
    assert stored_metadata["access_count"] == 1
    assert stored_metadata["last_access_time"] > 0
    connection.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_memory_schema_service_upgrades_legacy_documents_table(
    tmp_path,
) -> None:
    db_path = tmp_path / "legacy_schema.db"
    async with aiosqlite.connect(db_path) as connection:
        await connection.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        await connection.execute(
            "INSERT INTO documents(text, metadata) VALUES (?, ?)",
            ("旧记忆", '{"session_id": "demo:private:10001"}'),
        )

        async def create_write_ops_table() -> None:
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_write_ops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                )
            """)

        create_write_ops = AsyncMock(side_effect=create_write_ops_table)
        context = MemorySchemaContext(connection, create_write_ops)

        await MemorySchemaService().create_tables(context)

        cursor = await connection.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert {"doc_id", "created_at", "updated_at"}.issubset(columns)

        cursor = await connection.execute(
            "SELECT doc_id, created_at, updated_at FROM documents WHERE id = 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "legacy-1"
        assert row[1]
        assert row[2]

        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
        index_names = {row[0] for row in await cursor.fetchall()}
        assert "idx_doc_idempotency_key" in index_names
        assert "idx_documents_doc_id" in index_names

        cursor = await connection.execute("SELECT COUNT(*) FROM db_version")
        assert (await cursor.fetchone())[0] == 1
        create_write_ops.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_memory_schema_service_drops_only_legacy_fts_triggers(
    tmp_path,
) -> None:
    db_path = tmp_path / "legacy_triggers.db"
    async with aiosqlite.connect(db_path) as connection:
        await connection.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT)"
        )
        await connection.execute(
            "CREATE TABLE documents_fts (rowid INTEGER, content TEXT)"
        )
        await connection.execute(
            "CREATE TABLE audit_log (document_id INTEGER)"
        )
        await connection.execute("""
            CREATE TRIGGER "legacy""trigger" AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(rowid, content)
                VALUES (new.id, new.text);
            END
        """)
        await connection.execute("""
            CREATE TRIGGER keep_trigger AFTER UPDATE ON documents BEGIN
                INSERT INTO audit_log(document_id) VALUES (new.id);
            END
        """)
        context = MemorySchemaContext(connection, AsyncMock())

        dropped = await MemorySchemaService().drop_legacy_documents_fts_triggers(
            context
        )

        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
        trigger_names = {row[0] for row in await cursor.fetchall()}
        assert dropped == 1
        assert 'legacy"trigger' not in trigger_names
        assert "keep_trigger" in trigger_names
