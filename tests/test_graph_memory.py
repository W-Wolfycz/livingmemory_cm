"""Tests for graph-memory indexing and dual-route retrieval."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from livingmemory_cm.core.managers.graph_memory_manager import (
    GraphMemoryManager,
)
from livingmemory_cm.core.managers.memory_engine import MemoryEngine
from livingmemory_cm.core.models.graph_models import (
    GraphEdge,
    GraphEntry,
    GraphNode,
)
from livingmemory_cm.core.processors.graph_extractor import GraphExtractor
from livingmemory_cm.core.processors.text_processor import TextProcessor
from livingmemory_cm.core.retrieval.graph_keyword_retriever import (
    GraphKeywordRetriever,
)
from livingmemory_cm.core.retrieval.graph_retriever import GraphRetriever
from livingmemory_cm.core.retrieval.graph_vector_retriever import (
    GraphVectorRetriever,
)
from livingmemory_cm.core.retrieval.rrf_fusion import RRFFusion
from livingmemory_cm.storage.graph_store import GraphStore


@dataclass
class _FakeRetrieveResult:
    similarity: float
    data: dict


class _FakeDocumentStorage:
    def __init__(self, db: "_FakeFaissDB"):
        self._db = db

    async def get_documents(self, metadata_filters, ids=None, limit=50, offset=0):
        docs = list(self._db.docs.values())
        if ids is not None:
            id_set = set(ids)
            docs = [doc for doc in docs if doc["id"] in id_set]

        for key, value in (metadata_filters or {}).items():
            docs = [doc for doc in docs if doc["metadata"].get(key) == value]

        docs = docs[offset : offset + limit]
        return [dict(doc) for doc in docs]

    async def count_documents(self, metadata_filters):
        docs = list(self._db.docs.values())
        for key, value in (metadata_filters or {}).items():
            docs = [doc for doc in docs if doc["metadata"].get(key) == value]
        return len(docs)


class _FakeFaissDB:
    def __init__(self):
        self.docs: dict[int, dict] = {}
        self._next_id = 1
        self.document_storage = _FakeDocumentStorage(self)

    async def insert(self, content: str, metadata: dict) -> int:
        doc_id = self._next_id
        self._next_id += 1
        self.docs[doc_id] = {
            "id": doc_id,
            "doc_id": f"uuid-{doc_id}",
            "text": content,
            "metadata": dict(metadata),
        }
        return doc_id

    async def retrieve(
        self, query: str, k: int, fetch_k: int, rerank: bool, metadata_filters=None
    ):
        results: list[_FakeRetrieveResult] = []
        for doc in self.docs.values():
            if metadata_filters:
                matched = True
                for key, value in metadata_filters.items():
                    if doc["metadata"].get(key) != value:
                        matched = False
                        break
                if not matched:
                    continue

            text = doc["text"]
            score = 0.9 if query in text else 0.2
            results.append(
                _FakeRetrieveResult(
                    similarity=score,
                    data={
                        "id": doc["id"],
                        "text": text,
                        "metadata": dict(doc["metadata"]),
                    },
                )
            )

        results.sort(key=lambda item: item.similarity, reverse=True)
        return results[:k]

    async def delete(self, uuid_doc_id: str) -> None:
        target = None
        for doc_id, doc in self.docs.items():
            if doc["doc_id"] == uuid_doc_id:
                target = doc_id
                break
        if target is not None:
            self.docs.pop(target, None)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_graph_memory_manager_indexes_nodes_edges_and_entries(tmp_path: Path):
    db_path = tmp_path / "graph_memory.db"
    graph_store = GraphStore(str(db_path))
    await graph_store.initialize()

    vector_db = _FakeFaissDB()
    graph_manager = GraphMemoryManager(
        graph_store=graph_store,
        graph_vector_retriever=GraphVectorRetriever(vector_db),
        graph_extractor=GraphExtractor(),
    )

    metadata = {
        "session_id": "test:private:s1",
        "persona_id": "persona_1",
        "importance": 0.8,
        "create_time": 1.0,
        "last_access_time": 1.0,
        "canonical_summary": "项目会议安排在明天下午三点",
        "topics": ["项目会议"],
        "participants": ["张三", "李四"],
        "key_facts": ["明天下午三点开会"],
    }

    await graph_manager.index_memory(1, metadata["canonical_summary"], metadata)

    stats = await graph_store.get_memory_entry_stats()
    assert stats["graph_nodes"] >= 4
    assert stats["graph_edges"] >= 3
    assert stats["graph_entries"] >= 4
    assert len(vector_db.docs) == 1
    vector_metadata = next(iter(vector_db.docs.values()))["metadata"]
    assert vector_metadata["graph_vector_granularity"] == "memory"
    assert vector_metadata["graph_entry_count"] == stats["graph_entries"]


@pytest.mark.asyncio
async def test_graph_memory_manager_rejects_entry_id_mismatch(tmp_path: Path):
    db_path = tmp_path / "graph_memory_mismatch.db"
    graph_store = GraphStore(str(db_path))
    await graph_store.initialize()

    class MismatchedGraphStore:
        async def upsert_nodes(self, nodes):
            return await graph_store.upsert_nodes(nodes)

        async def add_edges(self, edges, node_key_to_id):
            return await graph_store.add_edges(edges, node_key_to_id)

        async def add_entries(self, entries, node_key_to_id, edge_key_to_id):
            entry_ids = await graph_store.add_entries(
                entries,
                node_key_to_id,
                edge_key_to_id,
            )
            return entry_ids[:-1]

        async def update_entry_vector_doc_ids(self, entry_vector_doc_ids):
            return await graph_store.update_entry_vector_doc_ids(entry_vector_doc_ids)

        async def delete_memory(self, source_memory_id):
            return await graph_store.delete_memory(source_memory_id)

    graph_manager = GraphMemoryManager(
        graph_store=MismatchedGraphStore(),
        graph_vector_retriever=GraphVectorRetriever(_FakeFaissDB()),
        graph_extractor=GraphExtractor(),
    )

    with pytest.raises(RuntimeError, match="graph entry id count mismatch"):
        await graph_manager.index_memory(
            1,
            "项目会议安排在明天下午三点",
            {
                "topics": ["项目会议"],
                "participants": ["张三"],
                "key_facts": ["明天下午三点开会"],
            },
        )


@pytest.mark.asyncio
async def test_graph_store_snapshot_builds_ui_ready_subgraphs(tmp_path: Path):
    db_path = tmp_path / "graph_snapshot.db"
    graph_store = GraphStore(str(db_path))
    await graph_store.initialize()

    manager = GraphMemoryManager(
        graph_store=graph_store,
        graph_vector_retriever=GraphVectorRetriever(_FakeFaissDB()),
        graph_extractor=GraphExtractor(),
    )

    await manager.index_memory(
        11,
        "Roadmap workshop with Alice",
        {
            "session_id": "test:private:s1",
            "persona_id": "persona_1",
            "importance": 0.82,
            "create_time": 10.0,
            "last_access_time": 12.0,
            "canonical_summary": "Roadmap workshop with Alice",
            "topics": ["roadmap"],
            "participants": ["Alice"],
            "key_facts": ["Finalize Q2 roadmap"],
        },
    )
    await manager.index_memory(
        12,
        "Deployment sync with Bob",
        {
            "session_id": "test:private:s1",
            "persona_id": "persona_1",
            "importance": 0.74,
            "create_time": 11.0,
            "last_access_time": 13.0,
            "canonical_summary": "Deployment sync with Bob",
            "topics": ["deployment"],
            "participants": ["Bob"],
            "key_facts": ["Prepare release checklist"],
        },
    )

    snapshot = await graph_store.get_graph_snapshot(session_id="test:private:s1")
    assert snapshot["nodes"]
    assert snapshot["edges"]
    assert snapshot["entries"]
    assert {memory["memory_id"] for memory in snapshot["memories"]} >= {11, 12}

    focused = await graph_store.get_subgraph_for_memories([11])
    assert focused["memories"]
    assert focused["memories"][0]["memory_id"] == 11
    assert all(entry["memory_id"] == 11 for entry in focused["entries"])
    assert any(node["type"] == "person" for node in focused["nodes"])


@pytest.mark.asyncio
async def test_graph_retriever_supports_keyword_and_vector_search(tmp_path: Path):
    db_path = tmp_path / "graph_search.db"
    graph_store = GraphStore(str(db_path))
    await graph_store.initialize()
    vector_db = _FakeFaissDB()

    manager = GraphMemoryManager(
        graph_store=graph_store,
        graph_vector_retriever=GraphVectorRetriever(vector_db),
        graph_extractor=GraphExtractor(),
    )
    metadata = {
        "session_id": "test:private:s1",
        "persona_id": "persona_1",
        "importance": 0.7,
        "create_time": 1.0,
        "last_access_time": 1.0,
        "canonical_summary": "项目讨论记录",
        "topics": ["项目讨论"],
        "participants": ["张三"],
        "key_facts": ["明天下午三点开会"],
    }
    await manager.index_memory(5, metadata["canonical_summary"], metadata)

    text_processor = TextProcessor()
    retriever = GraphRetriever(
        keyword_retriever=GraphKeywordRetriever(graph_store, text_processor),
        vector_retriever=GraphVectorRetriever(vector_db),
        rrf_fusion=RRFFusion(60),
        config={},
    )

    results = await retriever.search("张三", k=3, session_id="test:private:s1")
    assert results
    assert results[0].doc_id == 5
    assert (results[0].keyword_score or 0) > 0

    results = await retriever.search(
        "明天下午三点开会", k=3, session_id="test:private:s1"
    )
    assert results
    assert results[0].doc_id == 5
    assert (results[0].vector_score or 0) > 0


@pytest.mark.asyncio
async def test_graph_vector_search_deduplicates_shadow_generations() -> None:
    vector_db = _FakeFaissDB()
    await vector_db.insert(
        "旧代无关内容",
        {"source_memory_id": 1, "graph_vector_granularity": "memory"},
    )
    await vector_db.insert(
        "新版发布计划",
        {"source_memory_id": 1, "graph_vector_granularity": "memory"},
    )
    await vector_db.insert(
        "新版测试计划",
        {"source_memory_id": 2, "graph_vector_granularity": "memory"},
    )

    results = await GraphVectorRetriever(vector_db).search("新版", k=2)

    assert len(results) == 2
    assert {result.doc_id for result in results} == {1, 2}
    assert next(result for result in results if result.doc_id == 1).score == 0.9


def test_graph_extractor_uses_stable_sender_identity_across_nicknames() -> None:
    extractor = GraphExtractor()

    def person_key(display_name: str) -> str:
        graph = extractor.extract(
            1,
            "身份测试",
            {
                "key_facts": ["同一用户修改了昵称"],
                "participant_identities": [
                    {
                        "identity_key": "qq:10001",
                        "sender_id": "10001",
                        "platform": "qq",
                        "display_name": display_name,
                        "aliases": [display_name],
                    }
                ],
            },
        )
        return next(node.node_key for node in graph.nodes if node.node_type == "person")

    assert person_key("旧昵称") == person_key("新昵称")
    assert person_key("新昵称") == "person:account:qq:10001"


@pytest.mark.asyncio
async def test_graph_keyword_retriever_supports_configurable_second_hop(
    tmp_path: Path,
):
    db_path = tmp_path / "graph_second_hop.db"
    graph_store = GraphStore(str(db_path))
    await graph_store.initialize()

    nodes = [
        GraphNode("person", "张三", "张三"),
        GraphNode("topic", "项目", "项目"),
        GraphNode("fact", "周五上线", "周五上线"),
    ]
    node_key_to_id = await graph_store.upsert_nodes(nodes)
    edges = [
        GraphEdge(
            source_key="person:张三",
            target_key="topic:项目",
            relation_type="mentioned_in",
            source_memory_id=1,
        ),
        GraphEdge(
            source_key="topic:项目",
            target_key="fact:周五上线",
            relation_type="describes",
            source_memory_id=2,
        ),
    ]
    edge_key_to_id = await graph_store.add_edges(edges, node_key_to_id)
    await graph_store.add_entries(
        [
            GraphEntry(
                entry_key="entry-1",
                source_memory_id=1,
                session_id="s1",
                persona_id="p1",
                entry_type="participant",
                content="Participant: 张三",
                metadata={"importance": 0.5},
                node_keys=["person:张三"],
                relation_type="participant",
            ),
            GraphEntry(
                entry_key="entry-2",
                source_memory_id=2,
                session_id="s1",
                persona_id="p1",
                entry_type="fact",
                content="Fact: 周五上线",
                metadata={"importance": 0.5},
                node_keys=["fact:周五上线"],
                relation_type="fact",
            ),
        ],
        node_key_to_id,
        edge_key_to_id,
    )

    retriever = GraphKeywordRetriever(
        graph_store,
        TextProcessor(),
        config={
            "graph_expansion_hops": 2,
            "graph_expansion_limit": 12,
            "graph_second_hop_weight": 0.4,
        },
    )

    results = await retriever.search("张三", limit=5, session_id="s1", persona_id="p1")
    assert {item.doc_id for item in results} >= {1, 2}
    second_hop = next(item for item in results if item.doc_id == 2)
    assert "graph_second_hop" in second_hop.metadata["graph_match_source"]


@pytest.mark.asyncio
async def test_memory_engine_dual_route_promotes_graph_hits(tmp_path: Path):
    doc_db_path = tmp_path / "memory.db"
    engine = MemoryEngine(
        db_path=str(doc_db_path),
        faiss_db=_FakeFaissDB(),
        graph_vector_db=_FakeFaissDB(),
        config={
            "fallback_enabled": True,
            "graph_memory_enabled": True,
            # 合并后的单一图路权重；文档路权重由引擎自动推导为 1 - 0.4 = 0.6
            "graph_route_weight": 0.4,
        },
    )
    await engine.initialize()

    matching_id = await engine.add_memory(
        content="项目讨论记录",
        session_id="test:private:s1",
        persona_id="persona_1",
        importance=0.8,
        metadata={
            "topics": ["项目讨论"],
            "participants": ["张三"],
            "key_facts": ["明天下午三点开会"],
            "canonical_summary": "项目讨论记录",
        },
    )
    other_id = await engine.add_memory(
        content="普通对话记录",
        session_id="test:private:s1",
        persona_id="persona_1",
        importance=0.5,
        metadata={
            "topics": ["闲聊"],
            "participants": ["王五"],
            "key_facts": ["天气不错"],
            "canonical_summary": "普通对话记录",
        },
    )

    assert matching_id != other_id

    results = await engine.search_memories(
        query="张三",
        k=2,
        session_id="test:private:s1",
        persona_id="persona_1",
    )
    assert results
    assert results[0].doc_id == matching_id
    assert (results[0].score_breakdown or {}).get("graph_keyword_score", 0.0) > 0

    await engine.close()


@pytest.mark.asyncio
async def test_memory_engine_rebuild_graph_index(tmp_path: Path):
    doc_db_path = tmp_path / "memory_rebuild.db"
    engine = MemoryEngine(
        db_path=str(doc_db_path),
        faiss_db=_FakeFaissDB(),
        graph_vector_db=_FakeFaissDB(),
        config={"fallback_enabled": True, "graph_memory_enabled": True},
    )
    await engine.initialize()

    memory_id = await engine.add_memory(
        content="项目讨论记录",
        session_id="test:private:s1",
        persona_id="persona_1",
        importance=0.8,
        metadata={
            "topics": ["项目讨论"],
            "participants": ["张三"],
            "key_facts": ["明天下午三点开会"],
            "canonical_summary": "项目讨论记录",
        },
    )
    assert memory_id > 0

    assert engine.graph_memory_manager is not None
    await engine.graph_memory_manager.delete_memory(memory_id)

    rebuild_result = await engine.rebuild_graph_index()
    assert rebuild_result["rebuilt"] >= 1

    stats = await engine.get_statistics()
    assert stats.get("graph_entries", 0) >= 1
    assert stats["graph_index"]["state"] == "switched"
    assert stats["graph_index"]["memory_vectors"] == 1
    assert stats["graph_index"]["legacy_vectors"] == 0
    await engine.close()


@pytest.mark.asyncio
async def test_graph_index_status_detects_legacy_vectors(tmp_path: Path):
    engine = MemoryEngine(
        db_path=str(tmp_path / "memory_legacy_graph.db"),
        faiss_db=_FakeFaissDB(),
        graph_vector_db=_FakeFaissDB(),
        config={"fallback_enabled": True, "graph_memory_enabled": True},
    )
    await engine.initialize()

    assert engine.graph_vector_db is not None
    await engine.graph_vector_db.insert(
        content="旧 entry 级图向量",
        metadata={"source_memory_id": 1},
    )

    status = await engine.get_graph_index_status()

    assert status["state"] == "rebuild_required"
    assert status["total_vectors"] == 1
    assert status["memory_vectors"] == 0
    assert status["legacy_vectors"] == 1
    assert status["orphan_vectors"] == 1
    await engine.close()


@pytest.mark.asyncio
async def test_graph_rebuild_switches_complete_shadow_generation(tmp_path: Path):
    graph_store = GraphStore(str(tmp_path / "graph_shadow.db"))
    await graph_store.initialize()
    vector_db = _FakeFaissDB()
    manager = GraphMemoryManager(
        graph_store,
        GraphVectorRetriever(vector_db),
        GraphExtractor(),
    )
    metadata = {
        "session_id": "test:private:s1",
        "persona_id": "persona_1",
        "topics": ["旧主题"],
        "key_facts": ["旧事实"],
    }
    await manager.index_memory(1, "旧记忆一", metadata)
    await manager.index_memory(2, "旧记忆二", metadata)
    orphan_vector_id = await vector_db.insert(
        "孤立旧图向量",
        {"source_memory_id": 999},
    )

    result = await manager.rebuild_memories(
        [
            (
                1,
                "新记忆一",
                {
                    **metadata,
                    "topics": ["新主题"],
                    "key_facts": ["新事实"],
                },
            ),
            (
                3,
                "新记忆三",
                {
                    **metadata,
                    "topics": ["第三主题"],
                    "key_facts": ["第三事实"],
                },
            ),
        ]
    )

    assert orphan_vector_id not in vector_db.docs

    assert result == {"rebuilt": 2, "skipped": 0}
    vectors_by_source = await graph_store.list_vector_doc_ids_by_source()
    assert set(vectors_by_source) == {1, 3}
    assert all(len(vector_ids) == 1 for vector_ids in vectors_by_source.values())
    assert len(vector_db.docs) == 2
    assert all(
        document["metadata"].get("graph_vector_granularity") == "memory"
        for document in vector_db.docs.values()
    )


@pytest.mark.asyncio
async def test_graph_rebuild_failure_preserves_live_generation(tmp_path: Path):
    class ToggleFailRetriever(GraphVectorRetriever):
        fail = False

        async def add_memory_entries_batch(self, memories):
            if self.fail:
                raise RuntimeError("embedding unavailable")
            return await super().add_memory_entries_batch(memories)

    graph_store = GraphStore(str(tmp_path / "graph_failure.db"))
    await graph_store.initialize()
    vector_db = _FakeFaissDB()
    retriever = ToggleFailRetriever(vector_db)
    manager = GraphMemoryManager(graph_store, retriever, GraphExtractor())
    metadata = {
        "session_id": "test:private:s1",
        "persona_id": "persona_1",
        "topics": ["旧主题"],
        "key_facts": ["旧事实"],
    }
    await manager.index_memory(7, "必须保留的旧记忆", metadata)
    old_stats = await graph_store.get_memory_entry_stats()
    old_vectors = dict(vector_db.docs)

    retriever.fail = True
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await manager.rebuild_memories([(8, "不能切换的新记忆", metadata)])

    assert await graph_store.get_memory_entry_stats() == old_stats
    assert await graph_store.list_vector_doc_ids_by_source() == {7: [1]}
    assert vector_db.docs == old_vectors


@pytest.mark.asyncio
async def test_graph_rebuild_replays_writes_arriving_during_shadow_build(
    tmp_path: Path,
):
    class PausingRetriever(GraphVectorRetriever):
        def __init__(self, faiss_db):
            super().__init__(faiss_db)
            self.pause_next_batch = False
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def add_memory_entries_batch(self, memories):
            if self.pause_next_batch:
                self.pause_next_batch = False
                self.started.set()
                await self.release.wait()
            return await super().add_memory_entries_batch(memories)

    graph_store = GraphStore(str(tmp_path / "graph_delta.db"))
    await graph_store.initialize()
    vector_db = _FakeFaissDB()
    retriever = PausingRetriever(vector_db)
    manager = GraphMemoryManager(graph_store, retriever, GraphExtractor())
    metadata = {
        "session_id": "test:private:s1",
        "persona_id": "persona_1",
        "topics": ["主题"],
        "key_facts": ["事实"],
    }
    await manager.index_memory(1, "已有记忆", metadata)

    retriever.pause_next_batch = True
    rebuild_task = asyncio.create_task(
        manager.rebuild_memories([(1, "重建后的已有记忆", metadata)])
    )
    await retriever.started.wait()
    await manager.index_memory(2, "重建期间写入的新记忆", metadata)
    retriever.release.set()
    await rebuild_task

    vectors_by_source = await graph_store.list_vector_doc_ids_by_source()
    assert set(vectors_by_source) == {1, 2}
    assert all(len(vector_ids) == 1 for vector_ids in vectors_by_source.values())
    assert len(vector_db.docs) == 2


@pytest.mark.asyncio
async def test_graph_store_batch_delete_memories(tmp_path: Path):
    """GraphStore.batch_delete_memories 应一次删除多条 source_memory 的图数据。"""
    doc_db_path = tmp_path / "graph_batch_del.db"
    engine = MemoryEngine(
        db_path=str(doc_db_path),
        faiss_db=_FakeFaissDB(),
        graph_vector_db=_FakeFaissDB(),
        config={"fallback_enabled": True, "graph_memory_enabled": True},
    )
    await engine.initialize()

    ids = []
    for i in range(3):
        mid = await engine.add_memory(
            content=f"批量图删除测试{i}：讨论项目进度",
            session_id="test:private:s1",
            persona_id="persona_1",
            importance=0.8,
            metadata={
                "topics": ["项目"],
                "canonical_summary": f"批量图删除测试{i}",
            },
        )
        ids.append(mid)

    stats_before = await engine.graph_store.get_memory_entry_stats()

    assert engine.graph_memory_manager is not None
    await engine.graph_memory_manager.batch_delete_memories(ids)

    stats_after = await engine.graph_store.get_memory_entry_stats()
    assert stats_after["graph_entries"] < stats_before["graph_entries"]

    await engine.close()


@pytest.mark.asyncio
async def test_subgraph_memory_stats_recompute_after_node_limit(tmp_path: Path):
    store = GraphStore(str(tmp_path / "graph_limit.db"))
    await store.initialize()
    node_key_to_id = await store.upsert_nodes(
        [
            GraphNode("topic", "Alpha", "alpha"),
            GraphNode("topic", "Beta", "beta"),
        ]
    )
    await store.add_entries(
        [
            GraphEntry(
                entry_key="entry-alpha",
                source_memory_id=1,
                session_id="s1",
                persona_id="p1",
                entry_type="fact",
                content="Alpha fact",
                metadata={"importance": 0.8},
                node_keys=["topic:alpha"],
            ),
            GraphEntry(
                entry_key="entry-beta",
                source_memory_id=1,
                session_id="s1",
                persona_id="p1",
                entry_type="fact",
                content="Beta fact",
                metadata={"importance": 0.8},
                node_keys=["topic:beta"],
            ),
        ],
        node_key_to_id,
        {},
    )

    subgraph = await store.get_subgraph_for_memories([1], limit_nodes=1)

    assert len(subgraph["nodes"]) == 1
    assert len(subgraph["entries"]) == 1
    assert subgraph["memories"][0]["entry_count"] == 1
    assert subgraph["memories"][0]["node_count"] == 1


@pytest.mark.asyncio
async def test_search_nodes_by_tokens_batches_large_token_lists(tmp_path: Path):
    store = GraphStore(str(tmp_path / "graph_tokens.db"))
    await store.initialize()
    await store.upsert_nodes(
        [
            GraphNode("topic", "needle alpha", "needle alpha"),
            GraphNode("topic", "needle beta", "needle beta"),
        ]
    )
    tokens = [f"token-{index}" for index in range(1100)]
    tokens.extend(["needle", "needle", ""])

    results = await store.search_nodes_by_tokens(tokens, limit=10)

    assert {item["canonical_value"] for item in results} == {
        "needle beta",
        "needle alpha",
    }
