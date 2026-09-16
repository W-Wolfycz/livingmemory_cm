"""检索组件回归：向量候选池扩容 + MMR 去重分词。

候选池部分：AstrBot 的 ``FaissVecDB.retrieve()`` 先取 ``fetch_k`` 条全局最近邻、
再按 metadata 过滤，且每次调用都会重新 embedding。这里验证我们自实现的路径：
只 embedding 一次、候选不足时按需扩容，且不会被 ``get_documents`` 默认 limit=100
截断。

MMR 部分：纯空格分词对中文整句只产生一个 token，近似重复的 Jaccard 恒为 0，
去重形同失效；验证传入 TextProcessor 后按词级相似度真正生效。
"""

from __future__ import annotations

import pytest

from livingmemory_cm.core.processors.text_processor import TextProcessor
from livingmemory_cm.core.retrieval.hybrid_retriever import (
    HybridResult,
    HybridRetriever,
)
from livingmemory_cm.core.retrieval.vector_retriever import VectorRetriever


class _FakeEmbeddingProvider:
    def __init__(self, dim: int = 4) -> None:
        self.calls = 0
        self.dim = dim

    async def get_embedding(self, text: str):
        self.calls += 1
        return [0.1] * self.dim


class _FakeEmbeddingStorage:
    """模拟 FAISS：id 越小越近，按需返回最近的 k 条。"""

    def __init__(self, total: int) -> None:
        self.total = total
        self.calls: list[int] = []

    async def search(self, vector, k: int):
        self.calls.append(k)
        ids = list(range(1, min(k, self.total) + 1))
        scores = [[0.1 * i for i in range(len(ids))]]
        return scores, [ids]


class _FakeDocumentStorage:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self.limits: list[int | None] = []

    async def get_documents(
        self, metadata_filters=None, ids=None, offset=0, limit=100
    ):
        self.limits.append(limit)
        filters = metadata_filters or {}
        selected = [
            doc
            for doc in self.docs
            if (ids is None or doc["id"] in ids)
            and all(doc["metadata"].get(k) == v for k, v in filters.items())
        ]
        return selected[:limit] if limit else selected


def _make_db(total: int, target_ids: set[int], session: str = "s-target"):
    docs = []
    for doc_id in range(1, total + 1):
        docs.append(
            {
                "id": doc_id,
                "doc_id": f"uuid-{doc_id}",
                "text": f"内容 {doc_id}",
                "metadata": {
                    "session_id": session if doc_id in target_ids else "s-other",
                    "persona_id": "persona_demo",
                },
            }
        )
    provider = _FakeEmbeddingProvider()
    storage = _FakeEmbeddingStorage(total)
    documents = _FakeDocumentStorage(docs)
    db = type(
        "FakeFaiss",
        (),
        {
            "embedding_provider": provider,
            "embedding_storage": storage,
            "document_storage": documents,
        },
    )()
    return db, provider, storage, documents


@pytest.mark.asyncio
async def test_candidate_pool_expands_until_enough_survive_filter() -> None:
    """过滤后不足 k 条时自动扩容，且只 embedding 一次。"""
    db, provider, storage, documents = _make_db(total=500, target_ids=set(range(101, 106)))
    retriever = VectorRetriever(db)

    results = await retriever.search(
        "查询", k=5, session_id="s-target", persona_id="persona_demo"
    )

    assert [item.doc_id for item in results] == [101, 102, 103, 104, 105]
    assert provider.calls == 1, "扩容不应重复调用 embedding"
    assert storage.calls == [100, 300], "首轮 100 条不足，应按 3 倍扩容一次"
    assert documents.limits == [100, 300], (
        "必须显式放开 get_documents 默认 limit=100，按候选数取回"
    )


@pytest.mark.asyncio
async def test_candidate_pool_stops_when_index_exhausted() -> None:
    """索引取空即停止扩容，不做无意义重试。"""
    db, provider, storage, _ = _make_db(total=40, target_ids=set())
    retriever = VectorRetriever(db)

    results = await retriever.search("查询", k=5, session_id="s-target")

    assert results == []
    assert len(storage.calls) == 1, f"索引只有 40 条，取空后不应再扩容: {storage.calls}"
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_scores_are_normalized_and_sorted_desc() -> None:
    db, _, _, _ = _make_db(total=10, target_ids={1, 2, 3, 10})
    retriever = VectorRetriever(db)

    results = await retriever.search("查询", k=4, session_id="s-target")

    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


@pytest.mark.asyncio
async def test_falls_back_to_core_retrieve_when_internals_missing() -> None:
    """核心对象结构不符合预期时退回 FaissVecDB.retrieve()。"""
    calls: list[dict] = []

    class _FakeRetrieveOnly:
        async def retrieve(self, query, k, fetch_k, rerank, metadata_filters):
            calls.append(
                {
                    "query": query,
                    "k": k,
                    "fetch_k": fetch_k,
                    "rerank": rerank,
                    "metadata_filters": metadata_filters,
                }
            )
            return [
                type(
                    "R",
                    (),
                    {
                        "similarity": 0.8,
                        "data": {"id": 7, "text": "回退路径", "metadata": {"importance": 0.5}},
                    },
                )()
            ]

    retriever = VectorRetriever(_FakeRetrieveOnly())
    results = await retriever.search("查询", k=5, session_id="s-1")

    assert [item.doc_id for item in results] == [7]
    assert results[0].content == "回退路径"
    assert calls[0]["fetch_k"] == 100, "回退路径也应放大候选池"
    assert calls[0]["rerank"] is False


@pytest.mark.asyncio
async def test_embedding_failure_returns_empty() -> None:
    class _BrokenProvider:
        async def get_embedding(self, text):
            raise RuntimeError("provider down")

    db, _, _, _ = _make_db(total=10, target_ids={1})
    db.embedding_provider = _BrokenProvider()
    retriever = VectorRetriever(db)

    assert await retriever.search("查询", k=5, session_id="s-target") == []


# ── MMR 去重分词 ─────────────────────────────────────────────────────────────


def _near_duplicate_docs():
    return [
        HybridResult(1, 0.90, 0.9, "用户喜欢吃寿司", {}),
        HybridResult(2, 0.88, 0.9, "用户爱吃寿司", {}),
        HybridResult(3, 0.86, 0.8, "用户周末去爬山", {}),
    ]


def test_mmr_dedups_chinese_near_duplicates_with_tokenizer() -> None:
    """分数接近时，近似重复的中文记忆应被多样性项压下去。"""
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.mmr_lambda = 0.7
    retriever.text_processor = TextProcessor()

    picked = [item.doc_id for item in retriever._apply_mmr(_near_duplicate_docs(), 2)]

    assert picked == [1, 3]


def test_mmr_without_tokenizer_still_returns_k() -> None:
    """没传分词器时退化为空格分词，不报错（行为弱化但不崩）。"""
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.mmr_lambda = 0.7
    retriever.text_processor = None

    picked = retriever._apply_mmr(_near_duplicate_docs(), 2)

    assert len(picked) == 2
    assert [item.doc_id for item in picked][0] == 1


# ── documents 表达式索引是否真的被查询命中 ───────────────────────────────────


@pytest.mark.asyncio
async def test_status_created_index_is_used_by_list_and_recent_queries(tmp_path) -> None:
    """表达式索引必须与查询同形；用 EXPLAIN QUERY PLAN 锁住，防止表达式漂移。"""
    import json
    import time

    import aiosqlite

    from livingmemory_cm.core.memory.memory_schema_service import (
        MemorySchemaContext,
        MemorySchemaService,
    )
    from livingmemory_cm.core.memory.memory_search_service import (
        _DOC_CREATED_EXPR,
        _DOC_STATUS_EXPR,
    )

    connection = await aiosqlite.connect(tmp_path / "indexes.db")
    connection.row_factory = aiosqlite.Row

    async def _noop() -> None:
        return None

    await MemorySchemaService().create_tables(
        MemorySchemaContext(db_connection=connection, create_write_ops_table=_noop)
    )

    now = time.time()
    await connection.executemany(
        "INSERT INTO documents(id, doc_id, text, metadata, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, '', '')",
        [
            (
                i,
                f"uuid-{i}",
                f"记忆 {i}",
                json.dumps(
                    {
                        "session_id": f"demo:GroupMessage:group_{i % 20}",
                        "persona_id": f"persona_{i % 4}",
                        "status": "active" if i % 10 else "archived",
                        "memory_type": "GENERAL",
                        "create_time": now - (500 - i) * 60,
                    }
                ),
            )
            for i in range(1, 501)
        ],
    )
    await connection.commit()

    async def plan(sql: str, params=()) -> str:
        cursor = await connection.execute("EXPLAIN QUERY PLAN " + sql, params)
        rows = await cursor.fetchall()
        return " | ".join(str(row["detail"]) for row in rows)

    list_plan = await plan(
        f"SELECT id FROM documents WHERE {_DOC_STATUS_EXPR} = 'active' "
        f"ORDER BY {_DOC_CREATED_EXPR} DESC, id DESC LIMIT 20"
    )
    recent_plan = await plan(
        f"SELECT id FROM documents WHERE {_DOC_STATUS_EXPR} = 'active' "
        "AND CASE WHEN json_valid(metadata) THEN json_extract(metadata, "
        "'$.session_id') END = ? "
        f"AND {_DOC_CREATED_EXPR} >= ? "
        f"ORDER BY {_DOC_CREATED_EXPR} DESC, id DESC LIMIT 6",
        ("demo:GroupMessage:group_1", now - 72 * 3600),
    )

    assert "idx_doc_status_created" in list_plan, list_plan
    assert "idx_doc_status_created" in recent_plan, recent_plan

    await connection.close()


# ── Rerank 重排序 ────────────────────────────────────────────────────────────


class _FakeRerankResult:
    def __init__(self, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class _FakeRerankProvider:
    def __init__(self, scores=None, error=None, empty=False, delay=0.0) -> None:
        self.scores = scores
        self.error = error
        self.empty = empty
        self.delay = delay
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query, documents, top_n=None):
        import asyncio

        self.calls.append((query, list(documents)))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.empty:
            return []
        if self.scores is not None:
            return [
                _FakeRerankResult(index, score)
                for index, score in enumerate(self.scores)
            ]
        return []


def _rerank_service(provider, **config_overrides):
    from livingmemory_cm.core.memory.memory_search_service import MemorySearchService

    config = {"rerank_enabled": True, "rerank_candidates": 20, "rerank_weight": 1.0}
    config.update(config_overrides)
    return MemorySearchService(config, rerank_provider=provider)


def _candidates(scores: list[float]):
    return [
        HybridResult(index, score, score, f"记忆 {index}", {})
        for index, score in enumerate(scores)
    ]


@pytest.mark.asyncio
async def test_rerank_reorders_candidates_by_relevance() -> None:
    """重排分数按候选集归一化后与 final_score 混合，顺序随重排改变。"""
    provider = _FakeRerankProvider(scores=[0.1, 0.9, 0.5])
    service = _rerank_service(provider, rerank_weight=1.0)

    results = await service._apply_rerank("查询", _candidates([0.9, 0.5, 0.1]))

    assert [item.doc_id for item in results] == [1, 2, 0]
    assert results[0].score_breakdown["rerank"] == 1.0
    assert results[-1].score_breakdown["rerank"] == 0.0
    assert provider.calls[0][1] == ["记忆 0", "记忆 1", "记忆 2"]


@pytest.mark.asyncio
async def test_rerank_blends_with_existing_score() -> None:
    """权重 0.5 时不会把原分数完全丢掉。"""
    provider = _FakeRerankProvider(scores=[0.0, 1.0])
    service = _rerank_service(provider, rerank_weight=0.5)

    results = await service._apply_rerank("查询", _candidates([0.9, 0.4]))

    scores = {item.doc_id: item.final_score for item in results}
    assert scores[0] == pytest.approx(0.5 * 0.0 + 0.5 * 0.9)
    assert scores[1] == pytest.approx(0.5 * 1.0 + 0.5 * 0.4)
    assert [item.doc_id for item in results] == [1, 0]


@pytest.mark.asyncio
async def test_rerank_failures_fall_back_to_original_order() -> None:
    """调用失败/返回空/超时都必须沿用原排序，绝不能影响召回可用性。"""
    original = _candidates([0.9, 0.5, 0.1])
    expected = [item.doc_id for item in original]

    for provider in (
        _FakeRerankProvider(error=RuntimeError("provider down")),
        _FakeRerankProvider(empty=True),
    ):
        service = _rerank_service(provider)
        results = await service._apply_rerank("查询", _candidates([0.9, 0.5, 0.1]))
        assert [item.doc_id for item in results] == expected

    slow = _FakeRerankProvider(scores=[0.0, 1.0], delay=0.5)
    service = _rerank_service(slow, search_timeout_seconds=0.05)
    results = await service._apply_rerank("查询", _candidates([0.9, 0.5]))
    assert [item.doc_id for item in results] == [0, 1]


@pytest.mark.asyncio
async def test_rerank_skipped_when_disabled_or_without_provider() -> None:
    from livingmemory_cm.core.memory.memory_search_service import MemorySearchService

    provider = _FakeRerankProvider(scores=[0.0, 1.0])
    disabled = MemorySearchService(
        {"rerank_enabled": False}, rerank_provider=provider
    )
    without_provider = MemorySearchService({"rerank_enabled": True})

    for service in (disabled, without_provider):
        results = await service._apply_rerank("查询", _candidates([0.9, 0.5]))
        assert [item.doc_id for item in results] == [0, 1]
    assert provider.calls == []


@pytest.mark.asyncio
async def test_rerank_only_scores_head_and_keeps_tail() -> None:
    """超出候选上限的部分不参与重排，但保留在尾部。"""
    provider = _FakeRerankProvider(scores=[0.0, 1.0])
    service = _rerank_service(provider, rerank_candidates=2, rerank_weight=1.0)

    results = await service._apply_rerank("查询", _candidates([0.9, 0.5, 0.4, 0.3]))

    assert len(provider.calls[0][1]) == 2
    assert [item.doc_id for item in results[:2]] == [1, 0]
    assert [item.doc_id for item in results[2:]] == [2, 3]
