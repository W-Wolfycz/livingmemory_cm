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


# ── documents 表达式索引是否真的被查询命中 ───────────────────────────────────


@pytest.mark.asyncio
async def test_metadata_expression_indexes_back_every_hot_query(tmp_path) -> None:
    """EXPLAIN QUERY PLAN 锁住热路径的索引命中，守住"表达式漂移→静默全表扫描"。

    表达式只在 `core/memory/metadata_expr.py` 定义一次，索引 DDL 与查询都从这里取；
    一旦哪边被手写副本改歪，计划会退化为 SCAN / TEMP B-TREE，本用例就会失败。
    已知且接受的例外：按重要性排序（WebUI 手动排序）仍会走 TEMP B-TREE——它没有
    对应的表达式索引，这里不为其断言。
    """
    import json
    import time

    import aiosqlite

    from livingmemory_cm.core.memory.memory_schema_service import (
        MemorySchemaContext,
        MemorySchemaService,
    )
    from livingmemory_cm.core.memory.memory_search_service import (
        _DOC_CREATED_EXPR,
        _DOC_PERSONA_EXPR,
        _DOC_SESSION_EXPR,
        _DOC_STATUS_EXPR,
    )
    from livingmemory_cm.core.memory.metadata_expr import TYPE_EXPR as _DOC_TYPE_EXPR
    from livingmemory_cm.core.page_api_modules.memory_handler import MemoryHandler
    from livingmemory_cm.core.page_api_modules.utils import PageApiUtils

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
                        "session_id": f"demo:GroupMessage:group_{i % 30}",
                        "persona_id": f"persona_{i % 5}",
                        "status": "active" if i % 7 else "archived",
                        "memory_type": ["GENERAL", "FACT", "PREFERENCE"][i % 3],
                        "importance": (i % 10) / 10,
                        "create_time": now - (2000 - i) * 60,
                    }
                ),
            )
            for i in range(1, 2001)
        ],
    )
    await connection.commit()
    await connection.execute("ANALYZE")

    async def plan(sql: str, params=()) -> str:
        cursor = await connection.execute("EXPLAIN QUERY PLAN " + sql, params)
        rows = await cursor.fetchall()
        return " | ".join(str(row["detail"]) for row in rows)

    class _Query(dict):
        """页面 API 的过滤参数替身（只用到 .get）。"""

        def get(self, key, default=None):
            return dict.get(self, key, default)

    handler = MemoryHandler(PageApiUtils())
    plans: dict[str, str] = {}
    for label, args in (
        ("all", {"status": "all", "sort": "created_desc"}),
        ("status", {"status": "active", "sort": "created_desc"}),
        ("type", {"type": "FACT", "sort": "created_desc"}),
    ):
        spec, error = handler._build_query_spec(_Query(args))
        assert error is None
        where_clause, sort_expr = spec["where_clause"], spec["sort_expr"]
        plans[label + "/count"] = await plan(
            f"SELECT COUNT(*) FROM documents {where_clause}", spec["params"]
        )
        plans[label + "/page"] = await plan(
            f"SELECT id FROM documents {where_clause} ORDER BY {sort_expr}"
            " LIMIT 20 OFFSET 0",
            spec["params"],
        )

    plans["recent"] = await plan(
        f"SELECT id FROM documents WHERE {_DOC_STATUS_EXPR} = 'active' "
        f"AND {_DOC_SESSION_EXPR} = ? AND {_DOC_PERSONA_EXPR} = ? "
        f"AND {_DOC_CREATED_EXPR} >= ? "
        f"ORDER BY {_DOC_CREATED_EXPR} DESC, id DESC LIMIT 6",
        ("demo:GroupMessage:group_1", "persona_1", now - 72 * 3600),
    )
    plans["type_prefix"] = await plan(
        f"SELECT id FROM documents WHERE {_DOC_TYPE_EXPR} = ? LIMIT 5", ("FACT",)
    )
    plans["status_prefix"] = await plan(
        f"SELECT id FROM documents WHERE {_DOC_STATUS_EXPR} = ? LIMIT 5", ("active",)
    )
    await connection.close()

    # 每个热路径都必须命中对应索引，且过滤类查询不能退化成全表扫描
    assert "idx_doc_created" in plans["all/page"], plans["all/page"]
    assert "idx_doc_status_created" in plans["status/count"], plans["status/count"]
    assert "idx_doc_status_created" in plans["status/page"], plans["status/page"]
    assert "idx_doc_type_created" in plans["type/count"], plans["type/count"]
    assert "idx_doc_type_created" in plans["type/page"], plans["type/page"]
    assert "idx_doc_status_created" in plans["recent"], plans["recent"]
    assert "SCAN documents" not in plans["status/count"], plans["status/count"]
    assert "SCAN documents" not in plans["type/count"], plans["type/count"]
    assert "SCAN documents" not in plans["recent"], plans["recent"]
    # 排序由索引直接提供 → 不应再建临时 B-TREE（否则等于排了全表再截断）
    for key in ("all/page", "status/page", "type/page", "recent"):
        assert "TEMP B-TREE" not in plans[key], f"{key}: {plans[key]}"
    # 索引单独可用（用 status/type 做前缀过滤时仍是覆盖索引扫描）
    assert "idx_doc_type_created" in plans["type_prefix"], plans["type_prefix"]
    assert "idx_doc_status_created" in plans["status_prefix"], plans["status_prefix"]


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
async def test_rerank_resolver_is_used_per_call_not_cached() -> None:
    """守的静默失效：AstrBot 重建 provider 后旧实例 httpx 已关闭，若长期缓存实例，
    重排会永久静默降级（无报错、只是不再重排）。"""
    from livingmemory_cm.core.memory.memory_search_service import MemorySearchService

    first = _FakeRerankProvider(scores=[1.0, 0.0])
    second = _FakeRerankProvider(scores=[1.0, 0.0])
    resolved: list[_FakeRerankProvider] = [first, second, None]

    def resolver():
        return resolved.pop(0)

    service = MemorySearchService(
        {"rerank_enabled": True, "rerank_weight": 1.0},
        rerank_provider=None,
        rerank_provider_resolver=resolver,
    )

    await service._apply_rerank("查询", _candidates([0.9, 0.4]))
    await service._apply_rerank("查询", _candidates([0.9, 0.4]))
    # 第三次 resolver 返回 None（provider 已被移除）→ 沿用原排序且不报错
    unchanged = await service._apply_rerank("查询", _candidates([0.9, 0.4]))

    assert first.calls and second.calls, "每次调用都应重新解析一次"
    assert [item.doc_id for item in unchanged] == [0, 1]


@pytest.mark.asyncio
async def test_dual_route_batches_missing_payload_lookup() -> None:
    """守的静默性能退化：候选缺正文时逐条回库会让召回延迟按候选数线性增长。"""
    from livingmemory_cm.core.retrieval.dual_route_retriever import DualRouteRetriever

    class _Route:
        def __init__(self, results):
            self._results = results

        async def search(self, query, k, session_id=None, persona_id=None):
            return self._results

    class _GraphResult:
        """图路结果比文档路多 keyword_score/vector_score/score_breakdown。"""

        def __init__(self, doc_id, score, content, metadata):
            self.doc_id = doc_id
            self.final_score = score
            self.vector_score = score
            self.keyword_score = score
            self.content = content
            self.metadata = metadata
            self.score_breakdown: dict[str, float] = {}

    doc_route = _Route(
        [HybridResult(1, 0.9, 0.9, "", {}), HybridResult(2, 0.8, 0.8, "", {})]
    )
    graph_route = _Route([_GraphResult(3, 0.7, "图路内容", {"persona_id": "p"})])
    single_calls: list[int] = []
    batch_calls: list[list[int]] = []

    async def single_loader(doc_id):
        single_calls.append(doc_id)
        return None

    async def batch_loader(doc_ids):
        batch_calls.append(list(doc_ids))
        return {
            doc_id: {"id": doc_id, "text": f"补取内容 {doc_id}", "metadata": {"persona_id": "p"}}
            for doc_id in doc_ids
        }

    retriever = DualRouteRetriever(
        doc_route,
        graph_route,
        single_loader,
        {},
        memory_batch_loader=batch_loader,
    )
    results = await retriever.search("查询", k=5)

    assert sorted(batch_calls[0]) == [1, 2, 3], "缺内容的候选（含图路独有）必须一次批量取回"
    assert len(batch_calls) == 1
    assert single_calls == []
    assert {item.doc_id for item in results} == {1, 2, 3}
    assert all(item.content for item in results)


# ── 超长内容写入（全文入库 / 向量输入截断 / 失败回滚）──────────────────────


class _RecordingDocumentStorage:
    def __init__(self) -> None:
        self.inserted: list[tuple[str, str, dict]] = []

    async def insert_document(self, doc_id, text, metadata):
        self.inserted.append((doc_id, text, metadata))
        return len(self.inserted)


class _FailingVectorStorage:
    def __init__(self) -> None:
        self.deleted: list[list[int]] = []

    async def insert(self, vector, id):
        raise RuntimeError("faiss down")

    async def delete(self, ids):
        self.deleted.append(list(ids))


class _RecordingVectorStorage:
    def __init__(self) -> None:
        self.inserted: list[int] = []

    async def insert(self, vector, id):
        self.inserted.append(id)


class _DeleteRecordingFaiss:
    def __init__(self, documents, vectors, embedder) -> None:
        self.document_storage = documents
        self.embedding_storage = vectors
        self.embedding_provider = embedder
        self.deleted_uuids: list[str] = []
        self.core_inserted: list[str] = []

    async def insert(self, content, metadata=None):
        self.core_inserted.append(content)
        return 0

    async def delete(self, doc_id):
        self.deleted_uuids.append(doc_id)
        self.document_storage.inserted = [
            item for item in self.document_storage.inserted if item[0] != doc_id
        ]


def _long_content_retriever(faiss):
    from livingmemory_cm.core.retrieval.vector_retriever import VectorRetriever

    return VectorRetriever(faiss)


@pytest.mark.asyncio
async def test_long_content_stores_full_text_and_truncates_only_embedding() -> None:
    """守的静默失效：直接截断后入库会让中间内容永久丢失（列表/导出里看不出来）。"""
    from livingmemory_cm.core.retrieval.vector_retriever import _MAX_EMBEDDING_CHARS

    documents = _RecordingDocumentStorage()
    vectors = _RecordingVectorStorage()
    embedder = _FakeEmbeddingProvider()
    embed_inputs: list[str] = []

    async def _get_embedding(text):
        embed_inputs.append(text)
        return [0.1] * 4

    embedder.get_embedding = _get_embedding
    faiss = _DeleteRecordingFaiss(documents, vectors, embedder)

    content = "头" * 3000 + "中段唯一内容" + "尾" * 2000
    doc_id = await _long_content_retriever(faiss).add_document(
        content,
        {
            "importance": 0.5,
            "create_time": 1.0,
            "last_access_time": 1.0,
            "session_id": "s",
            "persona_id": "p",
        },
    )

    assert doc_id == 1
    assert documents.inserted[0][1] == content, "全文必须原样入库"
    assert len(embed_inputs[0]) <= _MAX_EMBEDDING_CHARS
    assert "中段唯一内容" not in embed_inputs[0], "向量输入只保留首尾"
    assert vectors.inserted == [1]


@pytest.mark.asyncio
async def test_long_content_rolls_back_document_when_vector_write_fails() -> None:
    """守的静默失效：向量写失败却不回滚，会留下"有文本没向量"的孤儿记忆。"""
    documents = _RecordingDocumentStorage()
    vectors = _FailingVectorStorage()
    faiss = _DeleteRecordingFaiss(documents, vectors, _FakeEmbeddingProvider())

    content = "长" * 5000
    with pytest.raises(RuntimeError):
        await _long_content_retriever(faiss).add_document(
            content,
            {
                "importance": 0.5,
                "create_time": 1.0,
                "last_access_time": 1.0,
                "session_id": "s",
                "persona_id": "p",
            },
        )

    assert len(faiss.deleted_uuids) == 1, "失败必须按 uuid 回滚文档行"
    assert documents.inserted == [], "回滚后不应残留孤儿文档"


