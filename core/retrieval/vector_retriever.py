"""
向量检索器 - 基于Faiss的向量密集检索
封装AstrBot的FaissVecDB,提供统一的检索接口
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from astrbot.core.db.vec_db.faiss_impl.vec_db import FaissVecDB

from ...log import logger, tag

_TRUNCATED_CONTENT_MARKER = "\n...[中间内容已截断]...\n"


@dataclass
class VectorResult:
    """向量检索结果"""

    doc_id: int
    score: float
    content: str
    metadata: dict[str, Any]


async def delete_faiss_documents_by_ids(
    faiss_db,
    doc_ids: list[int],
) -> list[int] | None:
    """Delete known FAISS document IDs with one index persistence operation.

    Returns ``None`` when the installed AstrBot storage does not expose the
    required bulk-capable primitives, allowing callers to use a compatibility
    fallback.
    """
    if not doc_ids:
        return []

    embedding_storage = getattr(faiss_db, "embedding_storage", None)
    document_storage = getattr(faiss_db, "document_storage", None)
    embedding_delete = getattr(embedding_storage, "delete", None)
    get_documents = getattr(document_storage, "get_documents", None)
    delete_by_uuid = getattr(document_storage, "delete_document_by_doc_id", None)
    if not (
        callable(embedding_delete)
        and callable(get_documents)
        and callable(delete_by_uuid)
    ):
        return None

    unique_ids = list(dict.fromkeys(int(doc_id) for doc_id in doc_ids))
    documents = await get_documents(
        metadata_filters={},
        ids=unique_ids,
        offset=0,
        limit=len(unique_ids),
    )
    deletable = [document for document in documents if document.get("doc_id")]
    if not deletable:
        return []

    found_ids = [int(document["id"]) for document in deletable]
    await embedding_delete(found_ids)
    for document in deletable:
        await delete_by_uuid(document["doc_id"])
    return found_ids


class VectorRetriever:
    """
    向量密集检索器

    封装AstrBot的FaissVecDB,提供统一的向量相似度检索接口。
    主要特性:
    1. 保持查询文本原样检索，避免额外预处理带来的行为分叉
    2. 元数据包含:importance, create_time, last_access_time, session_id, persona_id
    3. 相似度分数已归一化到[0,1]区间
    4. 支持通过metadata过滤session_id和persona_id
    5. ID映射缓存优化UUID查询性能
    6. 候选池自适应扩容：metadata 过滤发生在向量取数之后，候选不足时按需扩大
    """

    # 候选池参数：metadata 过滤会淘汰大量候选（同库多会话/多人格），
    # 取数太少会导致过滤后凑不满 k 条。
    _FETCH_MULTIPLIER = 20
    _MIN_FETCH_K = 64
    _MAX_FETCH_K = 512
    _EXPANSION_FACTOR = 3

    def __init__(
        self,
        faiss_db: FaissVecDB,
        config: dict[str, Any] | None = None,
    ):
        """
        初始化向量检索器

        Args:
            faiss_db: FaissVecDB实例
            config: 配置字典(可选)
        """
        self.faiss_db = faiss_db
        self.config = config or {}

        # 优化3: ID映射缓存 (int_id -> uuid)
        self._id_cache: dict[int, str] = {}
        self._cache_max_size = self.config.get("id_cache_size", 1000)

    @staticmethod
    def _fit_content_for_embedding(content: str, max_chars: int) -> str:
        """Keep both the opening context and tail conclusion within a char budget."""
        if len(content) <= max_chars:
            return content

        if max_chars <= len(_TRUNCATED_CONTENT_MARKER):
            return content[:max_chars]

        available = max_chars - len(_TRUNCATED_CONTENT_MARKER)
        head_chars = available // 2
        tail_chars = available - head_chars
        return content[:head_chars] + _TRUNCATED_CONTENT_MARKER + content[-tail_chars:]

    async def add_document(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> int:
        """
        添加文档到向量库

        Args:
            content: 文档内容
            metadata: 文档元数据(必须包含:importance, create_time, last_access_time,
                     session_id, persona_id)

        Returns:
            int: 文档ID
        """
        # 确保metadata存在
        metadata = metadata or {}

        # 验证必需的元数据字段
        required_fields = [
            "importance",
            "create_time",
            "last_access_time",
            "session_id",
            "persona_id",
        ]
        for field in required_fields:
            if field not in metadata:
                # 提供默认值
                if field == "importance":
                    metadata[field] = 0.5
                elif field in ["create_time", "last_access_time"]:
                    import time

                    metadata[field] = time.time()
                else:  # session_id, persona_id
                    metadata[field] = None

        # 插入到Faiss向量库，同样截断过长内容防止 embedding token 超限
        _MAX_CONTENT_CHARS = 4000
        insert_content = content
        if len(insert_content) > _MAX_CONTENT_CHARS:
            logger.warning(
                f"{tag('retrieval')} 记忆内容过长 ({len(insert_content)} 字符)，"
                f"保留开头和结尾并压缩至 {_MAX_CONTENT_CHARS} 字符"
            )
            insert_content = self._fit_content_for_embedding(
                insert_content,
                _MAX_CONTENT_CHARS,
            )
        doc_id = await self.faiss_db.insert(content=insert_content, metadata=metadata)

        return doc_id

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[VectorResult]:
        """
        执行向量相似度搜索

        Args:
            query: 查询字符串
            k: 返回的结果数量
            session_id: 会话ID过滤(可选)
            persona_id: 人格ID过滤(可选)

        Returns:
            List[VectorResult]: 向量检索结果,按相似度降序排列
        """
        if not query or not query.strip():
            return []

        processed_query = query

        # 防止 embedding API token 超限：截断过长的查询文本
        # 大多数 embedding 模型限制在 8192 tokens 以内，按字符数保守截断
        _MAX_QUERY_CHARS = 2000
        if len(processed_query) > _MAX_QUERY_CHARS:
            logger.warning(
                f"{tag('retrieval')} 查询文本过长 ({len(processed_query)} 字符)，"
                f"截断至 {_MAX_QUERY_CHARS} 字符以避免 token 超限"
            )
            processed_query = processed_query[:_MAX_QUERY_CHARS]

        # 构建元数据过滤器
        metadata_filters = {}
        if session_id is not None:
            metadata_filters["session_id"] = session_id
        if persona_id is not None:
            metadata_filters["persona_id"] = persona_id

        return await self._search_with_expansion(
            processed_query, k, metadata_filters
        )

    async def _search_with_expansion(
        self,
        query: str,
        k: int,
        metadata_filters: dict[str, Any],
    ) -> list[VectorResult]:
        """一次 embedding + 按需扩容候选池，凑够 k 条再返回。

        AstrBot 的 ``FaissVecDB.retrieve()`` 是"先取 fetch_k 条全局最近邻、再按
        metadata 过滤"，过滤后可能远少于 k 条（核心注释即写明 "maybe the size is
        less than k"），而且**每次调用都会重新 embedding**。所以这里自己做：
        只 embedding 一次，再按需扩大 FAISS 取数并重新过滤，直到凑够 k 条、
        索引取空或触到上限。

        核心对象结构不符合预期时退回 ``retrieve()``，保持向后兼容。
        """
        embedder = getattr(self.faiss_db, "embedding_provider", None)
        storage = getattr(self.faiss_db, "embedding_storage", None)
        documents = getattr(self.faiss_db, "document_storage", None)
        if not (
            embedder is not None
            and hasattr(embedder, "get_embedding")
            and storage is not None
            and hasattr(storage, "search")
            and documents is not None
            and hasattr(documents, "get_documents")
        ):
            return await self._search_via_retrieve(query, k, metadata_filters)

        try:
            vector = await embedder.get_embedding(query)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"{tag('retrieval')} 查询向量生成失败: {exc}")
            return []

        try:
            import numpy as np

            np_vector = np.array(vector, dtype="float32")
        except Exception as exc:
            logger.warning(f"{tag('retrieval')} 查询向量转换失败: {exc}")
            return await self._search_via_retrieve(query, k, metadata_filters)

        fetch_k = max(k * self._FETCH_MULTIPLIER, self._MIN_FETCH_K)
        results: list[VectorResult] = []
        while True:
            try:
                scores, indices = await self._faiss_search(
                    storage, np_vector, fetch_k
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"{tag('retrieval')} FAISS 检索失败: {exc}")
                return await self._search_via_retrieve(query, k, metadata_filters)

            candidate_ids = [
                int(item) for item in list(indices[0]) if int(item) != -1
            ]
            results = await self._load_candidates(
                candidate_ids, scores[0], metadata_filters
            )
            exhausted = len(candidate_ids) < fetch_k
            if len(results) >= k or exhausted or fetch_k >= self._MAX_FETCH_K:
                break
            fetch_k = min(fetch_k * self._EXPANSION_FACTOR, self._MAX_FETCH_K)

        logger.debug(
            f"{tag('retrieval')} 向量候选 {fetch_k} 条 → 过滤后 {len(results)} 条"
            f"（请求 {k} 条，过滤条件 {len(metadata_filters)} 项）"
        )
        return results

    async def _faiss_search(self, storage: Any, np_vector: Any, fetch_k: int):
        """执行 FAISS 检索，尽量把同步计算移出事件循环。

        AstrBot 的 ``EmbeddingStorage.search`` 是 ``async def`` 里直接调用同步
        ``index.search``：IndexFlatL2 在 1024 维、数万条规模下是几十毫秒级的纯计算，
        会整段阻塞事件循环（其他会话的消息处理一起被卡）。能拿到 ``index`` 与
        ``dimension`` 时改走线程池；拿不到就退回原调用，行为不变。
        """
        index = getattr(storage, "index", None)
        dimension = getattr(storage, "dimension", None)
        if index is None or not isinstance(dimension, int):
            return await storage.search(vector=np_vector, k=fetch_k)

        import numpy as np

        vector = np.asarray(np_vector, dtype=np.float32).ravel()
        if vector.shape[0] != dimension:
            raise ValueError(
                f"向量维度不匹配, 期望: {dimension}, 实际: {vector.shape[0]}"
            )
        return await asyncio.to_thread(
            lambda: index.search(vector.reshape(1, -1), fetch_k)
        )

    async def _load_candidates(
        self,
        candidate_ids: list[int],
        scores,
        metadata_filters: dict[str, Any],
    ) -> list[VectorResult]:
        """按 id 取回文档并按 metadata 过滤，保持相似度降序。"""
        if not candidate_ids:
            return []

        score_by_id = {
            doc_id: float(score)
            for doc_id, score in zip(candidate_ids, list(scores))
        }
        # 注意 core 的 get_documents 默认 limit=100，必须显式放开
        docs = await self.faiss_db.document_storage.get_documents(
            metadata_filters=metadata_filters,
            ids=candidate_ids,
            limit=len(candidate_ids),
        )
        loaded = {
            int(doc["id"]): doc for doc in docs if doc.get("id") is not None
        }
        # FaissVecDB 把 L2 距离归一化成相似度：1 - distance/2
        results = [
            VectorResult(
                doc_id=doc_id,
                score=max(0.0, min(1.0, 1.0 - score_by_id[doc_id] / 2.0)),
                content=loaded[doc_id].get("text", "") or "",
                metadata=self._coerce_metadata(loaded[doc_id].get("metadata")),
            )
            for doc_id in candidate_ids
            if doc_id in loaded
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return results

    @staticmethod
    def _coerce_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    async def _search_via_retrieve(
        self,
        query: str,
        k: int,
        metadata_filters: dict[str, Any],
    ) -> list[VectorResult]:
        """退回 AstrBot 自带的 retrieve()（核心结构不可用时的兼容路径）。"""
        fetch_k = max(k * self._FETCH_MULTIPLIER, self._MIN_FETCH_K)
        if not metadata_filters:
            fetch_k = k
        try:
            faiss_results = await self.faiss_db.retrieve(
                query=query,
                k=k,
                fetch_k=fetch_k,
                rerank=False,
                metadata_filters=metadata_filters or None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"{tag('retrieval')} 向量检索失败: {exc}")
            return []

        results = []
        for result in faiss_results:
            doc_data = result.data
            results.append(
                VectorResult(
                    doc_id=doc_data["id"],
                    score=result.similarity,
                    content=doc_data["text"],
                    metadata=self._coerce_metadata(doc_data.get("metadata")),
                )
            )
        return results

    async def _get_uuid_from_id(self, doc_id: int) -> str | None:
        """
        获取文档的UUID（带缓存优化）

        Args:
            doc_id: 整数文档ID

        Returns:
            Optional[str]: UUID字符串，如果不存在返回None
        """
        # 优化3: 先查缓存
        if doc_id in self._id_cache:
            return self._id_cache[doc_id]

        from ...log import logger, tag

        try:
            doc_storage = self.faiss_db.document_storage
            docs = await doc_storage.get_documents(
                metadata_filters={}, ids=[doc_id], limit=1
            )

            if not docs or len(docs) == 0:
                return None

            uuid_doc_id = docs[0].get("doc_id")

            # 更新缓存
            if uuid_doc_id and len(self._id_cache) < self._cache_max_size:
                self._id_cache[doc_id] = uuid_doc_id

            return uuid_doc_id

        except Exception as e:
            logger.error(f"{tag('retrieval')} [UUID查询] 失败 (doc_id={doc_id}): {e}")
            return None

    async def update_metadata(self, doc_id: int, metadata: dict[str, Any]) -> bool:
        """
        更新文档元数据（使用ORM方式）

        Args:
            doc_id: 文档ID (整数 id)
            metadata: 新的元数据字典

        Returns:
            bool: 是否成功更新
        """
        import json

        from ...log import logger, tag

        try:
            doc_storage = self.faiss_db.document_storage

            # 通过 id 获取文档
            docs = await doc_storage.get_documents(
                metadata_filters={}, ids=[doc_id], limit=1
            )

            if not docs or len(docs) == 0:
                logger.warning(f"{tag('retrieval')} [元数据更新] 文档不存在 (doc_id={doc_id})")
                return False

            doc = docs[0]

            # 获取当前元数据并更新
            current_metadata_str = doc.get("metadata", "{}")
            if isinstance(current_metadata_str, str):
                try:
                    current_metadata = json.loads(current_metadata_str)
                except (json.JSONDecodeError, TypeError):
                    current_metadata = {}
            else:
                current_metadata = current_metadata_str or {}

            # 合并新元数据
            current_metadata.update(metadata)

            # 优化2: 使用参数化查询确保SQL安全
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text

                # 使用参数化查询，避免SQL注入
                stmt = text("UPDATE documents SET metadata = :metadata WHERE id = :id")
                await session.execute(
                    stmt,
                    {
                        "metadata": json.dumps(current_metadata, ensure_ascii=False),
                        "id": doc_id,
                    },
                )

            logger.debug(f"{tag('retrieval')} [元数据更新] 成功 (doc_id={doc_id})")
            return True

        except Exception as e:
            from ...log import logger, tag

            logger.error(f"{tag('retrieval')} [元数据更新] 失败 (doc_id={doc_id}): {e}", exc_info=True)
            return False

    async def delete_document(self, doc_id: int) -> bool:
        """
        删除文档（修复版：正确使用 FaissVecDB.delete API + 缓存优化）

        Args:
            doc_id: 文档ID (documents表中的整数id)

        Returns:
            bool: 是否成功删除
        """
        from ...log import logger, tag

        try:
            # 优化3: 使用缓存的UUID查询方法
            uuid_doc_id = await self._get_uuid_from_id(doc_id)

            if not uuid_doc_id:
                logger.warning(f"{tag('retrieval')} [向量删除] 文档不存在或缺少UUID (doc_id={doc_id})")
                return False

            # 使用 UUID 调用 FaissVecDB.delete()
            # 这会同时删除 document_storage 和 embedding_storage
            await self.faiss_db.delete(uuid_doc_id)

            # 从缓存中移除
            self._id_cache.pop(doc_id, None)

            logger.debug(f"{tag('retrieval')} [向量删除] 成功删除 (doc_id={doc_id}, uuid={uuid_doc_id})")
            return True

        except Exception as e:
            from ...log import logger, tag

            logger.error(f"{tag('retrieval')} [向量删除] 失败 (doc_id={doc_id}): {e}", exc_info=True)
            return False

    async def delete_documents(self, doc_ids: list[int]) -> list[int]:
        """Delete multiple vector documents with one index save when supported."""
        deleted_ids = await delete_faiss_documents_by_ids(self.faiss_db, doc_ids)
        if deleted_ids is None:
            deleted_ids = []
            for doc_id in dict.fromkeys(doc_ids):
                if await self.delete_document(doc_id):
                    deleted_ids.append(doc_id)
        if len(deleted_ids) != len(set(doc_ids)):
            missing = sorted(set(doc_ids) - set(deleted_ids))
            raise RuntimeError(f"批量向量删除未找到文档: {missing}")

        for doc_id in deleted_ids:
            self._id_cache.pop(doc_id, None)
        return deleted_ids
