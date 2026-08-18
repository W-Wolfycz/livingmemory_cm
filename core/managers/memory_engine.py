"""
统一记忆引擎 - MemoryEngine
提供统一的记忆管理接口,整合所有底层组件

存储架构：
- 文档路：FAISS 向量索引（DocumentStorage 元数据）+ 加权 + MMR
- 图路（可选）：graph_store（节点/边）+ graph_vector_db + 内部 RRF 融合
"""

import asyncio
from pathlib import Path
from typing import Any

import aiosqlite

from ...storage.atom_store import AtomStore
from ...storage.graph_store import GraphStore
from ..memory import (
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
from ..managers.atom_lifecycle_manager import AtomLifecycleManager
from ..managers.graph_memory_manager import GraphMemoryManager
from ..processors.graph_extractor import GraphExtractor
from ..processors.text_processor import TextProcessor
from ..retrieval.dual_route_retriever import DualRouteRetriever
from ..retrieval.graph_keyword_retriever import GraphKeywordRetriever
from ..retrieval.graph_retriever import GraphRetriever
from ..retrieval.graph_vector_retriever import GraphVectorRetriever
from ..retrieval.hybrid_retriever import HybridResult, HybridRetriever
from ..retrieval.rrf_fusion import RRFFusion
from ..retrieval.vector_retriever import VectorRetriever


class MemoryEngine:
    """
    统一记忆引擎

    整合向量检索（文档路）和图检索（可选），提供完整的记忆管理接口。

    主要功能:
    1. 记忆CRUD操作(添加、检索、更新、删除)
    2. 自动化记忆整理和清理
    3. 重要性评估和时间衰减
    4. 会话隔离和统计

    ID管理体系说明：
    ==================
    1. **DocumentStorage (FAISS内部)**
       - 表: documents (SQLite，由SQLAlchemy管理)
       - 主键: id (INTEGER, AUTOINCREMENT) - 统一的整数标识符
       - UUID字段: doc_id (TEXT) - FAISS内部使用的UUID字符串
       - 关系: id ←→ doc_id (一对一映射)

    2. **FAISS向量索引**
       - 存储: EmbeddingStorage (FAISS索引文件)
       - 索引ID: 使用documents.id作为向量的整数索引

    插件对外接口：
    - add_memory() 返回: int (documents.id)
    - search_memories() 返回: HybridResult包含doc_id (int)
    - update_memory(memory_id: int) 参数: documents.id
    - delete_memory(memory_id: int) 参数: documents.id
    """

    def __init__(
        self,
        db_path: str,
        faiss_db,
        graph_vector_db=None,
        llm_provider=None,
        config: dict[str, Any] | None = None,
    ):
        """
        初始化记忆引擎

        Args:
            db_path: SQLite数据库路径
            faiss_db: FAISS向量数据库实例
            llm_provider: LLM提供者(可选,用于高级功能)
            config: 配置字典,支持以下参数:
                - decay_rate: 时间衰减率,默认0.01
                - cleanup_days_threshold: 清理天数阈值,默认30
                - cleanup_importance_threshold: 清理重要性阈值,默认0.3
                - stopwords_path: 停用词文件路径(可选)
        """
        self.db_path = db_path
        self.faiss_db = faiss_db
        self.graph_vector_db = graph_vector_db
        self.llm_provider = llm_provider
        self.config = config or {}
        self.graph_enabled = bool(self.config.get("graph_memory_enabled", False))
        self.atom_enabled = bool(
            self.config.get(
                "atom_enabled",
                self.config.get("graph_memory_atom_enabled", True),
            )
        )

        # 确保数据库目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 后台任务跟踪
        self._pending_tasks: set[asyncio.Task] = set()

        # 初始化组件(在initialize中完成)
        self.text_processor = None
        self.vector_retriever = None
        self.hybrid_retriever = None
        self.graph_store = None
        self.graph_extractor = None
        self.graph_keyword_retriever = None
        self.graph_vector_retriever = None
        self.graph_retriever = None
        self.graph_memory_manager = None
        self.dual_route_retriever = None
        self.atom_store = None
        self.atom_lifecycle_manager = None
        self.db_connection = None
        self._document_repository = DocumentRepository(self.faiss_db)
        self._search_service = MemorySearchService(self.config)
        self._statistics_service = MemoryStatisticsService(
            self.db_path,
            self.faiss_db,
        )
        self._lifecycle_service = MemoryLifecycleService()
        self._repair_service = MemoryRepairService()
        self._schema_service = MemorySchemaService()
        self._write_coordinator = MemoryWriteCoordinator()
        self._write_op_repair_enabled = bool(
            self.config.get("write_op_repair_enabled", True)
        )
        self._write_op_max_retries = int(self.config.get("write_op_max_retries", 3))

    async def initialize(self):
        """
        异步初始化引擎

        创建数据库表、初始化所有检索器组件
        """
        # 1. 连接数据库
        self.db_connection = await aiosqlite.connect(self.db_path)
        self.db_connection.row_factory = aiosqlite.Row
        await self.db_connection.execute("PRAGMA journal_mode = WAL")
        await self.db_connection.execute("PRAGMA busy_timeout = 10000")

        # 2. 创建表结构
        await self._create_tables()

        # 3. 初始化文本处理器
        stopwords_path = self.config.get("stopwords_path")
        self.text_processor = TextProcessor(stopwords_path)

        # 4. 初始化向量检索器
        self.vector_retriever = VectorRetriever(self.faiss_db, self.config)

        # 5. 初始化文档路检索器（纯向量 + 加权 + MMR）
        self.hybrid_retriever = HybridRetriever(
            self.vector_retriever, self.config
        )

        if self.graph_enabled and self.graph_vector_db is not None:
            self.graph_store = GraphStore(self.db_path)
            await self.graph_store.initialize()

            self.atom_store = AtomStore(self.db_path)
            await self.atom_store.initialize()

            if self.atom_enabled:
                self.atom_lifecycle_manager = AtomLifecycleManager(
                    self.atom_store, self.config
                )
                await self.atom_lifecycle_manager.start()

            self.graph_extractor = GraphExtractor(self.config)
            self.graph_keyword_retriever = GraphKeywordRetriever(
                self.graph_store,
                self.text_processor,
                self.config,
            )
            self.graph_vector_retriever = GraphVectorRetriever(
                self.graph_vector_db,
                self.config,
            )
            # 图路内部使用 RRF 融合 keyword + vector 两路结果
            graph_rrf = RRFFusion(k=self.config.get("rrf_k", 60))
            self.graph_retriever = GraphRetriever(
                self.graph_keyword_retriever,
                self.graph_vector_retriever,
                graph_rrf,
                self.config,
            )
            self.graph_memory_manager = GraphMemoryManager(
                self.graph_store,
                self.graph_vector_retriever,
                self.graph_extractor,
            )
            self.dual_route_retriever = DualRouteRetriever(
                self.hybrid_retriever,
                self.graph_retriever,
                self.get_memory,
                self.config,
            )

        if self._write_op_repair_enabled:
            await self._repair_incomplete_write_ops()

    async def close(self):
        """关闭数据库连接和清理资源"""
        if self.atom_lifecycle_manager is not None:
            await self.atom_lifecycle_manager.stop()
        if self._pending_tasks:
            for task in self._pending_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()
        if self.db_connection:
            await self.db_connection.close()
        if self.graph_vector_db is not None:
            await self.graph_vector_db.close()

    def _create_tracked_task(self, coro) -> None:
        """Create and track a background task, auto-discarding on completion."""
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _create_write_ops_table(self) -> None:
        """Compatibility facade for the resumable write-operation log."""
        await self._write_coordinator.create_write_ops_table(
            self._build_write_context()
        )

    async def _start_write_op(
        self,
        op_type: str,
        payload: dict[str, Any] | None = None,
        memory_id: int | None = None,
    ) -> int | None:
        """Compatibility facade for starting a write operation."""
        return await self._write_coordinator.start_write_op(
            self._build_write_context(),
            op_type,
            payload,
            memory_id,
        )

    async def _advance_write_op(
        self,
        op_id: int | None,
        step: str,
        *,
        status: str = "pending",
        memory_id: int | None = None,
        error: str | None = None,
        payload_patch: dict[str, Any] | None = None,
    ) -> None:
        """Compatibility facade for advancing a write operation."""
        await self._write_coordinator.advance_write_op(
            self._build_write_context(),
            op_id,
            step,
            status=status,
            memory_id=memory_id,
            error=error,
            payload_patch=payload_patch,
        )

    def _invalidate_search_cache(self) -> None:
        """Invalidate cached retrieval results after memory writes."""
        self._search_service.invalidate()

    def _build_write_context(self) -> MemoryWriteContext:
        """Build a fresh write context from the engine's live components."""
        return MemoryWriteContext(
            db_connection=self.db_connection,
            faiss_db=self.faiss_db,
            hybrid_retriever=self.hybrid_retriever,
            atom_store=self.atom_store,
            graph_memory_manager=self.graph_memory_manager,
            atom_enabled=self.atom_enabled,
            get_memory=self.get_memory,
            find_memory_by_idempotency_key=self._find_memory_by_idempotency_key,
            add_memory=self.add_memory,
            delete_memory=self.delete_memory,
            start_write_op=self._start_write_op,
            advance_write_op=self._advance_write_op,
            serialize_atom_for_repair=self._serialize_atom_for_repair,
            delete_graph_and_atoms_for_batch=self._delete_graph_and_atoms_for_batch,
            invalidate_search_cache=self._invalidate_search_cache,
        )

    def _build_lifecycle_context(self) -> MemoryLifecycleContext:
        """Build a fresh lifecycle context from live engine components."""
        return MemoryLifecycleContext(
            db_connection=self.db_connection,
            faiss_db=self.faiss_db,
            graph_memory_manager=self.graph_memory_manager,
            document_repository=self._document_repository,
            config=self.config,
            batch_delete_memories=self.batch_delete_memories,
            invalidate_search_cache=self._invalidate_search_cache,
        )

    def _build_repair_context(self) -> MemoryRepairContext:
        """Build a fresh repair context from the engine's live components."""
        return MemoryRepairContext(
            db_connection=self.db_connection,
            faiss_db=self.faiss_db,
            atom_store=self.atom_store,
            graph_memory_manager=self.graph_memory_manager,
            atom_enabled=self.atom_enabled,
            max_retries=self._write_op_max_retries,
            get_memory=self.get_memory,
            advance_write_op=self._advance_write_op,
            invalidate_search_cache=self._invalidate_search_cache,
        )

    def _build_schema_context(self) -> MemorySchemaContext:
        """Build a fresh schema context from the live database connection."""
        return MemorySchemaContext(
            db_connection=self.db_connection,
            create_write_ops_table=self._create_write_ops_table,
        )

    def _serialize_atom_for_repair(self, atom: Any) -> dict[str, Any]:
        """Compatibility facade for repair payload serialization."""
        return self._repair_service.serialize_atom(atom)

    def _deserialize_atom_from_repair(
        self,
        payload: dict[str, Any],
        parent_memory_id: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> Any:
        """Compatibility facade for rebuilding a repair atom."""
        return self._repair_service.deserialize_atom(
            payload,
            parent_memory_id,
            session_id,
            persona_id,
        )

    async def _repair_incomplete_write_ops(self) -> int:
        """Compatibility facade for incomplete write-operation replay."""
        return await self._repair_service.repair_incomplete_write_ops(
            self._build_repair_context()
        )

    async def _repair_add_write_op(
        self,
        op_id: int,
        memory_id: int | None,
        payload: dict[str, Any],
    ) -> bool:
        return await self._repair_service.repair_add_write_op(
            self._build_repair_context(),
            op_id,
            memory_id,
            payload,
        )

    async def _repair_delete_write_op(
        self,
        op_id: int,
        memory_id: int | None,
    ) -> bool:
        return await self._repair_service.repair_delete_write_op(
            self._build_repair_context(),
            op_id,
            memory_id,
        )

    async def _repair_batch_delete_write_op(
        self,
        op_id: int,
        payload: dict[str, Any],
    ) -> bool:
        return await self._repair_service.repair_batch_delete_write_op(
            self._build_repair_context(),
            op_id,
            payload,
        )

    async def _delete_document_indexes_for_batch(self, memory_ids: list[int]) -> int:
        return await self._repair_service.delete_document_indexes_for_batch(
            self._build_repair_context(),
            memory_ids,
        )

    async def _delete_graph_and_atoms_for_batch(self, memory_ids: list[int]) -> None:
        await self._repair_service.delete_graph_and_atoms_for_batch(
            self._build_repair_context(),
            memory_ids,
        )

    @staticmethod
    def _safe_json_dict(value: Any) -> dict[str, Any]:
        return MemoryRepairService.safe_json_dict(value)

    async def _create_tables(self):
        """创建数据库表

        注意：documents 表主要由 FAISS 的 DocumentStorage 类创建和管理。
        这里使用 CREATE TABLE IF NOT EXISTS 确保兼容性：
        - 如果 FAISS 已创建，不会重复创建（IF NOT EXISTS）
        - 如果 FAISS 未创建（极端情况），插件仍能正常工作
        - 插件需要直接操作此表进行高频更新（如访问时间）
        """
        await self._schema_service.create_tables(self._build_schema_context())

    async def _drop_legacy_documents_fts_triggers(self):
        await self._schema_service.drop_legacy_documents_fts_triggers(
            self._build_schema_context()
        )

    # ==================== 核心记忆操作 ====================

    async def add_memory(
        self,
        content: str,
        session_id: str | None = None,
        persona_id: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        atoms: list | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        """
        添加新记忆

        Args:
            content: 记忆内容
            session_id: 会话ID(支持多种格式,自动提取UUID)
            persona_id: 人格ID(支持多种格式,自动提取UUID)
            importance: 重要性(0-1)
            metadata: 额外元数据
            idempotency_key: 可选幂等键；已存在时直接返回原记忆 ID

        Returns:
            int: 记忆ID(doc_id)
        """
        return await self._write_coordinator.add_memory(
            self._build_write_context(),
            content=content,
            session_id=session_id,
            persona_id=persona_id,
            importance=importance,
            metadata=metadata,
            atoms=atoms,
            idempotency_key=idempotency_key,
        )

    async def _find_memory_by_idempotency_key(
        self, idempotency_key: str
    ) -> int | None:
        """通过 FAISS 共享文档存储查询幂等键，不扫描记忆正文。"""
        return await self._document_repository.find_by_idempotency_key(
            idempotency_key
        )

    async def search_memories(
        self,
        query: str,
        k: int = 5,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[HybridResult]:
        """
        检索相关记忆

        Args:
            query: 查询字符串
            k: 返回数量
            session_id: 会话ID过滤(可选,应传入unified_msg_origin完整格式)
            persona_id: 人格ID过滤(可选)

        Returns:
            List[HybridResult]: 检索结果列表
        """
        return await self._search_service.search(
            query=query,
            k=k,
            session_id=session_id,
            persona_id=persona_id,
            hybrid_retriever=self.hybrid_retriever,
            dual_route_retriever=self.dual_route_retriever,
            schedule_task=self._create_tracked_task,
            update_access_time=self._update_access_time_internal,
            migrate_session=self._migrate_session_data_if_needed,
            db_connection=self.db_connection,
        )

    async def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        """
        根据ID获取记忆

        Args:
            memory_id: 记忆ID

        Returns:
            Optional[Dict]: 记忆数据,包含text和metadata
        """
        return await self._document_repository.get_memory(memory_id)

    async def update_memory(
        self,
        memory_id: int,
        updates: dict[str, Any],
    ) -> bool:
        """
        更新记忆（确保多数据库同步）

        支持更新内容、重要性、元数据等。采用不同策略：
        - 内容更新：先创建后删除（避免数据丢失）+ 全库同步
        - 元数据更新：三库同步更新

        Args:
            memory_id: 记忆ID
            updates: 更新字典,可包含:
                - content: 新内容 (触发完整重建)
                - importance: 新重要性
                - metadata: 元数据更新

        Returns:
            bool: 是否更新成功
        """
        return await self._write_coordinator.update_memory(
            self._build_write_context(),
            memory_id,
            updates,
        )

    async def delete_memory(self, memory_id: int) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否删除成功
        """
        return await self._write_coordinator.delete_memory(
            self._build_write_context(),
            memory_id,
        )

    async def rebuild_graph_index(self) -> dict[str, int]:
        """Rebuild graph-memory artifacts from stored documents."""
        return await self._lifecycle_service.rebuild_graph_index(
            self._build_lifecycle_context()
        )

    async def get_graph_index_status(self) -> dict[str, Any]:
        """Return redacted graph-vector migration and consistency counts."""
        if (
            not self.graph_enabled
            or self.graph_vector_db is None
            or self.graph_store is None
        ):
            return {"state": "disabled"}

        document_storage = self.graph_vector_db.document_storage
        total_vectors = int(
            await document_storage.count_documents(metadata_filters={})
        )
        memory_vectors = int(
            await document_storage.count_documents(
                metadata_filters={"graph_vector_granularity": "memory"}
            )
        )
        legacy_vectors = max(0, total_vectors - memory_vectors)
        vectors_by_source = await self.graph_store.list_vector_doc_ids_by_source()
        source_memories = await self.graph_store.count_source_memories()
        referenced_sources = len(vectors_by_source)
        referenced_vectors = len(
            {
                int(vector_doc_id)
                for vector_doc_ids in vectors_by_source.values()
                for vector_doc_id in vector_doc_ids
            }
        )
        orphan_vectors = max(0, total_vectors - referenced_vectors)

        rebuild_state = {"state": "idle", "active": False}
        if self.graph_memory_manager is not None:
            get_rebuild_status = getattr(
                self.graph_memory_manager,
                "get_rebuild_status",
                None,
            )
            if callable(get_rebuild_status):
                rebuild_state = get_rebuild_status()

        last_state = str(rebuild_state.get("state") or "idle")
        missing_source_vectors = max(0, source_memories - referenced_sources)
        inconsistent = (
            referenced_vectors != total_vectors or missing_source_vectors > 0
        )
        if rebuild_state.get("active"):
            state = "rebuilding"
        elif last_state in {"failed", "cancelled"}:
            state = last_state
        elif legacy_vectors > 0 or inconsistent:
            state = "rebuild_required"
        elif last_state == "switched":
            state = "switched"
        else:
            state = "current"

        return {
            "state": state,
            "total_vectors": total_vectors,
            "memory_vectors": memory_vectors,
            "legacy_vectors": legacy_vectors,
            "source_memories": source_memories,
            "referenced_sources": referenced_sources,
            "referenced_vectors": referenced_vectors,
            "orphan_vectors": orphan_vectors,
            "missing_source_vectors": missing_source_vectors,
            "error_type": rebuild_state.get("error_type"),
            "result": rebuild_state.get("result") or {},
        }

    # ==================== 高级功能 ====================

    async def update_importance(self, memory_id: int, new_importance: float) -> bool:
        """
        更新记忆重要性

        Args:
            memory_id: 记忆ID
            new_importance: 新重要性值(0-1)

        Returns:
            bool: 是否更新成功
        """
        return await self.update_memory(memory_id, {"importance": new_importance})

    async def apply_daily_decay(self, decay_rate: float, days: int = 1) -> int:
        """
        批量应用重要性衰减

        Args:
            decay_rate: 每日衰减率 (0-1)
            days: 衰减天数（用于补偿错过的天数）

        Returns:
            int: 受影响的记忆数量
        """
        return await self._lifecycle_service.apply_daily_decay(
            self._build_lifecycle_context(),
            decay_rate,
            days,
        )

    async def update_access_time(self, memory_id: int) -> bool:
        """
        更新最后访问时间

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否更新成功
        """
        return await self._update_access_time_internal(memory_id)

    async def _update_access_time_internal(self, memory_id: int) -> bool:
        """内部方法:更新访问时间（直接更新documents表，不经过FAISS）"""
        return await self._lifecycle_service.update_access_time(
            self._build_lifecycle_context(),
            memory_id,
        )

    async def get_session_memories(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        获取会话的所有记忆（使用分批处理和数据库排序优化）

        Args:
            session_id: 会话ID(应传入完整的unified_msg_origin格式)
            limit: 限制数量

        Returns:
            List[Dict]: 记忆列表
        """
        return await self._document_repository.get_session_memories(
            session_id,
            limit,
        )

    async def batch_delete_memories(self, memory_ids: list[int]) -> int:
        """Batch delete multiple memories using bulk SQL operations."""
        return await self._write_coordinator.batch_delete_memories(
            self._build_write_context(),
            memory_ids,
        )

    async def cleanup_old_memories(
        self,
        days_threshold: int | None = None,
        importance_threshold: float | None = None,
    ) -> int:
        """
        清理旧记忆（使用分批处理避免内存问题）

        删除超过阈值且重要性低的记忆

        Args:
            days_threshold: 天数阈值,默认从配置读取
            importance_threshold: 重要性阈值,默认从配置读取

        Returns:
            int: 删除的记忆数量
        """
        return await self._lifecycle_service.cleanup_old_memories(
            self._build_lifecycle_context(),
            days_threshold,
            importance_threshold,
        )

    async def _migrate_session_data_if_needed(self, unified_msg_origin: str) -> None:
        """
        运行时自动迁移：将旧格式的session_id更新为unified_msg_origin格式

        支持各种平台的旧格式（通用匹配策略）：
        - WebChat UUID: "ac8c2cef-959e-4146-ad22-c82d0230ad06"
        - WebChat带前缀: "webchat!astrbot!ac8c2cef-959e-4146-ad22-c82d0230ad06"
        - QQ号: "10001"
        - 其他平台: 任意字符串

        目标格式: "platform:message_type:session_id"

        策略：
        1. 从unified_msg_origin解析出：platform、message_type、session_id
        2. 生成所有可能的旧格式匹配候选（递归拆分）
        3. 查找匹配任一候选且不含冒号的旧记录
        4. 批量更新为unified_msg_origin
        5. 使用unified_msg_origin本身作为迁移标记（避免重复）

        Args:
            unified_msg_origin: 完整的统一消息来源（格式：platform:type:session_id）
        """
        await self._lifecycle_service.migrate_session_data_if_needed(
            self._build_lifecycle_context(),
            unified_msg_origin,
        )

    async def get_statistics(self, persona_id: str | None = None) -> dict[str, Any]:
        """
        获取记忆统计信息（使用批量处理避免内存问题）

        Args:
            persona_id: 人格ID过滤（可选）。None=全局聚合。

        Returns:
            Dict: 统计信息,包含:
                - total_memories: 总记忆数
                - sessions: 各会话的记忆数（按UUID分组）
                - status_breakdown: 各状态的记忆数
                - avg_importance: 平均重要性
                - oldest_memory: 最旧记忆时间
                - newest_memory: 最新记忆时间
        """
        statistics = await self._statistics_service.get_statistics(
            persona_id,
            graph_store=self.graph_store,
        )
        statistics["graph_index"] = await self.get_graph_index_status()
        return statistics

    async def maintain_storage(self, *, vacuum: bool = False) -> dict[str, Any]:
        """Run SQLite storage maintenance and return size diagnostics."""
        return await self._statistics_service.maintain_storage(
            self.db_connection,
            vacuum=vacuum,
        )
