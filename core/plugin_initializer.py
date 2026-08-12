"""
插件初始化器
负责插件的初始化逻辑
"""

import asyncio
import time
from pathlib import Path
from typing import Any

from ..log import logger, tag
from astrbot.api.star import Context
from astrbot.core.provider.provider import EmbeddingProvider, Provider

from ..storage.conversation_store import ConversationStore
from ..storage.db_migration import DBMigration
from .base.config_manager import ConfigManager
from .base.exceptions import InitializationError, ProviderNotReadyError
from .bootstrap import (
    EmbeddingIndexBootstrapService,
    EmbeddingIndexSpec,
    FaissBootstrapService,
)
from .managers.conversation_manager import ConversationManager
from .managers.memory_engine import MemoryEngine
from .processors.memory_processor import MemoryProcessor
from .schedulers.decay_scheduler import DecayScheduler

FaissVecDB: Any = None

_FAISS_BOOTSTRAP_COMPAT = FaissBootstrapService()

# ── Faiss C++ fopen() 在 Windows 上使用 ANSI codepage ──
# Python 传给 Faiss 的路径是 UTF-8 字节，Windows fopen 期望 ANSI 编码，
# 含非 ASCII 字符的路径（如 C:\Users\<中文名>\...）被解读为乱码 →
# RuntimeError: could not open ... for reading: No such file or directory。
# 通过 monkey-patch faiss.read_index / write_index，经纯 ASCII 临时文件桥接。


def _needs_bridge(path: str) -> bool:
    """判断是否需要 ASCII 临时文件桥接。"""
    return _FAISS_BOOTSTRAP_COMPAT.needs_bridge(path)


def _safe_temp_dir() -> str:
    """返回保证纯 ASCII 且可写的临时目录。"""
    return _FAISS_BOOTSTRAP_COMPAT.safe_temp_dir()


def _make_temp_file(prefix: str) -> str:
    """创建 Faiss 桥接临时文件，返回纯 ASCII 路径。"""
    return _FAISS_BOOTSTRAP_COMPAT.make_temp_file(prefix)


def _sanitize_path(path: str) -> str:
    """脱敏路径：非 ASCII 部分替换为 [***]，避免日志泄露中文用户名。"""
    return _FAISS_BOOTSTRAP_COMPAT.sanitize_path(path)


class PluginInitializer:
    """插件初始化器"""

    def __init__(self, context: Context, config_manager: ConfigManager, data_dir: str):
        """
        初始化插件初始化器

        Args:
            context: AstrBot上下文
            config_manager: 配置管理器
            data_dir: 插件数据目录路径
        """
        self.context = context
        self.config_manager = config_manager
        self.data_dir = data_dir

        # 组件实例
        self.embedding_provider: EmbeddingProvider | None = None
        self.llm_provider: Provider | None = None
        self.db: Any | None = None
        self.graph_db: Any | None = None
        self.memory_engine: MemoryEngine | None = None
        self.memory_processor: MemoryProcessor | None = None
        self.db_migration: DBMigration | None = None
        self.conversation_manager: ConversationManager | None = None
        self.decay_scheduler: DecayScheduler | None = None

        # 初始化状态
        self._initialization_complete = False
        self._initialization_lock = asyncio.Lock()
        self._initialization_failed = False
        self._initialization_error: str | None = None
        self._providers_ready = False
        self._provider_check_attempts = 0
        self._max_provider_attempts = 60
        self._retry_task: asyncio.Task | None = None
        self._faiss_bootstrap = FaissBootstrapService()
        self._embedding_index_bootstrap: EmbeddingIndexBootstrapService | None = None

    async def initialize(self) -> bool:
        """
        执行初始化

        Returns:
            bool: 是否初始化成功
        """
        async with self._initialization_lock:
            if self._initialization_complete or self._initialization_failed:
                return self._initialization_complete

        logger.info(f"{tag('init')} LivingMemory 插件开始后台初始化...")

        try:
            # 1. 在 Provider 等待前先检查已有数据库，避免旧库在后台重试阶段
            # 才暴露为“插件已加载但核心不可用”的半初始化状态。
            await self._preflight_database_compatibility()

            # 2. 等待 Provider 就绪
            if not await self._wait_for_providers_non_blocking():
                missing = []
                if not self.embedding_provider:
                    missing.append(
                        "Embedding Provider（请在 AstrBot 中配置向量嵌入模型）"
                    )
                if not self.llm_provider:
                    missing.append("LLM Provider（请在 AstrBot 中配置语言模型）")
                logger.warning(
                    f"{tag('init')} 以下 Provider 暂时不可用，将在后台继续尝试: {', '.join(missing)}"
                )
                self._start_retry_task_if_needed()
                return False

            # 3. Provider 就绪，继续完整初始化
            await self._complete_initialization()
            return True

        except Exception as e:
            logger.error(f"{tag('init')} LivingMemory 插件初始化失败: {e}", exc_info=True)
            self._initialization_failed = True
            self._initialization_error = str(e)
            return False

    def _start_retry_task_if_needed(self) -> None:
        """启动后台重试任务（避免重复启动）"""
        if self._retry_task and not self._retry_task.done():
            return

        self._retry_task = asyncio.create_task(self._retry_initialization())
        self._retry_task.add_done_callback(self._on_retry_task_done)

    def _on_retry_task_done(self, task: asyncio.Task) -> None:
        """重试任务完成回调，回收状态并记录异常"""
        self._retry_task = None
        if task.cancelled():
            return
        try:
            exc = task.exception()
            if exc:
                logger.error(f"{tag('init')} Provider 重试任务异常退出: {exc}")
        except Exception:
            # 防御性处理：读取 task.exception() 时不应阻断主流程
            pass

    async def _wait_for_providers_non_blocking(self, max_wait: float = 5.0) -> bool:
        """非阻塞地检查 Provider 是否可用"""
        start_time = time.time()
        check_interval = 1.0

        while time.time() - start_time < max_wait:
            self._initialize_providers(silent=True)

            if self.embedding_provider and self.llm_provider:
                logger.info(
                    f"{tag('init')} Provider check passed: embedding and llm providers are ready."
                )
                self._providers_ready = True
                return True

            await asyncio.sleep(check_interval)
            self._provider_check_attempts += 1

        logger.debug(
            f"{tag('init')} Provider 在 {max_wait}秒内未就绪（已尝试 {self._provider_check_attempts} 次）"
            f"：embedding={'ready' if self.embedding_provider else 'not ready'}, "
            f"llm={'ready' if self.llm_provider else 'not ready'}"
        )
        return False

    async def _retry_initialization(self):
        """后台重试初始化任务（指数退避策略）"""
        base_interval = 2.0
        max_interval = 30.0
        current_interval = base_interval
        log_interval = 5

        while (
            not self._initialization_complete
            and not self._initialization_failed
            and self._provider_check_attempts < self._max_provider_attempts
        ):
            await asyncio.sleep(current_interval)

            self._initialize_providers(silent=True)
            self._provider_check_attempts += 1

            if self._provider_check_attempts % log_interval == 0:
                missing = []
                if not self.embedding_provider:
                    missing.append("Embedding Provider")
                if not self.llm_provider:
                    missing.append("LLM Provider")
                logger.info(
                    f"{tag('init')} 等待 Provider 就绪（未就绪: {', '.join(missing)}）..."
                    f"（已尝试 {self._provider_check_attempts}/{self._max_provider_attempts} 次，"
                    f"下次重试间隔 {current_interval:.1f}s）"
                )

            if self.embedding_provider and self.llm_provider:
                logger.info(
                    f"{tag('init')} Provider 在第 {self._provider_check_attempts} 次尝试后就绪，继续初始化。"
                )
                self._providers_ready = True

                try:
                    async with self._initialization_lock:
                        if not self._initialization_complete:
                            await self._complete_initialization()
                except Exception as e:
                    logger.error(f"{tag('init')} 重试初始化失败: {e}", exc_info=True)
                    self._initialization_failed = True
                    self._initialization_error = str(e)
                break

            # 指数退避，最大30秒
            current_interval = min(current_interval * 1.5, max_interval)

        if not self._initialization_complete and not self._initialization_failed:
            missing = []
            if not self.embedding_provider:
                missing.append("Embedding Provider（请配置向量嵌入模型）")
            if not self.llm_provider:
                missing.append("LLM Provider（请配置语言模型）")
            logger.error(
                f"{tag('init')} 以下 Provider 在 {self._provider_check_attempts} 次尝试后仍未就绪，初始化失败: "
                f"{', '.join(missing) if missing else '未知'}"
            )
            self._initialization_failed = True
            self._initialization_error = (
                "Provider 初始化超时。"
                f"未就绪 Provider: {', '.join(missing) if missing else '未知'}。"
                "请检查 provider_settings 配置和 AstrBot 默认 Provider。"
            )

    def _initialize_providers(self, silent: bool = False):
        """初始化 Embedding 和 LLM provider"""
        # 初始化 Embedding Provider
        emb_id = self.config_manager.get("provider_settings.embedding_provider_id")
        if emb_id:
            provider = self._get_provider_by_id(emb_id, silent=silent)
            if provider and isinstance(provider, EmbeddingProvider):
                self.embedding_provider = provider
                if not silent:
                    logger.info(f"{tag('init')} 成功从配置加载 Embedding Provider: {emb_id}")
            elif provider and not silent:
                logger.warning(f"{tag('init')} Provider {emb_id} 不是 EmbeddingProvider 类型")

        if not self.embedding_provider:
            embedding_providers = self.context.get_all_embedding_providers()
            if embedding_providers:
                self.embedding_provider = embedding_providers[0]
                if not silent:
                    provider_id = getattr(
                        self.embedding_provider.provider_config,
                        "id",
                        self.embedding_provider.provider_config.get("id", "unknown"),
                    )
                    logger.info(f"{tag('init')} 未指定 Embedding Provider，使用默认的: {provider_id}")
            else:
                self.embedding_provider = None
                if not silent:
                    logger.debug(f"{tag('init')} 没有可用的 Embedding Provider")

        # 初始化 LLM Provider
        self.llm_provider = None
        llm_id = self.config_manager.get("provider_settings.llm_provider_id")
        if llm_id:
            provider = self._get_provider_by_id(llm_id, silent=silent)
            if provider and isinstance(provider, Provider):
                self.llm_provider = provider
                if not silent:
                    logger.info(f"{tag('init')} 成功从配置加载 LLM Provider: {llm_id}")
            elif provider and not silent:
                logger.warning(
                    f"{tag('init')} Provider {llm_id} 不是聊天 Provider 类型，已忽略该配置。"
                )

        if not self.llm_provider:
            try:
                if silent and not self.context.get_all_providers():
                    self.llm_provider = None
                    return
                default_provider = self.context.get_using_provider()
                if default_provider and not isinstance(default_provider, Provider):
                    if not silent:
                        logger.warning(
                            f"{tag('init')} AstrBot 默认 Provider 类型不正确，期望聊天 Provider。"
                        )
                    self.llm_provider = None
                else:
                    self.llm_provider = default_provider
                if not silent and self.llm_provider:
                    logger.info(f"{tag('init')} 使用 AstrBot 当前默认的 LLM Provider。")
            except (ValueError, Exception) as e:
                if not silent:
                    logger.debug(f"{tag('init')} 获取默认 LLM Provider 失败: {e}")
                self.llm_provider = None

    def _get_provider_by_id(self, provider_id: str, *, silent: bool):
        """静默检查阶段绕过会打印 warning 的 AstrBot 查询接口。"""
        if not provider_id:
            return None
        if not silent:
            return self.context.get_provider_by_id(provider_id)
        provider_manager = getattr(self.context, "provider_manager", None)
        inst_map = getattr(provider_manager, "inst_map", None)
        if isinstance(inst_map, dict):
            return inst_map.get(provider_id)
        return None

    def _check_faiss_runtime(self) -> None:
        self._get_faiss_bootstrap().check_runtime()

    def _load_faiss_vec_db_class(self):
        global FaissVecDB
        if FaissVecDB is not None:
            return FaissVecDB
        loaded_class = self._get_faiss_bootstrap().load_vec_db_class()
        FaissVecDB = loaded_class
        return loaded_class

    def _get_faiss_bootstrap(self) -> FaissBootstrapService:
        """兼容未经过 __init__ 构造的测试/恢复实例。"""
        service = getattr(self, "_faiss_bootstrap", None)
        if service is None:
            service = FaissBootstrapService()
            self._faiss_bootstrap = service
        return service

    async def _complete_initialization(self):
        """完成完整的初始化流程"""
        if self._initialization_complete:
            return

        logger.info(f"{tag('init')} 开始完整初始化流程...")

        try:
            # 初始化数据库
            data_dir_path = Path(self.data_dir)
            db_path = data_dir_path / "livingmemory.db"
            index_path = data_dir_path / "livingmemory.index"
            graph_doc_path = data_dir_path / "livingmemory_graph_documents.db"
            graph_index_path = data_dir_path / "livingmemory_graph.index"
            graph_memory_enabled = self.config_manager.get("graph_memory.enabled", True)

            # 新安装时 preflight 不创建空数据库；在正式初始化阶段准备迁移管理器。
            if self.db_migration is None:
                self.db_migration = DBMigration(str(db_path))

            if not self.embedding_provider:
                raise ProviderNotReadyError("Embedding Provider 未初始化")
            if not self.llm_provider or not isinstance(self.llm_provider, Provider):
                raise ProviderNotReadyError("LLM Provider 未初始化或类型不正确")

            # 在 FAISS 打开或改写主数据库前再次验证版本，覆盖 Provider 等待期间
            # 数据库文件被替换的极端情况。
            await self._check_and_migrate_database()

            faiss_vec_db_cls = self._load_faiss_vec_db_class()

            # 在 FaissVecDB 打开正式索引前校验 SQLite/FAISS/Provider 一致性。
            # 需要修复时只构建影子索引，完整验证后才原子切换。
            index_specs = [
                EmbeddingIndexSpec("document", db_path, index_path),
            ]
            if graph_memory_enabled:
                index_specs.append(
                    EmbeddingIndexSpec(
                        "graph",
                        graph_doc_path,
                        graph_index_path,
                        vector_schema="graph-entry-v1",
                    )
                )
            await self._prepare_embedding_indexes(index_specs)

            self.db = faiss_vec_db_cls(
                str(db_path),
                str(index_path),
                self.embedding_provider,
            )
            await self.db.initialize()
            self.graph_db = None
            if graph_memory_enabled:
                self.graph_db = faiss_vec_db_cls(
                    str(graph_doc_path),
                    str(graph_index_path),
                    self.embedding_provider,
                )
                await self.graph_db.initialize()
            logger.info(f"{tag('init')} 数据库已初始化。数据目录: {self.data_dir}")

            # 初始化MemoryEngine
            stopwords_dir = data_dir_path / "stopwords"
            stopwords_dir.mkdir(parents=True, exist_ok=True)

            memory_engine_config = {
                "decay_rate": self.config_manager.get(
                    "importance_decay.decay_rate", 0.01
                ),
                "access_decay_window_days": self.config_manager.get(
                    "importance_decay.access_decay_window_days", 30.0
                ),
                "access_decay_max_count": self.config_manager.get(
                    "importance_decay.access_decay_max_count", 10
                ),
                "access_count_decay_multiplier": self.config_manager.get(
                    "importance_decay.access_count_decay_multiplier", 0.5
                ),
                "protected_importance_threshold": self.config_manager.get(
                    "importance_decay.protected_importance_threshold", 1.0
                ),
                "search_cache_ttl_seconds": self.config_manager.get(
                    "recall_engine.search_cache_ttl_seconds", 45.0
                ),
                "search_cache_max_size": self.config_manager.get(
                    "recall_engine.search_cache_max_size", 256
                ),
                "min_importance_for_retrieval": self.config_manager.get(
                    "recall_engine.min_importance_for_retrieval", 0.0
                ),
                "min_similarity_for_retrieval": self.config_manager.get(
                    "recall_engine.min_similarity_for_retrieval", 0.0
                ),
                "recent_memory_count": self.config_manager.get(
                    "recall_engine.recent_memory_count", 0
                ),
                "recent_memory_max_age_hours": self.config_manager.get(
                    "recall_engine.recent_memory_max_age_hours", 72
                ),
                "memory_type_filter": self.config_manager.get(
                    "recall_engine.memory_type_filter", "all"
                ),
                "cleanup_days_threshold": self.config_manager.get(
                    "maintenance.cleanup_days_threshold", 30
                ),
                "cleanup_importance_threshold": self.config_manager.get(
                    "maintenance.cleanup_importance_threshold", 0.3
                ),
                "stopwords_path": str(stopwords_dir),
                "graph_memory_enabled": graph_memory_enabled,
                "document_route_weight": self.config_manager.get(
                    "graph_memory.document_route_weight", 0.65
                ),
                "graph_route_weight": self.config_manager.get(
                    "graph_memory.graph_route_weight", 0.35
                ),
                "cross_route_bonus": self.config_manager.get(
                    "graph_memory.cross_route_bonus", 0.08
                ),
                "graph_expansion_limit": self.config_manager.get(
                    "graph_memory.expansion_limit", 24
                ),
                "graph_expansion_hops": self.config_manager.get(
                    "graph_memory.expansion_hops", 1
                ),
                "graph_second_hop_weight": self.config_manager.get(
                    "graph_memory.second_hop_weight", 0.4
                ),
                "dynamic_route_weighting": self.config_manager.get(
                    "graph_memory.dynamic_route_weighting", True
                ),
                "graph_max_topics": self.config_manager.get(
                    "graph_memory.max_topics_per_memory", 6
                ),
                "graph_max_participants": self.config_manager.get(
                    "graph_memory.max_participants_per_memory", 8
                ),
                "graph_max_facts": self.config_manager.get(
                    "graph_memory.max_facts_per_memory", 8
                ),
                "atom_enabled": self.config_manager.get(
                    "graph_memory.atom_enabled", True
                ),
                "atom_maintenance_interval_hours": self.config_manager.get(
                    "graph_memory.atom_maintenance_interval_hours", 24.0
                ),
                "atom_forget_delay_days": self.config_manager.get(
                    "graph_memory.atom_forget_delay_days", 7.0
                ),
                "atom_purge_delay_days": self.config_manager.get(
                    "graph_memory.atom_purge_delay_days", 30.0
                ),
            }

            self.memory_engine = MemoryEngine(
                db_path=str(db_path),
                faiss_db=self.db,
                graph_vector_db=self.graph_db,
                llm_provider=self.llm_provider,
                config=memory_engine_config,
            )
            await self.memory_engine.initialize()
            logger.info(f"{tag('init')} MemoryEngine 已初始化")

            # 初始化 ConversationManager
            conversation_db_path = data_dir_path / "conversations.db"
            conversation_store = ConversationStore(str(conversation_db_path))
            await conversation_store.initialize()

            self.conversation_manager = ConversationManager(store=conversation_store)
            logger.info(f"{tag('init')} ConversationManager 已初始化")

            # 初始化 MemoryProcessor
            # 注意：MemoryProcessor 不直接持有 llm_provider 实例引用，
            # 而是在每次调用时通过 AstrBot 上下文动态解析 Provider，
            # 以避免 AstrBot 重新创建 Provider 后旧实例的 httpx client 被关闭
            # 导致的 "Cannot send a request, as the client has been closed" 错误。
            llm_id = self.config_manager.get("provider_settings.llm_provider_id")
            self.memory_processor = MemoryProcessor(
                self.context,
                llm_provider=llm_id if llm_id else None,
                config={
                    "atom_enabled": memory_engine_config["atom_enabled"],
                },
            )
            logger.info(f"{tag('init')} MemoryProcessor 已初始化")

            # 异步初始化 TextProcessor
            if self.memory_engine and hasattr(self.memory_engine, "text_processor"):
                if self.memory_engine.text_processor and hasattr(
                    self.memory_engine.text_processor, "async_init"
                ):
                    await self.memory_engine.text_processor.async_init()
                    logger.info(f"{tag('init')} TextProcessor 停用词已加载")

            # 启动重要性衰减调度器
            # 维护任务由 maintenance 段统一控制：
            # - cleanup_days_threshold == 0：关闭自动清理
            # - backup_keep_days == 0：关闭自动备份
            decay_rate = self.config_manager.get("importance_decay.decay_rate", 0.01)
            cleanup_days = self.config_manager.get(
                "maintenance.cleanup_days_threshold", 30
            )
            backup_keep_days = self.config_manager.get(
                "maintenance.backup_keep_days", 7
            )
            if self.memory_engine and (decay_rate > 0 or cleanup_days > 0):
                scheduler = DecayScheduler(
                    memory_engine=self.memory_engine,
                    decay_rate=decay_rate,
                    data_dir=self.data_dir,
                    db_migration=self.db_migration,
                    backup_enabled=backup_keep_days > 0,
                    backup_keep_days=backup_keep_days,
                )
                await scheduler.start()
                self.decay_scheduler = scheduler
                logger.info(
                    f"{tag('init')} DecayScheduler 已启动 "
                    f"(cleanup_days={cleanup_days}, backup_keep_days={backup_keep_days})"
                )

            # 标记初始化完成
            self._initialization_complete = True
            logger.info(f"{tag('init')} LivingMemory 插件初始化成功。")

        except Exception as e:
            logger.error(f"{tag('init')} 完整初始化流程失败: {e}", exc_info=True)
            self._initialization_failed = True
            self._initialization_error = str(e)
            raise InitializationError(f"初始化失败: {e}") from e

    async def _preflight_database_compatibility(self) -> None:
        """在创建 FAISS/SQLite 运行组件前验证已有主数据库版本。"""
        db_path = Path(self.data_dir) / "livingmemory.db"
        if not db_path.exists():
            return
        if self.db_migration is None:
            self.db_migration = DBMigration(str(db_path))
        await self._check_and_migrate_database()

    async def _check_and_migrate_database(self) -> None:
        """检查并执行数据库迁移；失败时必须阻断初始化。"""
        try:
            if not self.db_migration:
                raise InitializationError("数据库迁移管理器未初始化")

            needs_migration = await self.db_migration.needs_migration()

            if not needs_migration:
                logger.info(f"{tag('init')} 数据库版本已是最新，无需迁移")
                return

            logger.info(f"{tag('init')} 检测到旧版本数据库，开始自动迁移。")

            result = await self.db_migration.migrate(progress_callback=None)

            if result.get("success"):
                logger.info(f"{tag('init')} 数据库迁移结果: {result.get('message')}")
                logger.info(f"{tag('init')}    耗时: {result.get('duration', 0):.2f}秒")
            else:
                message = str(result.get("message") or "数据库迁移失败")
                raise InitializationError(message)

        except asyncio.CancelledError:
            raise
        except InitializationError:
            raise
        except Exception as e:
            logger.error(f"{tag('init')} 数据库迁移检查失败: {e}", exc_info=True)
            raise InitializationError(f"数据库迁移检查失败: {e}") from e

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialization_complete

    @property
    def is_failed(self) -> bool:
        """是否初始化失败"""
        return self._initialization_failed

    @property
    def error_message(self) -> str | None:
        """错误消息"""
        return self._initialization_error

    async def ensure_initialized(self, timeout: float = 30.0) -> bool:
        """
        确保插件已初始化

        Args:
            timeout: 超时时间（秒）

        Returns:
            bool: 是否初始化成功
        """
        if self._initialization_complete:
            return True

        if self._initialization_failed:
            return False

        # 等待初始化完成
        start_time = time.time()
        while not self._initialization_complete and not self._initialization_failed:
            if time.time() - start_time > timeout:
                logger.error(f"{tag('init')} 等待插件初始化超时（{timeout}秒）")
                return False
            await asyncio.sleep(0.2)

        return self._initialization_complete

    async def _prepare_embedding_indexes(
        self,
        specs: list[EmbeddingIndexSpec],
    ) -> None:
        """启动前准备与当前 Provider 一致的主/图向量索引。"""
        if self.embedding_provider is None:
            raise ProviderNotReadyError("Embedding Provider 未初始化")
        service = getattr(self, "_embedding_index_bootstrap", None)
        if service is None:
            service = EmbeddingIndexBootstrapService(
                Path(self.data_dir) / "embedding_index_state.json",
                faiss_bootstrap=self._get_faiss_bootstrap(),
            )
            self._embedding_index_bootstrap = service
        await service.prepare(specs, self.embedding_provider)

    @staticmethod
    def _faiss_read_index_safe(index_path: str):
        """通过 ASCII 临时路径桥接 FAISS read_index。

        monkey-patch 已覆盖全局 faiss.read_index，此方法作为显式后备。
        """
        return _FAISS_BOOTSTRAP_COMPAT.faiss_read_index_safe(index_path)

    async def stop_scheduler(self) -> None:
        """停止衰减调度器"""
        if self.decay_scheduler:
            await self.decay_scheduler.stop()
            self.decay_scheduler = None

    async def stop_background_tasks(self) -> None:
        """停止初始化阶段的后台任务（如Provider重试）"""
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        self._retry_task = None
