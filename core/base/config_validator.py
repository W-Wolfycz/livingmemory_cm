"""
config_validator.py - 配置验证模块
提供配置验证和默认值管理功能。
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from ...log import logger, tag


class RecallEngineConfig(BaseModel):
    """回忆引擎配置"""

    top_k: int = Field(
        default=5, ge=0, le=50, description="返回记忆数量。设为 0 则跳过自动召回和注入"
    )
    max_k: int = Field(
        default=10, ge=1, le=50, description="Agent 主动检索时允许的最大返回数量"
    )
    injection_method: str = Field(
        default="extra_user_content",
        description=(
            "记忆注入方式: "
            "extra_user_content(推荐，临时消息追加到用户消息末尾，不影响前缀缓存且不污染对话历史), "
            "user_message_before(用户消息前), "
            "user_message_after(用户消息后), "
            "fake_tool_call(伪造工具调用)"
        ),
    )
    search_cache_ttl_seconds: float = Field(
        default=45.0, ge=0.0, le=600.0, description="检索缓存 TTL 秒数（0 关闭缓存）"
    )
    search_cache_max_size: int = Field(
        default=256, ge=0, le=10000, description="检索缓存最大条目数"
    )
    query_context_rounds: int = Field(
        default=2,
        ge=0,
        le=10,
        description="召回查询用于消歧的当前用户最近完整问答轮数，0 = 仅当前发言",
    )
    query_context_max_chars: int = Field(
        default=800,
        ge=0,
        le=4000,
        description="召回查询中历史问答的字符上限，0 = 不加入历史",
    )
    query_context_max_age_seconds: int = Field(
        default=0,
        ge=0,
        le=31_536_000,
        description="召回消歧 CM 问答最大年龄，0 表示不限时间",
    )
    injection_max_memories: int = Field(
        default=3, ge=1, le=10, description="去重后最多注入的记忆条数"
    )
    injection_max_chars: int = Field(
        default=3200, ge=500, le=12000, description="单次长期记忆注入字符预算"
    )
    min_importance_for_retrieval: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="召回最低重要性，0 表示不过滤",
    )
    min_similarity_for_retrieval: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="召回最低向量相似度，0 表示不过滤",
    )
    recent_memory_count: int = Field(
        default=0,
        ge=0,
        le=20,
        description="当前 session/persona 的近期记忆保留槽位，0 表示关闭",
    )
    recent_memory_max_age_hours: int = Field(
        default=72,
        ge=0,
        le=8760,
        description="近期记忆最大时间窗口，0 表示不限时间",
    )
    memory_type_filter: str = Field(
        default="all",
        pattern="^(all|event_only)$",
        description="记忆类型过滤：all 或 event_only",
    )


class ReflectionEngineConfig(BaseModel):
    """反思引擎配置"""

    trigger_count: int = Field(
        default=0,
        ge=0,
        le=2000,
        description="触发数量（0 = 跟随 CM；CM 配对模式按轮，混合模式按消息条数）",
    )


class AgentToolsConfig(BaseModel):
    """Agent 工具配置"""

    enable_recall_tool: bool = Field(
        default=True, description="是否启用 Agent 主动回忆工具"
    )
    enable_memorize_tool: bool = Field(
        default=False, description="是否启用 Agent 主动记忆写入工具"
    )


class MaintenanceConfig(BaseModel):
    """维护任务配置（每日衰减 → 清理 → 备份）

    是否启用某项任务由对应天数控制：
    - cleanup_days_threshold == 0：关闭自动清理
    - backup_keep_days == 0：关闭自动备份
    """

    cleanup_days_threshold: int = Field(
        default=30, ge=0, le=3650, description="清理天数阈值（0 关闭）"
    )
    cleanup_importance_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="清理重要性阈值"
    )
    backup_keep_days: int = Field(
        default=7, ge=0, le=365, description="备份保留天数（0 关闭）"
    )


class ProviderConfig(BaseModel):
    """Provider配置"""

    embedding_provider_id: str | None = Field(
        default=None, description="Embedding Provider ID"
    )
    llm_provider_id: str | None = Field(default=None, description="LLM Provider ID")


class ImportanceDecayConfig(BaseModel):
    """重要性衰减配置"""

    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="每日衰减率")
    access_decay_window_days: float = Field(
        default=30.0, ge=1.0, le=3650.0, description="访问频次强化的有效窗口天数"
    )
    access_decay_max_count: int = Field(
        default=10, ge=1, le=10000, description="抵消衰减所需的访问次数上限"
    )
    access_count_decay_multiplier: float = Field(
        default=0.5, ge=0.0, le=1.0, description="每日衰减后访问次数保留比例"
    )
    protected_importance_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="达到该重要性的记忆不参与每日衰减",
    )


class LogSettings(BaseModel):
    """日志设置（debug_to_info + log_with_bot_id）"""

    debug_to_info: bool = Field(
        default=False, description="把 debug 日志提级为 info 输出"
    )
    log_with_bot_id: bool = Field(
        default=False, description="日志前缀附加 platform_id 区分多 bot 实例"
    )


class GraphMemoryConfig(BaseModel):
    """Graph-memory retrieval configuration."""

    enabled: bool = Field(default=True, description="是否启用图记忆双路检索")
    document_route_weight: float = Field(
        default=0.65, ge=0.0, le=1.0, description="文档路权重"
    )
    graph_route_weight: float = Field(
        default=0.35, ge=0.0, le=1.0, description="图路权重"
    )
    cross_route_bonus: float = Field(
        default=0.08, ge=0.0, le=0.5, description="双路同时命中的额外加分"
    )
    expansion_limit: int = Field(
        default=24, ge=1, le=200, description="图邻居扩展候选上限"
    )
    expansion_hops: int = Field(
        default=1, ge=1, le=2, description="图关键词检索邻居扩展跳数"
    )
    second_hop_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="二跳图扩展候选权重"
    )
    dynamic_route_weighting: bool = Field(
        default=True, description="是否按查询意图动态调整文档路和图路权重"
    )
    max_topics_per_memory: int = Field(
        default=6, ge=1, le=20, description="单条记忆最多索引主题数"
    )
    max_participants_per_memory: int = Field(
        default=8, ge=1, le=30, description="单条记忆最多索引参与者数"
    )
    max_facts_per_memory: int = Field(
        default=8, ge=1, le=30, description="单条记忆最多索引事实数"
    )
    # Atom-level memory configuration
    atom_enabled: bool = Field(
        default=True, description="是否启用记忆原子化（细化粒度+时间衰减）"
    )
    atom_maintenance_interval_hours: float = Field(
        default=24.0, ge=1.0, le=168.0, description="原子生命周期维护间隔(小时)"
    )
    atom_forget_delay_days: float = Field(
        default=7.0, ge=1.0, le=90.0, description="过期原子延迟遗忘天数"
    )
    atom_purge_delay_days: float = Field(
        default=30.0, ge=1.0, le=365.0, description="遗忘原子物理清理延迟天数"
    )

    @model_validator(mode="after")
    def validate_route_weights(self):
        """Normalize route weights to sum to 1.0 for numerically stable fusion."""
        total = self.document_route_weight + self.graph_route_weight
        if total <= 0:
            self.document_route_weight = 0.65
            self.graph_route_weight = 0.35
        elif total != 1.0:
            self.document_route_weight = self.document_route_weight / total
            self.graph_route_weight = self.graph_route_weight / total
        return self


class LivingMemoryConfig(BaseModel):
    """完整插件配置"""

    recall_engine: RecallEngineConfig = Field(default_factory=RecallEngineConfig)
    reflection_engine: ReflectionEngineConfig = Field(
        default_factory=ReflectionEngineConfig
    )
    agent_tools: AgentToolsConfig = Field(default_factory=AgentToolsConfig)
    provider_settings: ProviderConfig = Field(default_factory=ProviderConfig)
    graph_memory: GraphMemoryConfig = Field(default_factory=GraphMemoryConfig)
    importance_decay: ImportanceDecayConfig = Field(
        default_factory=ImportanceDecayConfig, description="重要性衰减配置"
    )
    maintenance: MaintenanceConfig = Field(
        default_factory=MaintenanceConfig, description="维护任务配置（清理 + 备份）"
    )
    log: LogSettings = Field(default_factory=LogSettings, description="日志设置")

    model_config = {"extra": "allow"}  # 允许额外字段，向前兼容


def validate_config(raw_config: dict[str, Any]) -> LivingMemoryConfig:
    """
    验证并返回规范化的配置对象。

    Args:
        raw_config: 原始配置字典

    Returns:
        LivingMemoryConfig: 验证后的配置对象

    Raises:
        ValueError: 配置验证失败
    """
    try:
        config = LivingMemoryConfig(**raw_config)
        logger.info(f"{tag('config')} 配置验证成功")
        return config
    except Exception as e:
        logger.error(f"{tag('config')} 配置验证失败: {e}")
        raise ValueError(f"插件配置无效: {e}") from e


def get_default_config() -> dict[str, Any]:
    """
    获取默认配置字典。

    Returns:
        dict[str, Any]: 默认配置
    """
    return LivingMemoryConfig().model_dump()


def merge_config_with_defaults(user_config: dict[str, Any]) -> dict[str, Any]:
    """
    将用户配置与默认配置合并。

    Args:
        user_config: 用户提供的配置

    Returns:
        dict[str, Any]: 合并后的配置
    """
    default_config = get_default_config()

    def deep_merge(default: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
        """深度合并两个字典"""
        result = default.copy()
        for key, value in user.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(default_config, user_config)
    logger.debug(f"{tag('config')} 配置已与默认值合并")
    return merged
