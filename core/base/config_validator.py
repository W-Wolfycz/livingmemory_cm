"""
config_validator.py - 配置验证模块
提供配置验证和默认值管理功能。
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationError

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
        description=(
            "召回查询用于消歧的当前用户最近问答轮数，0 = 仅当前发言；仅用于构建召回查询"
            "（消解「那个」「继续」等指代），不注入上下文。单位跟随 CM llm_status_filter："
            "仅 llm_success 时 N=完整问答轮，含其他状态时 N=消息条数"
        ),
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
    search_timeout_seconds: float = Field(
        default=5.0,
        ge=0.0,
        le=60.0,
        description=(
            "主链路记忆召回（含 embedding 检索）的最长等待秒数，"
            "超时跳过注入、不阻塞用户回复；0 表示不限时"
        ),
    )
    rerank_enabled: bool = Field(
        default=False,
        description=(
            "启用 Rerank 重排序：召回候选交给 AstrBot 的 Rerank 提供商按查询相关性"
            "重新打分，用于把真正相关的记忆排进注入列表。"
            "提供商不可用、调用失败或返回空时自动沿用原排序"
        ),
    )
    rerank_provider_id: str | None = Field(
        default=None,
        description=(
            "Rerank 提供商 ID，需手填（AstrBot 的 provider 选择器只列对话模型，"
            "没有 rerank 专用选择器）；留空时回退到已加载的第一个 Rerank 提供商"
        ),
    )
    rerank_candidates: int = Field(
        default=20,
        ge=2,
        le=100,
        description=(
            "参与重排序的候选条数（2-100）。候选越多重排越准、调用越慢；"
            "重排后仍只注入 top_k 条"
        ),
    )
    rerank_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "重排序分数在最终得分中的占比（0-1）：0.6 表示六成看重排相关性、"
            "四成沿用原有重要性/时间加权；1.0 完全按重排结果排序"
        ),
    )


class ReflectionEngineConfig(BaseModel):
    """反思引擎配置"""

    trigger_count: int = Field(
        default=0,
        ge=0,
        le=2000,
        description="触发数量（0 = 跟随 CM；CM 配对模式按轮，混合模式按消息条数）",
    )
    refusal_advance_count: int = Field(
        default=5,
        ge=1,
        le=2000,
        description="模型内容安全/政策 skip 或 Provider 高置信内容安全拒绝（含 AstrBot content_filter）时推进的 CM 单位数",
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
    freeze_grace_days: int = Field(
        default=7,
        ge=0,
        le=3650,
        description=(
            "解冻 persona 后的清理豁免天数（0 关闭豁免）；"
            "冻结期间照常衰减但绝不自动删除"
        ),
    )


class ProviderConfig(BaseModel):
    """Provider配置"""

    embedding_provider_id: str | None = Field(
        default=None, description="Embedding Provider ID"
    )
    embedding_batch_size: int = Field(
        default=16,
        ge=1,
        le=2048,
        description=(
            "索引重建时单次请求嵌入的文本条数（1-2048）。不同 Provider 上限不同"
            "（OpenAI 官方 2048、DashScope text-embedding-v4 仅 10）；超出上限时"
            "插件会自动降批重试，但按上限配置可减少无效请求"
        ),
    )
    llm_provider_id: str | None = Field(default=None, description="LLM Provider ID")
    llm_max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "萃取 LLM 调用的最大尝试次数（含首次，范围 1-10，默认对齐 AstrBot "
            "provider_settings.request_max_retries 的 5 次）。"
            "仅普通调用失败与非法 JSON 响应触发重试；status=skip 与内容安全拒绝不重试"
        ),
    )


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


class GraphMemoryConfig(BaseModel):
    """Graph-memory retrieval configuration."""

    enabled: bool = Field(default=True, description="是否启用图记忆双路检索")
    graph_route_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="图路融合分数在最终双路排序中的占比；文档路权重自动为 1 - 图路权重",
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


class LivingMemoryConfig(BaseModel):
    """完整插件配置"""

    log_with_bot_id: bool = Field(
        default=False,
        description=(
            "日志前缀末尾附加事件所属平台的实例标识 platform_id（event.get_platform_id()，与 UMO 首段一致、不含账号），"
            "格式 [livingmemory_cm][子模块][platform:xxx]；关闭时只保留模块与子模块段。"
            "全局配置项，不属于任何配置段"
        ),
    )
    recall_engine: RecallEngineConfig = Field(default_factory=RecallEngineConfig)
    reflection_engine: ReflectionEngineConfig = Field(
        default_factory=ReflectionEngineConfig
    )
    provider_settings: ProviderConfig = Field(default_factory=ProviderConfig)
    graph_memory: GraphMemoryConfig = Field(default_factory=GraphMemoryConfig)
    importance_decay: ImportanceDecayConfig = Field(
        default_factory=ImportanceDecayConfig, description="重要性衰减配置"
    )
    maintenance: MaintenanceConfig = Field(
        default_factory=MaintenanceConfig, description="维护任务配置（清理 + 备份）"
    )

    model_config = {"extra": "allow"}  # 允许额外字段，向前兼容


def normalize_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """逐项修正越界/非法配置，返回 (修正后的配置, 修正记录)。

    AstrBot 配置页保存时只按 schema 校验类型、不校验范围，而 pydantic 的边界是硬
    约束：任一项越界会让 ``validate_config`` 整体失败，``ConfigManager`` 于是把
    **整份生效配置**降级为默认值——用户在面板上调过的召回、萃取、图路、维护参数
    会一起被静默重置。这里在校验前逐项钳制：

    - 数值越界 → 截断到最近边界（取 pydantic 错误里的 ``le``/``ge``）；
    - 其它校验失败（类型、枚举、嵌套结构）→ 该项回退默认值；
    - 某一节不是 dict → 整节重置（由后续合并补默认值）。

    修正记录形如 ``{"path": "recall_engine.top_k", "old": 60, "new": 50, "reason": ...}``，
    由调用方记日志并写回配置文件；无法修正的错误保持原样，交给 ``validate_config``
    走既有的整包兜底。
    """
    normalized = dict(config)
    corrections: list[dict[str, Any]] = []

    for section_name, field_info in LivingMemoryConfig.model_fields.items():
        section_model = field_info.annotation
        if not (
            isinstance(section_model, type)
            and issubclass(section_model, BaseModel)
        ):
            continue

        payload = normalized.get(section_name)
        if payload is None:
            continue
        if not isinstance(payload, dict):
            corrections.append(
                {
                    "path": section_name,
                    "old": payload,
                    "new": {},
                    "reason": "配置节不是对象，已重置该节",
                }
            )
            normalized[section_name] = {}
            continue

        payload = dict(payload)
        normalized[section_name] = payload
        for _ in range(32):  # 每轮至少修正一项，设上限防御异常输入
            try:
                section_model.model_validate(payload)
                break
            except ValidationError as exc:
                fixed = False
                for error in exc.errors():
                    location = error.get("loc") or ()
                    if len(location) != 1:
                        continue  # 嵌套结构问题不在逐项修正范围内
                    key = str(location[0])
                    if key not in section_model.model_fields:
                        continue
                    old_value = payload.get(key)
                    new_value = _corrected_value(
                        section_model.model_fields[key],
                        old_value,
                        error,
                    )
                    if new_value is _UNCHANGED:
                        continue
                    payload[key] = new_value
                    corrections.append(
                        {
                            "path": f"{section_name}.{key}",
                            "old": old_value,
                            "new": new_value,
                            "reason": error.get("msg", ""),
                        }
                    )
                    fixed = True
                if not fixed:
                    break
            except Exception:
                break

    return normalized, corrections


_UNCHANGED = object()

_CLAMP_ERROR_TYPES = {
    "less_than_equal": "le",
    "greater_than_equal": "ge",
    "less_than": "lt",
    "greater_than": "gt",
}


def _corrected_value(field_info: Any, value: Any, error: dict[str, Any]) -> Any:
    """按校验错误给出该字段的修正值；无法修正时返回 ``_UNCHANGED``。"""
    context = error.get("ctx") or {}
    bound_key = _CLAMP_ERROR_TYPES.get(str(error.get("type")))
    if bound_key and bound_key in context and isinstance(value, (int, float)):
        bound = context[bound_key]
        if isinstance(bound, (int, float)):
            if bound_key in ("lt", "gt"):
                # 严格边界：整数退一格，浮点用极小步长
                step = 1 if isinstance(bound, int) else abs(bound) * 1e-6 or 1e-6
                bound = bound - step if bound_key == "lt" else bound + step
            return type(value)(bound) if isinstance(bound, type(value)) else bound

    default = field_info.get_default(call_default_factory=True)
    if value == default:
        return _UNCHANGED
    return default


def _migrate_graph_route_weight(config: dict[str, Any]) -> dict[str, Any]:
    """旧配置 graph_memory.document_route_weight / graph_route_weight → 单一 graph_route_weight。

    真实 AstrBot 4.27.3 完整性注入后，`document_route_weight` 隐藏兼容键恒存在
    （默认 0.65 也会被注入），因此不能再用“键是否存在”判断旧配置，而要按值判断：

    1. `graph_route_weight` 非默认 0.35：视为显式 graph 值，直接采用（优先）；
    2. 否则 `document_route_weight` 存在且非默认 0.65：推导 graph = 1 - document；
    3. 否则保持默认 0.35。

    迁移后移除 `document_route_weight`。本函数幂等：已迁移配置（无
    document_route_weight）会原样返回。
    """
    graph_mem = config.get("graph_memory")
    if not isinstance(graph_mem, dict):
        return config
    new_graph_mem = dict(graph_mem)
    legacy_graph = new_graph_mem.get("graph_route_weight")
    legacy_document = new_graph_mem.get("document_route_weight")
    # 非 AstrBot/测试直接注入路径下 graph_route_weight 可能缺失（None）；
    # AstrBot 完整性注入后则恒存在。因此同时排除 None 与默认值才算“显式”。
    if legacy_graph is not None and legacy_graph != 0.35:
        new_graph_mem["graph_route_weight"] = legacy_graph
    elif legacy_document is not None and legacy_document != 0.65:
        new_graph_mem["graph_route_weight"] = 1.0 - legacy_document
    else:
        new_graph_mem["graph_route_weight"] = 0.35
    new_graph_mem.pop("document_route_weight", None)
    migrated = dict(config)
    migrated["graph_memory"] = new_graph_mem
    return migrated


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
    migrated = _migrate_graph_route_weight(raw_config)
    try:
        config = LivingMemoryConfig(**migrated)
        logger.debug(f"{tag('config')} 配置验证成功")
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

    # 旧配置迁移需在合并默认值之前执行：否则默认 graph_route_weight 已存在，
    # 文档-only 旧配置将无法推导（优先级：显式 graph > document 推导 > 默认）。
    migrated = _migrate_graph_route_weight(dict(user_config))
    merged = deep_merge(default_config, migrated)
    logger.debug(f"{tag('config')} 配置已与默认值合并")
    return merged
