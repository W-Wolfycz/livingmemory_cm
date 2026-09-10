"""
配置管理器
集中管理插件配置的加载、验证和访问
"""

from typing import Any

from ...log import logger, tag

from .config_validator import (
    get_default_config,
    merge_config_with_defaults,
    validate_config,
)
from .exceptions import ConfigurationError


class ConfigManager:
    """配置管理器"""

    def __init__(self, user_config: dict[str, Any] | None = None):
        """
        初始化配置管理器

        Args:
            user_config: 用户提供的配置字典
        """
        self._raw_config = user_config or {}
        self._config: dict[str, Any] = {}
        self._config_obj = None
        self._load_config()

    def _load_config(self) -> None:
        """加载并验证配置"""
        try:
            # 合并默认配置
            merged_config = merge_config_with_defaults(self._raw_config)
            # 验证配置
            self._config_obj = validate_config(merged_config)
            self._config = self._config_obj.model_dump()
            # 迁移成功后从原始配置对象移除隐藏兼容键，避免 AstrBot 完整性检查
            # 反复把旧值注入回磁盘，导致用户在 UI 的后续操作被 legacy 值覆盖
            # （如把 graph_route_weight 设回默认 0.35）。
            self._remove_legacy_keys_from_source()
        except Exception:
            logger.warning(f"{tag('config')} 配置验证失败，已降级为默认配置", exc_info=True)
            # 配置验证失败，使用默认配置
            try:
                self._config = get_default_config()
                self._config_obj = validate_config(self._config)
            except Exception as e2:
                raise ConfigurationError(f"加载默认配置失败: {e2}") from e2

    def _remove_legacy_keys_from_source(self) -> None:
        """把迁移结果写回原始配置对象，并移除已迁移的隐藏兼容键。

        迁移逻辑（config_validator）只在浅拷贝上工作；若只删除原始对象中的
        ``graph_memory.document_route_weight`` 而不写回迁移值，
        ``persist_legacy_cleanup`` 落盘的仍是 AstrBot 注入的默认值
        （graph_route_weight=0.35），下次加载时迁移结果会静默丢失。因此这里同步
        把 ``self._config`` 中迁移后的值写回原始对象：reload 后旧用户意图保持，
        且 legacy 键已清除，用户在 UI 的新设置不再被旧值覆盖。本方法只改内存；
        AstrBotConfig 路径的落盘由 ``persist_legacy_cleanup`` 在异步初始化阶段完成。

        顶层 ``log_with_bot_id`` 是唯一入口，不存在旧 ``log`` 组的迁移。
        """
        source = self._raw_config
        if not isinstance(source, dict):
            return
        graph_memory = source.get("graph_memory")
        migrated_graph = self._config.get("graph_memory")
        if isinstance(graph_memory, dict) and isinstance(migrated_graph, dict):
            graph_memory.pop("document_route_weight", None)
            graph_memory["graph_route_weight"] = migrated_graph.get(
                "graph_route_weight", 0.35
            )

    async def persist_legacy_cleanup(self) -> None:
        """把迁移结果与隐藏兼容键的清除持久化到磁盘（仅 AstrBotConfig 路径）。

        由 ``main.py`` 的 ``initialize()`` 调用：AstrBot 传入的配置对象是
        ``AstrBotConfig``（dict 子类），其 ``save_config_async`` 会落盘当前快照；
        其他路径（普通 dict / 本地测试）没有落盘方法，直接跳过。失败只记日志，
        不影响插件启动——最坏情况是下次加载重复一次迁移。
        """
        source = self._raw_config
        saver = getattr(source, "save_config_async", None)
        if not callable(saver):
            return
        try:
            await saver()
        except Exception as exc:
            logger.warning(
                f"{tag('config')} 持久化隐藏兼容键清理失败（不影响本次运行）: {exc}"
            )
            return
        logger.debug(
            f"{tag('config')} 已持久化隐藏兼容键清理（graph_memory.document_route_weight）"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的嵌套键（如 "provider_settings.llm_provider_id"）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def get_section(self, section: str) -> dict[str, Any]:
        """
        获取配置节

        Args:
            section: 配置节名称

        Returns:
            配置节字典
        """
        return self._config.get(section, {})

    def get_all(self) -> dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()

    @property
    def provider_settings(self) -> dict[str, Any]:
        """Provider设置"""
        return self.get_section("provider_settings")

    @property
    def recall_engine(self) -> dict[str, Any]:
        """召回引擎配置"""
        return self.get_section("recall_engine")

    @property
    def reflection_engine(self) -> dict[str, Any]:
        """反思引擎配置"""
        return self.get_section("reflection_engine")

    @property
    def graph_memory(self) -> dict[str, Any]:
        """Graph-memory settings."""
        return self.get_section("graph_memory")
