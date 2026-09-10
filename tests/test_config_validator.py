"""配置合并与旧配置迁移（graph_route_weight 合并）测试。

本文件同时模拟 AstrBot 4.27.3 在插件 __init__ 之前用 _conf_schema.json 做的
配置完整性注入：旧 `log` 配置组与 `document_route_weight` 由隐藏兼容键保护并
注入默认值（log_with_bot_id 顶层恒为 False、document_route_weight 恒为 0.65），
迁移逻辑必须在这种“隐藏键已被注入默认值”的输入上仍能正确恢复旧值。
"""

import json
from pathlib import Path

import pytest

from livingmemory_cm.core.base.config_validator import (
    GraphMemoryConfig,
    LivingMemoryConfig,
    get_default_config,
    merge_config_with_defaults,
    validate_config,
)


def test_graph_memory_config_has_single_graph_route_weight() -> None:
    config = GraphMemoryConfig()
    assert config.graph_route_weight == 0.35
    assert not hasattr(config, "document_route_weight")
    # 单一权重不再要求归一化校验器
    assert "validate_route_weights" not in type(config).__dict__


# ============ graph_route_weight 迁移 ============
# 注入后 document_route_weight 恒存在（默认 0.65 也会被注入），
# 必须按值而非“键是否存在”判断旧配置。


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param({}, 0.35, id="no-legacy-keeps-default"),
        pytest.param(
            {"graph_memory": {"graph_route_weight": 0.7}}, 0.7, id="graph-only"
        ),
        pytest.param(
            {"graph_memory": {"document_route_weight": 0.6}}, 0.4, id="document-only"
        ),
        pytest.param(
            {
                "graph_memory": {
                    "graph_route_weight": 0.4,
                    "document_route_weight": 0.6,
                }
            },
            0.4,
            id="explicit-graph-preferred",
        ),
        pytest.param(
            {
                "graph_memory": {
                    "graph_route_weight": 0.35,
                    "document_route_weight": 0.6,
                }
            },
            0.4,
            id="injected-default-graph-derives-from-document",
        ),
        pytest.param(
            {
                "graph_memory": {
                    "graph_route_weight": 0.35,
                    "document_route_weight": 0.65,
                }
            },
            0.35,
            id="injected-double-default-keeps-default",
        ),
    ],
)
def test_graph_route_weight_migration(raw, expected) -> None:
    """merge 与 validate 两条路径一致：单一 graph_route_weight，legacy 键移除。"""
    merged = merge_config_with_defaults(raw)
    assert merged["graph_memory"]["graph_route_weight"] == pytest.approx(expected)
    assert "document_route_weight" not in merged["graph_memory"]

    validated = validate_config(raw)
    assert validated.graph_memory.graph_route_weight == pytest.approx(expected)
    assert "document_route_weight" not in validated.model_dump()["graph_memory"]


# ==================== log_with_bot_id 顶层全局项 ====================


def test_config_section_order_log_with_bot_id_first() -> None:
    defaults = get_default_config()
    keys = list(defaults.keys())
    assert keys[0] == "log_with_bot_id"
    assert "provider_settings" in keys
    assert len(keys) == 7
    assert "log" not in defaults
    assert "document_route_weight" not in defaults["graph_memory"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param({}, False, id="no-config-keeps-default"),
        pytest.param({"log_with_bot_id": True}, True, id="top-level-true"),
        pytest.param({"log_with_bot_id": False}, False, id="top-level-false"),
        # 旧的 log 配置组已不再迁移：即便里面是 true 也不生效
        pytest.param(
            {"log": {"log_with_bot_id": True}}, False, id="legacy-group-ignored"
        ),
        pytest.param(
            {"log_with_bot_id": True, "log": {"log_with_bot_id": False}},
            True,
            id="top-level-wins-over-legacy",
        ),
    ],
)
def test_log_with_bot_id_is_top_level_only(raw, expected) -> None:
    """顶层 log_with_bot_id 是唯一入口，旧 log 组不参与解析。"""
    assert validate_config(raw).log_with_bot_id is expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param({"log_with_bot_id": True}, True, id="top-level-true"),
        pytest.param({"log": {"log_with_bot_id": True}}, False, id="legacy-ignored"),
        pytest.param({}, False, id="default"),
    ],
)
def test_config_manager_get_log_with_bot_id(raw, expected) -> None:
    from livingmemory_cm.core.base.config_manager import ConfigManager

    assert ConfigManager(raw).get("log_with_bot_id") is expected


def test_config_manager_removes_graph_legacy_key_from_source() -> None:
    """graph 迁移结果写回原始配置对象，并移除隐藏兼容键。

    legacy 键必须删除（避免 UI 后续操作被旧值覆盖），同时迁移结果要写回原始
    对象（否则 reload 后迁移值随注入默认值一起丢失）。
    """
    from livingmemory_cm.core.base.config_manager import ConfigManager

    source = {
        "graph_memory": {"graph_route_weight": 0.35, "document_route_weight": 0.6},
    }
    manager = ConfigManager(source)

    assert manager.get("graph_memory.graph_route_weight") == pytest.approx(0.4)
    # legacy 键已从原始对象移除
    assert "document_route_weight" not in source["graph_memory"]
    # 迁移后的值已写回原始对象，落盘/reload 后保持一致
    assert source["graph_memory"]["graph_route_weight"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_config_manager_persist_legacy_cleanup_calls_saver() -> None:
    """AstrBotConfig 路径：落盘快照含迁移结果，且不再含 legacy 键。"""
    from livingmemory_cm.core.base.config_manager import ConfigManager

    class _SavingConfig(dict):
        def __init__(self, data):
            super().__init__(data)
            self.saved = 0
            self.snapshot = None

        async def save_config_async(self):
            self.saved += 1
            self.snapshot = dict(self)

    source = _SavingConfig({"graph_memory": {"document_route_weight": 0.6}})
    manager = ConfigManager(source)
    await manager.persist_legacy_cleanup()
    assert source.saved == 1
    # 落盘快照即 reload 后的输入：迁移值保持、legacy 键消失
    assert source.snapshot["graph_memory"]["graph_route_weight"] == pytest.approx(0.4)
    assert "document_route_weight" not in source.snapshot["graph_memory"]


@pytest.mark.asyncio
async def test_config_manager_persist_legacy_cleanup_ignores_plain_dict() -> None:
    """普通 dict（本地测试/非 AstrBot 路径）没有落盘方法，静默跳过。"""
    from livingmemory_cm.core.base.config_manager import ConfigManager

    source = {"graph_memory": {"document_route_weight": 0.6}}
    manager = ConfigManager(source)
    await manager.persist_legacy_cleanup()  # 不应抛异常
    assert manager.get("graph_memory.graph_route_weight") == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_config_manager_persist_legacy_cleanup_saver_failure_is_logged() -> None:
    """save_config_async 抛异常时不影响初始化，只记日志。"""
    from livingmemory_cm.core.base.config_manager import ConfigManager

    class _BrokenSavingConfig(dict):
        async def save_config_async(self):
            raise RuntimeError("disk full")

    source = _BrokenSavingConfig({"graph_memory": {"document_route_weight": 0.6}})
    manager = ConfigManager(source)
    await manager.persist_legacy_cleanup()  # 不应抛异常
    assert manager.get("graph_memory.graph_route_weight") == pytest.approx(0.4)


# ==================== _conf_schema.json 隐藏兼容键 ====================


def _load_schema() -> dict:
    schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_conf_schema_has_only_document_weight_hidden_key() -> None:
    schema = _load_schema()
    keys = list(schema.keys())
    assert keys[0] == "log_with_bot_id"
    # 旧的 log 隐藏兼容组已删除，不再保留一次性迁移
    assert "log" not in schema
    # 唯一保留的隐藏兼容键：graph_memory.document_route_weight
    doc = schema["graph_memory"]["items"].get("document_route_weight")
    assert doc is not None
    assert doc.get("invisible") is True
    assert doc.get("type") == "float"
    assert doc.get("default") == 0.65
    assert schema["log_with_bot_id"].get("type") == "bool"


def test_conf_schema_visible_structure_unchanged_6_plus_1() -> None:
    schema = _load_schema()
    total = 0
    section_count = 0
    for key, value in schema.items():
        if value.get("invisible"):
            continue
        if value.get("type") == "object" and isinstance(value.get("items"), dict):
            section_count += 1
            visible_items = [
                k for k, v in value["items"].items() if not v.get("invisible")
            ]
            total += len(visible_items)
        else:
            total += 1
    assert section_count == 6
    assert total == 47


def test_conf_schema_graph_document_visible_count_unaffected() -> None:
    schema = _load_schema()
    graph_items = schema["graph_memory"]["items"]
    visible_graph = [k for k, v in graph_items.items() if not v.get("invisible")]
    assert "document_route_weight" not in visible_graph
    # 可见图段项数与合并前一致（不含隐藏兼容键）
    assert len(visible_graph) == 14


def test_llm_max_retries_default_and_schema() -> None:
    """llm_max_retries 默认 5，validator 与 schema 一致，越界被拒绝。"""
    from livingmemory_cm.core.base.config_validator import (
        ProviderConfig,
        validate_config,
    )

    assert ProviderConfig().llm_max_retries == 5
    schema = _load_schema()
    item = schema["provider_settings"]["items"]["llm_max_retries"]
    assert item["type"] == "int"
    assert item["default"] == 5

    assert validate_config({}).provider_settings.llm_max_retries == 5
    assert (
        validate_config({"provider_settings": {"llm_max_retries": 7}})
        .provider_settings.llm_max_retries
        == 7
    )

    with pytest.raises(Exception):
        validate_config({"provider_settings": {"llm_max_retries": 0}})
    with pytest.raises(Exception):
        validate_config({"provider_settings": {"llm_max_retries": 11}})


def test_search_timeout_seconds_default_and_schema() -> None:
    """search_timeout_seconds 默认 5 秒，validator 与 schema 一致，越界被拒绝。"""
    from livingmemory_cm.core.base.config_validator import (
        RecallEngineConfig,
        validate_config,
    )

    assert RecallEngineConfig().search_timeout_seconds == 5.0
    schema = _load_schema()
    item = schema["recall_engine"]["items"]["search_timeout_seconds"]
    assert item["type"] == "float"
    assert item["default"] == 5.0

    assert validate_config({}).recall_engine.search_timeout_seconds == 5.0
    assert (
        validate_config({"recall_engine": {"search_timeout_seconds": 12.5}})
        .recall_engine.search_timeout_seconds
        == 12.5
    )
    assert (
        validate_config({"recall_engine": {"search_timeout_seconds": 0}})
        .recall_engine.search_timeout_seconds
        == 0.0
    )

    with pytest.raises(Exception):
        validate_config({"recall_engine": {"search_timeout_seconds": -1}})
    with pytest.raises(Exception):
        validate_config({"recall_engine": {"search_timeout_seconds": 61}})


def test_embedding_batch_size_default_and_schema() -> None:
    """embedding_batch_size 默认 16，validator 与 schema 一致，越界被拒绝。"""
    from livingmemory_cm.core.base.config_validator import (
        ProviderConfig,
        validate_config,
    )

    assert ProviderConfig().embedding_batch_size == 16
    schema = _load_schema()
    item = schema["provider_settings"]["items"]["embedding_batch_size"]
    assert item["type"] == "int"
    assert item["default"] == 16

    assert validate_config({}).provider_settings.embedding_batch_size == 16
    assert (
        validate_config({"provider_settings": {"embedding_batch_size": 10}})
        .provider_settings.embedding_batch_size
        == 10
    )

    with pytest.raises(Exception):
        validate_config({"provider_settings": {"embedding_batch_size": 0}})
    with pytest.raises(Exception):
        validate_config({"provider_settings": {"embedding_batch_size": 2049}})
