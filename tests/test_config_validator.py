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


# ==================== 越界/非法配置逐项修正 ====================


def test_normalize_config_clamps_out_of_range_values() -> None:
    """越界数值截断到最近边界，而不是让整份配置回退默认。"""
    from livingmemory_cm.core.base.config_validator import normalize_config

    merged = merge_config_with_defaults(
        {
            "recall_engine": {"top_k": 60, "search_timeout_seconds": 100.0},
            "provider_settings": {"llm_max_retries": 0},
            "maintenance": {"cleanup_days_threshold": 99999},
        }
    )

    normalized, corrections = normalize_config(merged)

    assert normalized["recall_engine"]["top_k"] == 50
    assert normalized["recall_engine"]["search_timeout_seconds"] == 60.0
    assert normalized["provider_settings"]["llm_max_retries"] == 1
    assert normalized["maintenance"]["cleanup_days_threshold"] == 3650
    paths = {item["path"] for item in corrections}
    assert paths == {
        "recall_engine.top_k",
        "recall_engine.search_timeout_seconds",
        "provider_settings.llm_max_retries",
        "maintenance.cleanup_days_threshold",
    }
    # 修正后必须能通过严格校验
    validate_config(normalized)


def test_normalize_config_resets_broken_section_and_enum() -> None:
    """配置节不是对象 → 重置该节；枚举越界 → 回退默认值。"""
    from livingmemory_cm.core.base.config_validator import normalize_config

    merged = merge_config_with_defaults(
        {
            "importance_decay": "broken",
            "graph_memory": {"enabled": False},
            "recall_engine": {"memory_type_filter": "nonsense"},
        }
    )

    normalized, corrections = normalize_config(merged)

    assert normalized["importance_decay"] == {}
    assert normalized["recall_engine"]["memory_type_filter"] == "all"
    # 未越界的设置必须原样保留
    assert normalized["graph_memory"]["enabled"] is False
    reasons = {item["path"]: item["new"] for item in corrections}
    assert reasons["importance_decay"] == {}
    assert reasons["recall_engine.memory_type_filter"] == "all"
    validate_config(normalized)


def test_config_manager_keeps_other_settings_when_one_value_is_out_of_range() -> None:
    """核心回归：单项越界不再让整份生效配置降级为默认值。"""
    from livingmemory_cm.core.base.config_manager import ConfigManager

    source = {
        "recall_engine": {"top_k": 60},
        "graph_memory": {"enabled": False, "graph_route_weight": 0.7},
        "maintenance": {"freeze_grace_days": 3},
    }

    manager = ConfigManager(source)

    assert manager.get("recall_engine.top_k") == 50
    assert manager.get("graph_memory.enabled") is False
    assert manager.get("graph_memory.graph_route_weight") == pytest.approx(0.7)
    assert manager.get("maintenance.freeze_grace_days") == 3
    # 修正结果写回原始对象，供 persist_legacy_cleanup 落盘
    assert source["recall_engine"]["top_k"] == 50
