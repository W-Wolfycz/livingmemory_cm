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


def test_validate_config_default_graph_route_weight() -> None:
    validated = validate_config({})
    assert validated.graph_memory.graph_route_weight == 0.35
    dumped = validated.model_dump()["graph_memory"]
    assert dumped.get("graph_route_weight") == 0.35
    assert "document_route_weight" not in dumped


def test_merge_legacy_both_weights_prefers_graph() -> None:
    merged = merge_config_with_defaults(
        {
            "graph_memory": {
                "document_route_weight": 0.6,
                "graph_route_weight": 0.4,
            }
        }
    )
    graph_mem = merged["graph_memory"]
    assert graph_mem["graph_route_weight"] == 0.4
    assert "document_route_weight" not in graph_mem


def test_merge_legacy_document_only_derives_graph() -> None:
    merged = merge_config_with_defaults(
        {"graph_memory": {"document_route_weight": 0.6}}
    )
    graph_mem = merged["graph_memory"]
    assert graph_mem["graph_route_weight"] == pytest.approx(0.4)
    assert "document_route_weight" not in graph_mem


def test_merge_legacy_graph_only_keeps_graph() -> None:
    merged = merge_config_with_defaults(
        {"graph_memory": {"graph_route_weight": 0.7}}
    )
    graph_mem = merged["graph_memory"]
    assert graph_mem["graph_route_weight"] == 0.7
    assert "document_route_weight" not in graph_mem


def test_merge_no_legacy_keeps_default() -> None:
    merged = merge_config_with_defaults({})
    assert merged["graph_memory"]["graph_route_weight"] == 0.35
    assert "document_route_weight" not in merged["graph_memory"]


def test_validate_config_legacy_document_only() -> None:
    validated = validate_config(
        {"graph_memory": {"document_route_weight": 0.6}}
    )
    assert validated.graph_memory.graph_route_weight == pytest.approx(0.4)


def test_validate_config_legacy_both_keeps_graph() -> None:
    validated = validate_config(
        {
            "graph_memory": {
                "document_route_weight": 0.6,
                "graph_route_weight": 0.4,
            }
        }
    )
    assert validated.graph_memory.graph_route_weight == 0.4


# ============ AstrBot 完整性注入后的 graph 迁移 ============
# 注入后 document_route_weight 恒存在（默认 0.65 也会被注入），
# 必须按值而非“键是否存在”判断旧配置。


def test_merge_astrbot_injected_document_non_default_derives() -> None:
    # 旧用户：document=0.6，graph 被注入为默认 0.35 → 推导 graph=0.4
    merged = merge_config_with_defaults(
        {
            "graph_memory": {
                "graph_route_weight": 0.35,
                "document_route_weight": 0.6,
            }
        }
    )
    graph_mem = merged["graph_memory"]
    assert graph_mem["graph_route_weight"] == pytest.approx(0.4)
    assert "document_route_weight" not in graph_mem


def test_merge_astrbot_injected_graph_explicit_preferred() -> None:
    # 显式 graph 优先：即使 document 也非默认，仍保留 graph
    merged = merge_config_with_defaults(
        {
            "graph_memory": {
                "graph_route_weight": 0.4,
                "document_route_weight": 0.6,
            }
        }
    )
    graph_mem = merged["graph_memory"]
    assert graph_mem["graph_route_weight"] == 0.4
    assert "document_route_weight" not in graph_mem


def test_merge_astrbot_injected_double_default_keeps_default() -> None:
    # 双默认（graph=0.35、document=0.65）→ 保持默认 0.35
    merged = merge_config_with_defaults(
        {
            "graph_memory": {
                "graph_route_weight": 0.35,
                "document_route_weight": 0.65,
            }
        }
    )
    graph_mem = merged["graph_memory"]
    assert graph_mem["graph_route_weight"] == 0.35
    assert "document_route_weight" not in graph_mem


def test_validate_astrbot_injected_document_non_default_derives() -> None:
    validated = validate_config(
        {
            "graph_memory": {
                "graph_route_weight": 0.35,
                "document_route_weight": 0.6,
            }
        }
    )
    assert validated.graph_memory.graph_route_weight == pytest.approx(0.4)
    assert "document_route_weight" not in validated.model_dump()["graph_memory"]


# ==================== log_with_bot_id 顶层全局项迁移 ====================


def test_config_section_order_log_with_bot_id_first() -> None:
    defaults = get_default_config()
    keys = list(defaults.keys())
    assert keys[0] == "log_with_bot_id"
    assert "provider_settings" in keys
    assert len(keys) == 8
    assert "log" not in defaults
    assert "document_route_weight" not in defaults["graph_memory"]


def test_default_log_with_bot_id_false() -> None:
    defaults = get_default_config()
    assert defaults["log_with_bot_id"] is False
    validated = validate_config({})
    assert validated.log_with_bot_id is False
    assert not hasattr(validated, "log")


def test_merge_legacy_log_group_migrates_to_top_level() -> None:
    merged = merge_config_with_defaults({"log": {"log_with_bot_id": True}})
    assert merged["log_with_bot_id"] is True
    assert "log" not in merged


def test_merge_astrbot_injected_legacy_true_migrates() -> None:
    # AstrBot 注入后：顶层 log_with_bot_id 恒为默认 False（旧配置无顶层键），
    # 旧 log.log_with_bot_id=true 由隐藏兼容键保留 → 必须以旧意图迁移为 True，
    # 否则真实旧用户迁移会静默失效。
    merged = merge_config_with_defaults(
        {"log_with_bot_id": False, "log": {"log_with_bot_id": True}}
    )
    assert merged["log_with_bot_id"] is True
    assert "log" not in merged


def test_merge_top_level_true_keeps_true() -> None:
    merged = merge_config_with_defaults(
        {"log_with_bot_id": True, "log": {"log_with_bot_id": False}}
    )
    assert merged["log_with_bot_id"] is True
    assert "log" not in merged


def test_merge_top_level_false_legacy_false_stays_false() -> None:
    # 顶层显式 false（新 UI 关闭）且 legacy 为注入默认 false → 保持 false
    merged = merge_config_with_defaults(
        {"log_with_bot_id": False, "log": {"log_with_bot_id": False}}
    )
    assert merged["log_with_bot_id"] is False
    assert "log" not in merged


def test_merge_no_log_config_keeps_default() -> None:
    merged = merge_config_with_defaults({})
    assert merged["log_with_bot_id"] is False
    assert "log" not in merged


def test_validate_config_legacy_log_group_migrates() -> None:
    validated = validate_config({"log": {"log_with_bot_id": True}})
    assert validated.log_with_bot_id is True
    assert not hasattr(validated, "log")


def test_validate_config_legacy_log_group_disabled() -> None:
    validated = validate_config({"log": {"log_with_bot_id": False}})
    assert validated.log_with_bot_id is False


def test_validate_astrbot_injected_legacy_true_migrates() -> None:
    validated = validate_config(
        {"log_with_bot_id": False, "log": {"log_with_bot_id": True}}
    )
    assert validated.log_with_bot_id is True
    assert not hasattr(validated, "log")


def test_validate_top_level_true_keeps_true() -> None:
    validated = validate_config(
        {"log_with_bot_id": True, "log": {"log_with_bot_id": True}}
    )
    assert validated.log_with_bot_id is True


def test_validate_top_level_false_legacy_false_stays_false() -> None:
    validated = validate_config(
        {"log_with_bot_id": False, "log": {"log_with_bot_id": False}}
    )
    assert validated.log_with_bot_id is False


def test_config_manager_get_top_level_log_with_bot_id() -> None:
    from livingmemory_cm.core.base.config_manager import ConfigManager

    manager = ConfigManager({"log_with_bot_id": True})
    assert manager.get("log_with_bot_id") is True


def test_config_manager_get_migrated_log_with_bot_id() -> None:
    from livingmemory_cm.core.base.config_manager import ConfigManager

    manager = ConfigManager({"log": {"log_with_bot_id": True}})
    assert manager.get("log_with_bot_id") is True


def test_config_manager_get_default_log_with_bot_id() -> None:
    from livingmemory_cm.core.base.config_manager import ConfigManager

    manager = ConfigManager({})
    assert manager.get("log_with_bot_id") is False


def test_config_manager_astrbot_injected_legacy_true_migrates() -> None:
    from livingmemory_cm.core.base.config_manager import ConfigManager

    manager = ConfigManager(
        {"log_with_bot_id": False, "log": {"log_with_bot_id": True}}
    )
    assert manager.get("log_with_bot_id") is True


def test_config_manager_removes_legacy_keys_from_source() -> None:
    """迁移成功后从原始配置对象移除隐藏兼容键，避免 UI 后续操作被覆盖。"""
    from livingmemory_cm.core.base.config_manager import ConfigManager

    source = {
        "log_with_bot_id": False,
        "log": {"log_with_bot_id": True},
        "graph_memory": {"graph_route_weight": 0.35, "document_route_weight": 0.6},
    }
    manager = ConfigManager(source)

    assert manager.get("log_with_bot_id") is True
    assert manager.get("graph_memory.graph_route_weight") == pytest.approx(0.4)
    assert "log" not in source
    assert "document_route_weight" not in source["graph_memory"]


@pytest.mark.asyncio
async def test_config_manager_persist_legacy_cleanup_calls_saver() -> None:
    """AstrBotConfig 路径：落盘调用 save_config_async；失败只记日志不抛出。"""
    from livingmemory_cm.core.base.config_manager import ConfigManager

    class _SavingConfig(dict):
        def __init__(self, data):
            super().__init__(data)
            self.saved = 0

        async def save_config_async(self):
            self.saved += 1

    source = _SavingConfig(
        {"log_with_bot_id": False, "log": {"log_with_bot_id": True}}
    )
    manager = ConfigManager(source)
    await manager.persist_legacy_cleanup()
    assert source.saved == 1
    assert "log" not in source


@pytest.mark.asyncio
async def test_config_manager_persist_legacy_cleanup_ignores_plain_dict() -> None:
    """普通 dict（本地测试/非 AstrBot 路径）没有落盘方法，静默跳过。"""
    from livingmemory_cm.core.base.config_manager import ConfigManager

    source = {"log_with_bot_id": False, "log": {"log_with_bot_id": True}}
    manager = ConfigManager(source)
    await manager.persist_legacy_cleanup()  # 不应抛异常
    assert manager.get("log_with_bot_id") is True


@pytest.mark.asyncio
async def test_config_manager_persist_legacy_cleanup_saver_failure_is_logged() -> None:
    """save_config_async 抛异常时不影响初始化，只记日志。"""
    from livingmemory_cm.core.base.config_manager import ConfigManager

    class _BrokenSavingConfig(dict):
        async def save_config_async(self):
            raise RuntimeError("disk full")

    source = _BrokenSavingConfig({"log": {"log_with_bot_id": True}})
    manager = ConfigManager(source)
    await manager.persist_legacy_cleanup()  # 不应抛异常
    assert manager.get("log_with_bot_id") is True


# ==================== _conf_schema.json 隐藏兼容键 ====================


def _load_schema() -> dict:
    schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_conf_schema_hidden_log_group_and_document_weight() -> None:
    schema = _load_schema()
    keys = list(schema.keys())
    assert keys[0] == "log_with_bot_id"
    # 隐藏兼容 log 组存在且 invisible
    log_group = schema.get("log")
    assert log_group is not None
    assert log_group.get("invisible") is True
    assert log_group.get("type") == "object"
    assert isinstance(log_group.get("items"), dict)
    assert set(log_group["items"].keys()) == {"log_with_bot_id"}
    assert log_group["items"]["log_with_bot_id"]["type"] == "bool"
    assert log_group["items"]["log_with_bot_id"]["default"] is False
    # 隐藏兼容 document_route_weight 存在且 invisible
    doc = schema["graph_memory"]["items"].get("document_route_weight")
    assert doc is not None
    assert doc.get("invisible") is True
    assert doc.get("type") == "float"
    assert doc.get("default") == 0.65
    # 可见键不因新增隐藏键而改变
    assert schema["log_with_bot_id"].get("type") == "bool"


def test_conf_schema_visible_structure_unchanged_7_plus_1() -> None:
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
    assert section_count == 7
    assert total == 44


def test_conf_schema_graph_document_visible_count_unaffected() -> None:
    schema = _load_schema()
    graph_items = schema["graph_memory"]["items"]
    visible_graph = [k for k, v in graph_items.items() if not v.get("invisible")]
    assert "document_route_weight" not in visible_graph
    # 可见图段项数与合并前一致（不含隐藏兼容键）
    assert len(visible_graph) == 14
