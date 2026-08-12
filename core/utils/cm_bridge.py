"""CM (chat_memory) 插件状态桥接（CM-only 单路径版）。

LM 通过本模块感知 CM 的 context_takeover 状态。CM 恒启用，启动时已校验，
运行期仅做实例查找，不再缓存或降级。
"""

from __future__ import annotations

from typing import Any

_CM_PLUGIN_NAME = "chat_memory"


def _resolve_star_instance(raw: Any) -> Any:
    """从 context.get_registered_star 返回值中取出真正的插件实例。

    AstrBot 不同版本可能返回包装对象或实例本身，用 getattr 链兜底。
    """
    candidate = raw
    for attr in ("star", "star_cls", "star_instance", "instance"):
        try:
            value = getattr(raw, attr, None)
        except Exception:
            value = None
        if value is not None:
            candidate = value
            break
    return candidate


def get_cm_plugin(context: Any) -> Any:
    """返回 chat_memory 的插件实例，未注册时返回 None。"""
    if context is None:
        return None
    getter = getattr(context, "get_registered_star", None)
    if not callable(getter):
        return None
    raw = getter(_CM_PLUGIN_NAME)
    if raw is None:
        return None
    return _resolve_star_instance(raw)


def get_cm_status(context: Any) -> tuple[bool, int]:
    """返回 (ct_enabled, ct_limit_rounds)。"""
    plugin = get_cm_plugin(context)
    if plugin is None:
        return (False, 0)

    ct_enabled = bool(getattr(plugin, "ct_enable", False))
    ct_limit_raw = getattr(plugin, "ct_limit_rounds", 0)
    ct_limit = int(ct_limit_raw) if ct_limit_raw else 0
    return (ct_enabled, ct_limit)


__all__ = [
    "get_cm_plugin",
    "get_cm_status",
]
