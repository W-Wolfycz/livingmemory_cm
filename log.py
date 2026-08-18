"""包内日志 wrapper：

- ``log_with_bot_id``：在日志前缀中附加机器人实例标识
  （如 ``[livingmemory_cm:bot-10000]``），区分多 bot 共存场景；前缀直接使用
  AstrBot 事件 ``event.get_self_id()`` 的原始 self_id，按原文输出便于按 Bot ID
  定位日志。会话/用户引用仍通过 ``log_ref`` 保持脱敏（如 ``[session:<hash>]``）。
  前缀通过 ``tag(module, event)`` 在调用点拼装——只有能拿到 event 的调用点
  （hook/命令）才会带 bot 前缀，后台调度等无 event 的日志保持模块名或默认。
  需要真实体现 Bot ID 的关键事件入口（recall/reflection/session reset）
  应使用 ``tag_event(module, event)`` 传入事件对象。

日志级别完全跟随 AstrBot 原生配置，插件不再自行提级（debug→info）；需要查看
详细运行信息时，请在 AstrBot 全局日志级别中开启 debug。

各模块统一通过 ``logger.debug/info/...`` 调用，前缀用 ``tag('module_name')``
函数获取。建议每个文件用一个固定 module 名（如 recall / reflection / store）。
"""

import hashlib

from astrbot.api import logger as _astrbot_logger


class _LoggerProxy:
    """转发到 astrbot logger，保持 AstrBot 原生日志级别。"""

    def debug(self, msg, *args, **kwargs):
        _astrbot_logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        _astrbot_logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        _astrbot_logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        _astrbot_logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        _astrbot_logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        _astrbot_logger.exception(msg, *args, **kwargs)


logger = _LoggerProxy()


# ==================== bot 实例区分 ====================

_with_bot_id = False


def log_ref(value, label: str = "id") -> str:
    """返回不暴露原值的稳定短引用，供日志关联使用。"""
    if value is None or value == "":
        return f"{label}:none"
    digest = hashlib.blake2s(
        str(value).encode("utf-8", errors="replace"),
        digest_size=6,
        person=b"lmemlog",
    ).hexdigest()
    return f"{label}:{digest}"


def configure(log_with_bot_id: bool = False) -> None:
    """启动时由 main.py 调用，根据配置开关区分实例。"""
    global _with_bot_id
    _with_bot_id = bool(log_with_bot_id)


def set_log_with_bot_id(enabled: bool) -> None:
    """独立 setter（用于配置热更新场景）。"""
    global _with_bot_id
    _with_bot_id = bool(enabled)


def tag(module: str | None = None, event=None) -> str:
    """日志前缀。

    优先级：
    1. ``log_with_bot_id=True`` 且传入 event 且能取到 self_id → ``[livingmemory_cm:bot-<self_id>]``
    2. 传入 module → ``[livingmemory_cm:module]``
    3. 默认 → ``[livingmemory_cm]``

    Bot 标识使用 AstrBot 事件 ``event.get_self_id()`` 的原始 self_id，按原文
    输出（如 ``[livingmemory_cm:bot-10000]``）便于定位；会话/用户引用请继续使用
    ``log_ref`` 保持脱敏。

    建议调用点固定一个 module 名，例如::

        from ..log import logger, tag
        logger.info(f"{tag('recall')} 召回 5 条")

    关键事件入口（recall/reflection/session reset）如需真实体现 Bot ID，请使用
    ``tag_event(module, event)``。
    """
    if _with_bot_id and event is not None:
        try:
            self_id = event.get_self_id()
            if self_id:
                return f"[livingmemory_cm:bot-{self_id}]"
        except Exception:
            # 取不到 self_id（事件/平台未实现）时不加 bot 前缀，保持模块名。
            pass
    if module:
        return f"[livingmemory_cm:{module}]"
    return "[livingmemory_cm]"


def tag_event(module: str | None = None, event=None) -> str:
    """与 ``tag(module, event)`` 等价，语义上明确“这是事件入口”。

    供 recall/reflection/session reset 等有 event 的关键入口使用，以便
    ``log_with_bot_id=True`` 时按原始 self_id 附加 Bot 前缀。
    """
    return tag(module, event)
