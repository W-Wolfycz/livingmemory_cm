"""包内日志 wrapper：

- ``debug_to_info``：把 debug 日志提级为 info 输出，让用户无需改 AstrBot 后台
  日志级别即可看到详细运行信息（与 time_awareness / chat_memory 等插件一致）。
- ``log_with_bot_id``：在日志前缀中附加机器人实例标识
  （如 ``[livingmemory_cm:bot-7f3a1c2d]``），区分多 bot 共存场景；原始
  platform ID 不写入日志。
  前缀通过 ``tag(module, event)`` 在调用点拼装——只有能拿到 event 的调用点
  （hook/命令）才会带 platform_id，后台调度等无 event 的日志保持模块名或默认。

各模块统一通过 ``logger.debug/info/...`` 调用，前缀用 ``tag('module_name')``
函数获取。建议每个文件用一个固定 module 名（如 recall / reflection / store）。
"""

import hashlib

from astrbot.api import logger as _astrbot_logger


class _LoggerProxy:
    """转发到 astrbot logger，但 ``debug`` 受 ``debug_to_info`` 控制。"""

    def __init__(self):
        self.debug_to_info = False

    def debug(self, msg, *args, **kwargs):
        if self.debug_to_info:
            _astrbot_logger.info(msg, *args, **kwargs)
        else:
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


def configure(debug_to_info: bool = False, log_with_bot_id: bool = False) -> None:
    """启动时由 main.py 调用，根据配置开关提级 / 区分实例。"""
    logger.debug_to_info = bool(debug_to_info)
    global _with_bot_id
    _with_bot_id = bool(log_with_bot_id)


def set_log_with_bot_id(enabled: bool) -> None:
    """独立 setter（用于配置热更新场景）。"""
    global _with_bot_id
    _with_bot_id = bool(enabled)


def tag(module: str | None = None, event=None) -> str:
    """日志前缀。

    优先级：
    1. ``log_with_bot_id=True`` 且传入 event 且能取到 platform_id → ``[livingmemory_cm:bot-hash]``
    2. 传入 module → ``[livingmemory_cm:module]``
    3. 默认 → ``[livingmemory_cm]``

    建议调用点固定一个 module 名，例如::

        from ..log import logger, tag
        logger.info(f"{tag('recall')} 召回 5 条")
    """
    if _with_bot_id and event is not None:
        try:
            pid = event.get_platform_id()
            if pid:
                return f"[livingmemory_cm:bot-{log_ref(pid, 'ref').split(':', 1)[1]}]"
        except Exception:
            pass
    if module:
        return f"[livingmemory_cm:{module}]"
    return "[livingmemory_cm]"
