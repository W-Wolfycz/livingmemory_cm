"""包内日志 wrapper：

- ``log_with_bot_id``：在日志前缀末尾追加平台实例标识（``platform_id``），
  与模块名、子模块名并存：``[livingmemory_cm][recall][platform:BOT1]``。
  关闭时前缀为 ``[livingmemory_cm][recall]``，模块定位始终保留。
- 标识取 AstrBot 事件 ``event.get_platform_id()``（即 ``platform_meta.id``，
  与 UMO 首段一致、平台内唯一、不含账号），由核心在其文档中保证唯一性。
  会话/用户引用仍通过 ``log_ref`` 保持脱敏（如 ``[session:<hash>]``）。
  前缀通过 ``tag(module, event)`` 在调用点拼装——只有能拿到 event 的调用点
  （hook/命令）才会带平台段，后台调度等无 event 的日志保持模块与子模块。
  关键事件入口（recall/reflection/session reset）应使用
  ``tag_event(module, event)`` 传入事件对象。

日志级别完全跟随 AstrBot 原生配置，插件不再自行提级（debug→info）；需要查看
详细运行信息时，请在 AstrBot 全局日志级别中开启 debug。

各模块统一通过 ``logger.debug/info/...`` 调用，前缀用 ``tag('module_name')``
函数获取。建议每个文件用一个固定 module 名（如 recall / reflection / store）。
"""

import hashlib

from astrbot.api import logger as _astrbot_logger

# 前缀第一段固定为插件模块名；各调用点传入的名字作为子模块段。
PLUGIN_TAG = "livingmemory_cm"


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
    """日志前缀：``[livingmemory_cm][子模块][platform:{platform_id}]``。

    段位规则：
    1. 第一段恒为模块名 ``[livingmemory_cm]``；
    2. 传入 ``module`` 时追加子模块段 ``[recall]``；
    3. ``log_with_bot_id=True`` 且传入 event 且能取到 ``platform_id`` 时，
       末尾追加 ``[platform:BOT1]``；取不到（事件/平台未实现）时省略该段。

    平台标识用 AstrBot 事件 ``event.get_platform_id()``（``platform_meta.id``，
    与 UMO 首段一致、平台内唯一、不含账号），不使用 self_id；关闭开关时保持
    ``[模块][子模块]``，不会丢掉模块定位。会话/用户引用请继续使用 ``log_ref``
    保持脱敏。

    建议调用点固定一个 module 名，例如::

        from ..log import logger, tag
        logger.info(f"{tag('recall')} 召回 5 条")

    关键事件入口（recall/reflection/session reset）如需平台标识，请使用
    ``tag_event(module, event)`` 传入事件对象。
    """
    prefix = f"[{PLUGIN_TAG}]"
    if module:
        prefix += f"[{module}]"
    platform_id = _platform_id(event) if _with_bot_id else ""
    if platform_id:
        prefix += f"[platform:{platform_id}]"
    return prefix


def _platform_id(event) -> str:
    """取事件所属平台实例 ID；无 event 或取不到时返回空串。"""
    if event is None:
        return ""
    try:
        value = event.get_platform_id()
    except Exception:
        # 事件/平台未实现该接口时退化为无平台段，不影响模块定位。
        return ""
    return str(value or "").strip()


def tag_event(module: str | None = None, event=None) -> str:
    """与 ``tag(module, event)`` 等价，语义上明确“这是事件入口”。

    供 recall/reflection/session reset 等有 event 的关键入口使用，以便
    ``log_with_bot_id=True`` 时按 ``platform_id`` 附加平台段。
    """
    return tag(module, event)
