"""
消息处理工具模块
负责消息内容提取、去重、限制等操作
"""

import asyncio
from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
        from ..base.config_manager import ConfigManager


class MessageUtils:
    """消息处理工具类"""

    def __init__(self, config_manager: "ConfigManager"):
        """
        初始化消息处理工具

        Args:
            config_manager: 配置管理器
        """
        self.config_manager = config_manager

    async def get_event_message_str(self, event: AstrMessageEvent) -> str:
        """取本轮用户消息文本；正文为空时回退到被引用消息的文本。

        守的静默失效：只发引用不打字（纯引用消息）时 `message_str` 为空，
        若不回退，记忆召回会被直接跳过——而用户的意思恰恰是"针对被引用的那条"，
        表现为"引用回复时 AI 突然不带记忆"，且没有任何报错。
        """
        get_message_str = getattr(event, "get_message_str", None)
        raw_message = ""

        if callable(get_message_str):
            raw_message = get_message_str()
            if asyncio.iscoroutine(raw_message):
                raw_message = await raw_message
        else:
            raw_message = getattr(event, "message_str", "")

        text = raw_message.strip() if isinstance(raw_message, str) else ""
        if text:
            return text

        return self._quoted_message_text(event)

    @staticmethod
    def _quoted_message_text(event: AstrMessageEvent) -> str:
        """从引用（Reply）组件里取被引用消息的文本；取不到返回空串。

        按类名判断而不 import AstrBot 的 Reply，避免不同版本组件路径差异。
        优先用组件自带的 `message_str`，退化时拼接 `chain` 里的纯文本段。
        """
        message_obj = getattr(event, "message_obj", None)
        chain = getattr(message_obj, "message", None) or []
        if not isinstance(chain, (list, tuple)):
            return ""

        for component in chain:
            # 优先看组件自身的 type（真实组件是 ComponentType.Reply，值 "reply"），
            # 类名只作兜底——按类名硬编码会随版本/替身实现漂移。
            component_type = getattr(component, "type", "")
            type_value = getattr(component_type, "value", component_type)
            if str(type_value).lower() != "reply" and component.__class__.__name__ != "Reply":
                continue
            quoted = getattr(component, "message_str", "") or ""
            if not isinstance(quoted, str):
                quoted = ""
            if not quoted.strip():
                pieces = []
                for inner in getattr(component, "chain", None) or []:
                    piece = getattr(inner, "text", "")
                    if isinstance(piece, str):
                        pieces.append(piece)
                quoted = "".join(pieces)
            if quoted.strip():
                return quoted.strip()
        return ""
