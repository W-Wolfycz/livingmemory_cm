"""
消息处理工具模块
负责消息内容提取、去重、限制等操作
"""

import asyncio
import hashlib
import re
import time
from typing import TYPE_CHECKING

from ...log import logger, tag
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest

if TYPE_CHECKING:
    from ..managers.conversation_manager import ConversationManager
    from ..base.config_manager import ConfigManager


class MessageUtils:
    """消息处理工具类"""

    def __init__(
        self,
        config_manager: "ConfigManager",
        conversation_manager: "ConversationManager | None",
    ):
        """
        初始化消息处理工具

        Args:
            config_manager: 配置管理器
            conversation_manager: 会话管理器
        """
        self.config_manager = config_manager
        self.conversation_manager = conversation_manager

        # 消息去重缓存
        self._message_dedup_cache: dict[str, float] = {}
        self._dedup_cache_max_size = 1000
        self._dedup_cache_ttl = 300

    async def build_dedup_key(
        self, event: AstrMessageEvent, session_id: str, content: str
    ) -> str | None:
        """构建去重键：优先使用 message_id，缺失时退化为消息内容指纹。"""
        raw_message_id = getattr(
            getattr(event, "message_obj", None), "message_id", None
        )
        if raw_message_id is not None:
            message_id = str(raw_message_id).strip()
            if message_id:
                return f"id:{message_id}"

        sender_id = event.get_sender_id() if hasattr(event, "get_sender_id") else ""
        timestamp = getattr(getattr(event, "message_obj", None), "timestamp", 0)
        fingerprint = f"{session_id}|{sender_id}|{timestamp}|{content}"
        digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
        return f"fallback:{digest}"

    async def is_duplicate_message(self, dedup_key: str | None) -> bool:
        """检查消息是否已经处理过（惰性过期 + 溢出时逐条淘汰）"""
        if not dedup_key:
            return False

        result = dedup_key in self._message_dedup_cache
        if not result:
            return False

        # 惰性过期检查：命中时若已过期则视为未命中
        if time.time() - self._message_dedup_cache[dedup_key] > self._dedup_cache_ttl:
            del self._message_dedup_cache[dedup_key]
            return False

        return True

    async def mark_message_processed(self, dedup_key: str | None):
        """标记消息已处理（超限时淘汰最早插入的条目）"""
        if not dedup_key:
            return
        cache = self._message_dedup_cache
        if len(cache) >= self._dedup_cache_max_size:
            # 淘汰最早插入的条目（O(n) 但仅超限时触发，n≤1000）
            oldest_key = min(cache.items(), key=lambda x: x[1])[0]
            del cache[oldest_key]
        cache[dedup_key] = time.time()

    async def extract_message_content(
        self, event: AstrMessageEvent, req: ProviderRequest | None = None
    ) -> str:
        """提取消息内容，按组件原始顺序拼接，保留文字与图片的相对位置。
        若 AstrBot 已完成图片转述（req.extra_user_content_parts 中含 <image_caption> 标签），
        则按图片出现顺序依次替换，不会重复消费同一条转述。
        """
        from astrbot.core.message.components import (
            At,
            AtAll,
            Face,
            File,
            Forward,
            Image,
            Plain,
            Record,
            Reply,
            Video,
        )

        # 预先提取所有图片转述（按 extra_user_content_parts 中的出现顺序）
        # AstrBot 按消息链中图片的顺序依次追加转述，与 get_messages() 中 Image 的顺序一一对应
        caption_queue: list[str] = []
        if req is not None:
            for part in getattr(req, "extra_user_content_parts", []):
                text = getattr(part, "text", "")
                if not text:
                    continue
                for m in re.findall(
                    r"<image_caption>(.*?)</image_caption>", text, re.DOTALL
                ):
                    m = m.strip()
                    if m:
                        caption_queue.append(m)

        parts = []
        caption_idx = 0

        # 按组件原始顺序遍历，保留文字与图片的相对位置
        for component in event.get_messages():
            if isinstance(component, Plain):
                text = component.text.strip() if component.text else ""
                if text:
                    parts.append(text)
            elif isinstance(component, Image):
                if caption_idx < len(caption_queue):
                    parts.append(f"[图片: {caption_queue[caption_idx]}]")
                    caption_idx += 1
                else:
                    parts.append("[图片]")
            elif isinstance(component, Record):
                parts.append("[语音]")
            elif isinstance(component, Video):
                parts.append("[视频]")
            elif isinstance(component, File):
                file_name = component.name or "未知文件"
                parts.append(f"[文件: {file_name}]")
            elif isinstance(component, Face):
                parts.append(f"[表情:{component.id}]")
            elif isinstance(component, At):
                if isinstance(component, AtAll):
                    parts.append("[At:全体成员]")
                else:
                    parts.append(f"[At:{component.qq}]")
            elif isinstance(component, Forward):
                parts.append("[转发消息]")
            elif isinstance(component, Reply):
                if component.message_str:
                    parts.append(f"[引用: {component.message_str[:30]}]")
                else:
                    parts.append("[引用消息]")
            else:
                component_type = getattr(
                    component,
                    "type",
                    component.__class__.__name__,
                )
                logger.debug(f"{tag('msg')} 跳过未知消息组件: {component_type}")

        return " ".join(parts).strip()

    async def get_event_message_str(self, event: AstrMessageEvent) -> str:
        """Get normalized raw message text from event."""
        get_message_str = getattr(event, "get_message_str", None)
        raw_message = ""

        if callable(get_message_str):
            raw_message = get_message_str()
            if asyncio.iscoroutine(raw_message):
                raw_message = await raw_message
        else:
            raw_message = getattr(event, "message_str", "")

        if not isinstance(raw_message, str):
            return ""

        return raw_message.strip()
