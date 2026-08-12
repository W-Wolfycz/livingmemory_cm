"""
会话管理器 - ConversationManager（CM-only 单路径版）。

CM 接管后，消息归档由 CM 单一负责；本类仅维护会话元数据
（如 reflection_cursors_v2/v3）和会话生命周期记录，不再做消息写入/读取/缓存。
"""

import json
from typing import Any

from ...log import log_ref, logger, tag

from ...storage.conversation_store import ConversationStore
from ..models.conversation_models import Session


class ConversationManager:
    """会话管理器（CM-only）：仅维护会话记录与元数据。"""

    def __init__(self, store: ConversationStore):
        """
        初始化会话管理器。

        Args:
            store: ConversationStore 实例
        """
        self.store = store
        logger.info(f"{tag('conv')} 初始化完成（CM-only 模式：仅维护元数据）")

    async def create_or_get_session(
        self, session_id: str, platform: str = "unknown"
    ) -> Session:
        """创建或获取会话。"""
        session = await self.store.get_session(session_id)
        if session:
            await self.store.update_session_activity(session_id)
            return session

        session = await self.store.create_session(session_id, platform)
        logger.debug(
            f"{tag('conv')} 创建新会话: {log_ref(session_id, 'session')}"
        )
        return session

    async def get_session_info(self, session_id: str) -> Session | None:
        """获取会话信息，不存在返回 None。"""
        session = await self.store.get_session(session_id)
        if not session:
            logger.warning(
                f"{tag('conv')} [{log_ref(session_id, 'session')}] 会话不存在"
            )
        return session

    async def get_recent_sessions(self, limit: int = 10) -> list[Session]:
        """获取最近活跃的会话。"""
        return await self.store.get_recent_sessions(limit)

    async def clear_session(self, session_id: str):
        """清空会话历史并重置记忆元数据。"""
        await self.store.delete_session_messages(session_id)
        await self.reset_session_metadata(session_id)
        logger.info(
            f"{tag('conv')} 已清空会话并重置记忆上下文: "
            f"{log_ref(session_id, 'session')}"
        )

    async def update_session_metadata(
        self, session_id: str, key: str, value: Any
    ) -> None:
        """更新会话元数据（合并写入）。"""
        session = await self.store.get_session(session_id)
        if not session:
            logger.warning(
                f"{tag('conv')} 会话 {log_ref(session_id, 'session')} 不存在，无法更新元数据"
            )
            return

        session.metadata[key] = value

        connection = getattr(self.store, "connection", None)
        if connection is not None:
            try:
                await connection.execute(
                    """
                    UPDATE sessions
                    SET metadata = ?
                    WHERE session_id = ?
                """,
                    (json.dumps(session.metadata, ensure_ascii=False), session_id),
                )
                await connection.commit()
            except Exception as e:
                logger.error(f"{tag('conv')} 更新会话元数据失败: {e}", exc_info=True)

        logger.debug(
            f"{tag('conv')} 更新会话元数据: {log_ref(session_id, 'session')}, "
            f"key={key}"
        )

    async def get_session_metadata(
        self, session_id: str, key: str, default: Any = None
    ) -> Any:
        """获取会话元数据，不存在返回 default。"""
        session = await self.store.get_session(session_id)
        if not session:
            return default
        return session.metadata.get(key, default)

    async def reset_session_metadata(self, session_id: str) -> None:
        """重置指定会话的所有元数据（用于 /reset、/new）。"""
        session = await self.store.get_session(session_id)
        if not session:
            logger.warning(
                f"{tag('conv')} 尝试重置元数据失败，会话 "
                f"{log_ref(session_id, 'session')} 不存在"
            )
            return
        session.metadata = {}
        connection = getattr(self.store, "connection", None)
        if connection is not None:
            try:
                await connection.execute(
                    """
                    UPDATE sessions
                    SET metadata = ?
                    WHERE session_id = ?
                """,
                    ("{}", session_id),
                )
                await connection.commit()
            except Exception as e:
                logger.error(f"{tag('conv')} 重置会话元数据失败: {e}", exc_info=True)
        logger.info(
            f"{tag('conv')} 已重置会话 {log_ref(session_id, 'session')} 的元数据 "
            "(记忆总结计数器已清零)"
        )
