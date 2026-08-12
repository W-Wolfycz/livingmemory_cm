"""
会话存储层 - ConversationStore（CM-only 单路径版）。

CM 接管后，消息归档由 chat_memory 单一负责；本类仅维护 sessions 表
（会话元数据 + reflection_cursors_v2 等记忆元信息），不再写入或读取
messages 表。messages 表保留是为了向后兼容旧版本数据，clear_session 仍
会清理它以应对迁移场景。
"""

import asyncio
import time
from pathlib import Path

import aiosqlite

from ..log import logger, tag

from ..core.models.conversation_models import Session


class ConversationStore:
    """会话存储管理器（CM-only）：仅维护 sessions 表。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        self.connection = await aiosqlite.connect(self.db_path)
        if self.connection is not None:
            self.connection.row_factory = aiosqlite.Row
            await self.connection.execute("PRAGMA journal_mode = WAL")
            await self.connection.execute("PRAGMA busy_timeout = 10000")
        await self._create_tables()
        await self._create_indexes()
        logger.info(f"{tag('store')} 数据库初始化完成: {self.db_path}")

    async def close(self) -> None:
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info(f"{tag('store')} 数据库连接已关闭")

    async def _create_tables(self) -> None:
        if self.connection is None:
            return
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                platform TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_active_at REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                participants TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)
        # messages 表保留：旧版本数据兼容 + clear_session 清理场景
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                sender_name TEXT,
                group_id TEXT,
                platform TEXT,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        await self.connection.commit()

    async def _create_indexes(self) -> None:
        if self.connection is None:
            return
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_id ON sessions(session_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_last_active ON sessions(last_active_at DESC)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, timestamp DESC)"
        )
        await self.connection.commit()

    # ==================== 会话管理 ====================

    async def create_session(self, session_id: str, platform: str) -> Session:
        now = time.time()
        if not isinstance(platform, str):
            platform = getattr(platform, "name", str(platform))
            logger.warning(
                f"{tag('store')} [create_session] platform 参数不是字符串类型，已自动转换为: {platform}"
            )
        if self.connection is None:
            raise RuntimeError("数据库连接未初始化")
        async with self._write_lock:
            cursor = await self.connection.execute(
                """
                INSERT INTO sessions (session_id, platform, created_at, last_active_at, message_count, participants, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, platform, now, now, 0, "[]", "{}"),
            )
            await self.connection.commit()

        session = Session(
            id=cursor.lastrowid if cursor.lastrowid else 0,
            session_id=session_id,
            platform=platform,
            created_at=now,
            last_active_at=now,
            message_count=0,
            participants=[],
            metadata={},
        )
        logger.debug(f"{tag('store')} 创建会话: {session_id}")
        return session

    async def get_session(self, session_id: str) -> Session | None:
        if self.connection is None:
            return None
        async with self.connection.execute(
            """
            SELECT id, session_id, platform, created_at, last_active_at,
                   message_count, participants, metadata
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return Session.from_dict(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "platform": row["platform"],
                "created_at": row["created_at"],
                "last_active_at": row["last_active_at"],
                "message_count": row["message_count"],
                "participants": row["participants"],
                "metadata": row["metadata"],
            }
        )

    async def update_session_activity(self, session_id: str) -> None:
        now = time.time()
        if self.connection is None:
            return
        async with self._write_lock:
            await self.connection.execute(
                """
                UPDATE sessions
                SET last_active_at = ?
                WHERE session_id = ?
                """,
                (now, session_id),
            )
            await self.connection.commit()

    async def get_recent_sessions(self, limit: int = 10) -> list[Session]:
        if self.connection is None:
            return []
        async with self.connection.execute(
            """
            SELECT id, session_id, platform, created_at, last_active_at,
                   message_count, participants, metadata
            FROM sessions
            ORDER BY last_active_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            Session.from_dict(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "platform": row["platform"],
                    "created_at": row["created_at"],
                    "last_active_at": row["last_active_at"],
                    "message_count": row["message_count"],
                    "participants": row["participants"],
                    "metadata": row["metadata"],
                }
            )
            for row in rows
        ]

    async def delete_session_messages(self, session_id: str) -> int:
        """清空会话消息（兼容旧版本数据；CM-only 下 LM 通常不写 messages）。"""
        if self.connection is None:
            return 0
        async with self._write_lock:
            cursor = await self.connection.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            deleted_count = cursor.rowcount
            await self.connection.execute(
                "UPDATE sessions SET message_count = 0 WHERE session_id = ?",
                (session_id,),
            )
            await self.connection.commit()
        if deleted_count:
            logger.info(
                f"{tag('store')} 删除会话消息: session={session_id}, count={deleted_count}"
            )
        return deleted_count
