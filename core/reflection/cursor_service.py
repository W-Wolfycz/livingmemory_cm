"""反思复合游标的构造、比较、兼容迁移与持久化。"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ...log import log_ref, logger, tag

if TYPE_CHECKING:
    from ..managers.conversation_manager import ConversationManager


LEGACY_CURSOR_MAX_ID = 2**63 - 1


@dataclass(frozen=True)
class ReflectionCursor:
    """CM 严格 keyset 游标；时间戳相同时用自增记录 ID 消除歧义。"""

    created_at: str
    record_id: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "record_id": self.record_id,
        }


class ReflectionCursorService:
    """管理 `reflection_cursors_v3`，并同步 v2 时间戳兼容值。"""

    def __init__(self, conversation_manager: "ConversationManager") -> None:
        self.conversation_manager = conversation_manager

    @staticmethod
    def build_partition_key(
        conversation_id: str,
        persona_id: str,
        scope: str,
        mode_signature: str = "",
    ) -> str:
        raw = (
            f"{conversation_id}\x1f{persona_id}\x1f{scope}\x1f{mode_signature}"
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    @staticmethod
    def build_batch_id(
        cursor_key: str,
        start_cursor: ReflectionCursor,
        end_cursor: ReflectionCursor,
    ) -> str:
        raw = (
            f"{cursor_key}\x1f{start_cursor.created_at}\x1f{start_cursor.record_id}"
            f"\x1f{end_cursor.created_at}\x1f{end_cursor.record_id}"
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    @staticmethod
    def parse_utc_timestamp(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except (TypeError, ValueError):
            return None

    @classmethod
    def normalize_timestamp(cls, value: Any) -> str:
        parsed = cls.parse_utc_timestamp(value)
        if parsed is None:
            return ""
        return (
            parsed.replace(tzinfo=timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )

    @classmethod
    def from_metadata(
        cls,
        value: Any,
        *,
        legacy: bool = False,
    ) -> ReflectionCursor | None:
        if legacy:
            normalized = cls.normalize_timestamp(value)
            return (
                ReflectionCursor(normalized, LEGACY_CURSOR_MAX_ID)
                if normalized
                else None
            )
        if not isinstance(value, dict):
            return None
        normalized = cls.normalize_timestamp(value.get("created_at"))
        try:
            record_id = int(value.get("record_id"))
        except (TypeError, ValueError):
            return None
        if not normalized or record_id < 0:
            return None
        return ReflectionCursor(normalized, record_id)

    @classmethod
    def from_record(cls, record: dict) -> ReflectionCursor | None:
        normalized = cls.normalize_timestamp(
            record.get("created_at_utc") or record.get("created_at")
        )
        try:
            record_id = int(record.get("record_id"))
        except (TypeError, ValueError):
            return None
        if not normalized or record_id < 0:
            return None
        return ReflectionCursor(normalized, record_id)

    @classmethod
    def sort_key(cls, cursor: ReflectionCursor) -> tuple[datetime, int]:
        parsed = cls.parse_utc_timestamp(cursor.created_at)
        return (parsed or datetime.min, cursor.record_id)

    @classmethod
    def record_sort_key(cls, record: dict) -> tuple[datetime, int]:
        cursor = cls.from_record(record)
        if cursor is None:
            raise RuntimeError(
                "chat_memory 查询结果缺少 record_id，需升级到 1.1.1 或更高版本"
            )
        return cls.sort_key(cursor)

    @classmethod
    def latest_from_records(
        cls, records: list[dict]
    ) -> ReflectionCursor | None:
        cursors = [cls.from_record(record) for record in records]
        if any(cursor is None for cursor in cursors):
            raise RuntimeError(
                "chat_memory 查询结果缺少 record_id，需升级到 1.1.1 或更高版本"
            )
        valid = [cursor for cursor in cursors if cursor is not None]
        return max(valid, key=cls.sort_key, default=None)

    async def load(
        self,
        session_id: str,
        cursor_key: str,
    ) -> tuple[ReflectionCursor | None, bool]:
        """优先读取 v3 复合游标；缺失时兼容旧 v2 时间戳游标。"""
        v3_cursors = await self.conversation_manager.get_session_metadata(
            session_id, "reflection_cursors_v3", {}
        )
        if isinstance(v3_cursors, dict):
            cursor = self.from_metadata(v3_cursors.get(cursor_key))
            if cursor is not None:
                return cursor, False

        v2_cursors = await self.conversation_manager.get_session_metadata(
            session_id, "reflection_cursors_v2", {}
        )
        if isinstance(v2_cursors, dict):
            cursor = self.from_metadata(v2_cursors.get(cursor_key), legacy=True)
            if cursor is not None:
                return cursor, True
        return None, False

    async def store(
        self,
        session_id: str,
        cursor_key: str,
        cursor: ReflectionCursor,
    ) -> None:
        """先写 canonical v3，再尽力同步旧 v2 时间戳以保留回退兼容。"""
        v3_cursors = await self.conversation_manager.get_session_metadata(
            session_id, "reflection_cursors_v3", {}
        )
        if not isinstance(v3_cursors, dict):
            v3_cursors = {}
        else:
            v3_cursors = dict(v3_cursors)
        v3_cursors[cursor_key] = cursor.to_metadata()
        await self.conversation_manager.update_session_metadata(
            session_id, "reflection_cursors_v3", v3_cursors
        )

        try:
            v2_cursors = await self.conversation_manager.get_session_metadata(
                session_id, "reflection_cursors_v2", {}
            )
            if not isinstance(v2_cursors, dict):
                v2_cursors = {}
            else:
                v2_cursors = dict(v2_cursors)
            v2_cursors[cursor_key] = cursor.created_at
            await self.conversation_manager.update_session_metadata(
                session_id, "reflection_cursors_v2", v2_cursors
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                f"{tag('reflection')} [{log_ref(session_id, 'session')}] v3 游标已推进，但 v2 兼容游标同步失败",
                exc_info=True,
            )


__all__ = ["LEGACY_CURSOR_MAX_ID", "ReflectionCursor", "ReflectionCursorService"]
