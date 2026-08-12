"""ChatMemory 反思历史的严格游标读取与内容过滤。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .cursor_service import ReflectionCursor, ReflectionCursorService


class CMHistoryReader:
    """只读访问 ChatMemory，返回一个有限、严格有序的反思批次。"""

    @staticmethod
    def record_matches_content_filter(record: dict, cm_plugin: Any) -> bool:
        include_kinds = set(getattr(cm_plugin, "ct_include_kinds", set()) or set())
        if not include_kinds:
            return True
        kinds = set(record.get("content_kind") or [])
        if not kinds:
            return False
        if bool(getattr(cm_plugin, "ct_include_all_match", False)):
            return kinds.issubset(include_kinds)
        return bool(kinds & include_kinds)

    @staticmethod
    def content_kind_query(cm_plugin: Any) -> list[str] | None:
        """返回与 CM takeover 一致的内容类型白名单。"""
        include_kinds = sorted(
            str(item)
            for item in (getattr(cm_plugin, "ct_include_kinds", set()) or set())
        )
        return include_kinds or None

    async def query_latest_cursor(
        self,
        *,
        cm_plugin: Any,
        extraction_mode: str,
        umo: str,
        conversation_id: str,
        user_id: str | None,
        persona_id: str,
        llm_status: Any,
    ) -> ReflectionCursor | None:
        """新分区只读取最新一条/一轮建立基线，不扫描既有完整历史。"""
        if extraction_mode == "rounds":
            rounds = await cm_plugin.query_rounds(
                umo=umo,
                conversation_id=conversation_id,
                user_id=user_id,
                limit_rounds=1,
                llm_status=llm_status,
                persona_id=persona_id,
            )
            return ReflectionCursorService.latest_from_records(
                [message for rnd in rounds for message in rnd]
            )
        messages = await cm_plugin.query_history(
            umo=umo,
            conversation_id=conversation_id,
            user_id=user_id,
            limit=1,
            llm_status=llm_status,
            persona_id=persona_id,
        )
        return ReflectionCursorService.latest_from_records(messages)

    async def resolve_legacy_cursor_record_id(
        self,
        *,
        cm_plugin: Any,
        umo: str,
        conversation_id: str,
        user_id: str | None,
        persona_id: str,
        cursor: ReflectionCursor,
    ) -> ReflectionCursor | None:
        """用 CM 中同一时间戳的最大记录 ID 将旧 v2 时间戳游标精确化。"""
        timestamp = ReflectionCursorService.parse_utc_timestamp(cursor.created_at)
        if timestamp is None:
            return None
        try:
            messages = await cm_plugin.query_history(
                umo=umo,
                conversation_id=conversation_id,
                user_id=user_id,
                limit=1,
                persona_id=persona_id,
                since=timestamp,
                until=timestamp,
            )
        except TypeError as exc:
            if "record_id" in str(exc) or "since" in str(exc):
                raise RuntimeError(
                    "chat_memory 版本过旧，无法迁移复合游标；需升级到 1.1.1 或更高版本"
                ) from exc
            raise
        return ReflectionCursorService.latest_from_records(messages)

    async def query_rounds_paginated(
        self,
        *,
        cm_plugin: Any,
        umo: str,
        conversation_id: str,
        user_id: str | None,
        persona_id: str,
        llm_status: Any,
        cursor: ReflectionCursor,
        page_size: int,
        target_rounds: int,
        max_rounds: int = 2000,
    ) -> list[list[dict]]:
        """从游标后最旧完整轮次开始取得一个有限萃取批次。"""
        rounds: list[list[dict]] = []
        scanned = 0
        scan_limit = max(max_rounds, target_rounds)
        page_cursor = cursor
        content_kind = self.content_kind_query(cm_plugin)
        all_match = bool(getattr(cm_plugin, "ct_include_all_match", False))
        while len(rounds) < target_rounds and scanned < scan_limit:
            request_limit = min(1000, page_size, scan_limit - scanned)
            try:
                page = await cm_plugin.query_rounds(
                    umo=umo,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    limit_rounds=request_limit,
                    llm_status=llm_status,
                    content_kind=content_kind,
                    persona_id=persona_id,
                    since=ReflectionCursorService.parse_utc_timestamp(
                        page_cursor.created_at
                    ),
                    from_oldest=True,
                    after_id=page_cursor.record_id,
                    content_kind_all_match=all_match,
                )
            except TypeError as exc:
                if any(
                    name in str(exc)
                    for name in (
                        "from_oldest",
                        "after_id",
                        "content_kind_all_match",
                    )
                ):
                    raise RuntimeError(
                        "chat_memory 版本过旧，需升级到支持复合游标的 1.1.1 或更高版本"
                    ) from exc
                raise
            if not page:
                break
            scanned += len(page)
            if any(
                not rnd or not self.record_matches_content_filter(rnd[0], cm_plugin)
                for rnd in page
            ):
                raise RuntimeError("chat_memory 返回了不符合内容白名单的轮次")
            rounds.extend(page)
            next_cursor = ReflectionCursorService.latest_from_records(
                list(page[-1]) if page[-1] else []
            )
            if next_cursor is None or ReflectionCursorService.sort_key(
                next_cursor
            ) <= ReflectionCursorService.sort_key(page_cursor):
                raise RuntimeError("chat_memory 复合游标未向前推进")
            page_cursor = next_cursor
            if len(page) < request_limit:
                break
        if len(rounds) < target_rounds and scanned >= scan_limit:
            raise RuntimeError(
                f"CM 复合游标扫描达到安全上限 {scan_limit} 轮，未取得完整批次"
            )
        rounds.sort(
            key=lambda rnd: ReflectionCursorService.record_sort_key(rnd[0])
            if rnd
            else (datetime.min, -1)
        )
        return rounds[:target_rounds]

    async def query_messages_paginated(
        self,
        *,
        cm_plugin: Any,
        umo: str,
        conversation_id: str,
        user_id: str | None,
        persona_id: str,
        llm_status: Any,
        cursor: ReflectionCursor,
        page_size: int,
        target_messages: int,
        max_messages: int = 4000,
    ) -> list[dict]:
        """从游标后最旧混合消息开始取得一个有限萃取批次。"""
        messages: list[dict] = []
        scanned = 0
        scan_limit = max(max_messages, target_messages)
        page_cursor = cursor
        content_kind = self.content_kind_query(cm_plugin)
        all_match = bool(getattr(cm_plugin, "ct_include_all_match", False))
        while len(messages) < target_messages and scanned < scan_limit:
            request_limit = min(1000, page_size, scan_limit - scanned)
            try:
                page = await cm_plugin.query_history(
                    umo=umo,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    limit=request_limit,
                    llm_status=llm_status,
                    content_kind=content_kind,
                    persona_id=persona_id,
                    since=ReflectionCursorService.parse_utc_timestamp(
                        page_cursor.created_at
                    ),
                    from_oldest=True,
                    after_id=page_cursor.record_id,
                    content_kind_all_match=all_match,
                )
            except TypeError as exc:
                if any(
                    name in str(exc)
                    for name in (
                        "from_oldest",
                        "after_id",
                        "content_kind_all_match",
                    )
                ):
                    raise RuntimeError(
                        "chat_memory 版本过旧，需升级到支持复合游标的 1.1.1 或更高版本"
                    ) from exc
                raise
            if not page:
                break
            scanned += len(page)
            if any(
                not self.record_matches_content_filter(message, cm_plugin)
                for message in page
            ):
                raise RuntimeError("chat_memory 返回了不符合内容白名单的消息")
            messages.extend(page)
            next_cursor = ReflectionCursorService.latest_from_records(page)
            if next_cursor is None or ReflectionCursorService.sort_key(
                next_cursor
            ) <= ReflectionCursorService.sort_key(page_cursor):
                raise RuntimeError("chat_memory 复合游标未向前推进")
            page_cursor = next_cursor
            if len(page) < request_limit:
                break
        if len(messages) < target_messages and scanned >= scan_limit:
            raise RuntimeError(
                f"CM 复合游标扫描达到安全上限 {scan_limit} 条，未取得完整批次"
            )
        messages.sort(key=ReflectionCursorService.record_sort_key)
        return messages[:target_messages]


__all__ = ["CMHistoryReader"]
