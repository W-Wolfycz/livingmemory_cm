"""ChatMemory 反思窗口的查询、分区与后台任务编排。"""

from __future__ import annotations

import asyncio
from typing import Any

from ...log import log_ref, logger, tag
from ..utils import get_cm_plugin, get_persona_id
from .batch_writer import ReflectionBatchWriter
from .cm_history_reader import CMHistoryReader
from .cursor_service import ReflectionCursor, ReflectionCursorService


class ReflectionService:
    """协调 CM 历史读取、复合游标和单会话批次写入。"""

    def __init__(
        self,
        *,
        context: Any,
        conversation_manager: Any,
        cursor_service: ReflectionCursorService,
        history_reader: CMHistoryReader,
        batch_writer: ReflectionBatchWriter,
        storage_tasks: set[asyncio.Task],
        storage_sessions_inflight: set[str],
        storage_state_lock: asyncio.Lock,
    ) -> None:
        self.context = context
        self.conversation_manager = conversation_manager
        self.cursor_service = cursor_service
        self.history_reader = history_reader
        self.batch_writer = batch_writer
        self.storage_tasks = storage_tasks
        self.storage_sessions_inflight = storage_sessions_inflight
        self.storage_state_lock = storage_state_lock
        self.shutting_down = False

    @staticmethod
    def get_extraction_mode(cm_plugin: Any) -> str:
        """完全跟随 CM：仅 llm_success 为配对轮，其余配置均为混合消息。"""
        statuses = set(getattr(cm_plugin, "ct_llm_status_filter", []) or [])
        return "rounds" if statuses == {"llm_success"} else "messages"

    async def dispatch(
        self,
        *,
        event: Any,
        session_id: str,
        trigger_count: int,
        cm_limit: int,
        extraction_mode: str,
    ) -> None:
        """查询游标后的最旧完整批次，达到阈值后派发幂等写入任务。"""
        cm_plugin = get_cm_plugin(self.context)
        session_ref = log_ref(session_id, "session")
        if cm_plugin is None:
            logger.debug(f"{tag('reflection')} [{session_ref}] CM 实例不可用，跳过萃取")
            return

        umo = session_id
        session_task = asyncio.create_task(
            self.conversation_manager.create_or_get_session(
                session_id, platform="unknown"
            )
        )
        cid_task = asyncio.create_task(
            self.context.conversation_manager.get_curr_conversation_id(umo)
        )
        try:
            await session_task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                f"{tag('reflection')} [{session_ref}] CM 路径：建会话失败: {exc}"
            )
        try:
            cid = await cid_task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                f"{tag('reflection')} [{session_ref}] CM 路径：读取 conversation_id 失败: {exc}"
            )
            cid = None
        if not cid:
            logger.debug(
                f"{tag('reflection')} [{session_ref}] CM 路径：无当前 conversation_id，跳过萃取"
            )
            return

        persona_id = await get_persona_id(self.context, event)
        is_group_chat = bool(
            getattr(
                cm_plugin,
                "_is_group_umo",
                lambda _umo: "GroupMessage" in _umo,
            )(umo)
        )
        full_group = bool(
            getattr(cm_plugin, "ct_full_group", False) and is_group_chat
        )
        current_user_id = str(event.get_sender_id() or "")
        query_user_id = None if full_group else current_user_id

        status_signature = ",".join(
            sorted(
                str(item)
                for item in (
                    getattr(cm_plugin, "ct_llm_status_filter", []) or []
                )
            )
        )
        kinds_signature = ",".join(
            sorted(
                str(item)
                for item in (
                    getattr(cm_plugin, "ct_include_kinds", set()) or set()
                )
            )
        )
        cursor_key = self.cursor_service.build_partition_key(
            conversation_id=str(cid),
            persona_id=persona_id,
            scope="full_group" if full_group else f"user:{current_user_id}",
            mode_signature=(
                f"{extraction_mode}|statuses={status_signature}|"
                f"kinds={kinds_signature}|"
                f"all={bool(getattr(cm_plugin, 'ct_include_all_match', False))}"
            ),
        )
        last_cursor, migrated_from_v2 = await self.cursor_service.load(
            session_id, cursor_key
        )

        llm_status = getattr(cm_plugin, "ct_llm_status_filter", None)
        if last_cursor is None:
            try:
                new_baseline = await self.history_reader.query_latest_cursor(
                    cm_plugin=cm_plugin,
                    extraction_mode=extraction_mode,
                    umo=umo,
                    conversation_id=str(cid),
                    user_id=query_user_id,
                    persona_id=persona_id,
                    llm_status=llm_status,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"{tag('reflection')} [{session_ref}] CM 路径：建立游标基线失败: {exc}"
                )
                return
            if new_baseline:
                await self.cursor_service.store(
                    session_id, cursor_key, new_baseline
                )
                logger.info(
                    f"{tag('reflection')} [{session_ref}] CM 路径："
                    "为 CID/persona/scope 建立基线，跳过旧历史"
                )
            return

        if migrated_from_v2:
            resolved_cursor = (
                await self.history_reader.resolve_legacy_cursor_record_id(
                    cm_plugin=cm_plugin,
                    umo=umo,
                    conversation_id=str(cid),
                    user_id=query_user_id,
                    persona_id=persona_id,
                    cursor=last_cursor,
                )
            )
            if resolved_cursor is not None:
                last_cursor = resolved_cursor
            await self.cursor_service.store(session_id, cursor_key, last_cursor)
            logger.info(
                f"{tag('reflection')} [{session_ref}] CM 路径："
                "已将旧时间戳游标迁移为复合游标"
            )

        try:
            if extraction_mode == "rounds":
                units = await self.history_reader.query_rounds_paginated(
                    cm_plugin=cm_plugin,
                    umo=umo,
                    conversation_id=str(cid),
                    user_id=query_user_id,
                    persona_id=persona_id,
                    llm_status=llm_status,
                    cursor=last_cursor,
                    page_size=max(50, trigger_count * 2, cm_limit * 2),
                    target_rounds=trigger_count,
                )
            else:
                units = await self.history_reader.query_messages_paginated(
                    cm_plugin=cm_plugin,
                    umo=umo,
                    conversation_id=str(cid),
                    user_id=query_user_id,
                    persona_id=persona_id,
                    llm_status=llm_status,
                    cursor=last_cursor,
                    page_size=max(100, trigger_count * 2, cm_limit * 2),
                    target_messages=trigger_count,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                f"{tag('reflection')} [{session_ref}] CM 路径："
                f"{extraction_mode} 查询失败: {exc}"
            )
            return

        available_count = len(units)
        unit = "轮" if extraction_mode == "rounds" else "条消息"
        logger.debug(
            f"{tag('reflection')} [{session_ref}] CM 路径：游标后可用 "
            f"{available_count}{unit}，触发阈值 {trigger_count}{unit}"
        )
        if available_count < trigger_count:
            return

        selected_units = units[:trigger_count]
        messages = (
            [message for round_messages in selected_units for message in round_messages]
            if extraction_mode == "rounds"
            else selected_units
        )
        end_cursor = self.cursor_service.latest_from_records(messages)
        if end_cursor is None or self.shutting_down:
            return

        async with self.storage_state_lock:
            if session_id in self.storage_sessions_inflight:
                logger.debug(
                    f"{tag('reflection')} [{session_ref}] CM 路径："
                    "已有记忆反思任务在执行，跳过本次触发"
                )
                return
            self.storage_sessions_inflight.add(session_id)

        try:
            task = asyncio.create_task(
                self._write_batch(
                    session_id=session_id,
                    cm_messages=messages,
                    persona_id=persona_id,
                    start_cursor=last_cursor,
                    end_cursor=end_cursor,
                    cursor_key=cursor_key,
                    current_user_id=current_user_id,
                )
            )
        except Exception:
            self.storage_sessions_inflight.discard(session_id)
            raise

        self.storage_tasks.add(task)
        task.add_done_callback(
            lambda completed, sid=session_id: self._on_storage_task_done(
                completed, sid
            )
        )

    async def _write_batch(
        self,
        *,
        session_id: str,
        cm_messages: list[dict],
        persona_id: str,
        start_cursor: ReflectionCursor,
        end_cursor: ReflectionCursor,
        cursor_key: str,
        current_user_id: str,
    ) -> None:
        await self.batch_writer.write(
            session_id=session_id,
            cm_messages=cm_messages,
            persona_id=persona_id,
            start_cursor=start_cursor,
            end_cursor=end_cursor,
            cursor_key=cursor_key,
            current_user_id=current_user_id,
        )

    def _on_storage_task_done(
        self, task: asyncio.Task, session_id: str
    ) -> None:
        self.storage_tasks.discard(task)
        self.storage_sessions_inflight.discard(session_id)
        if task.cancelled():
            logger.debug(f"{tag('reflection')} [{log_ref(session_id, 'session')}] 存储任务已取消")
            return
        exc = task.exception()
        if exc:
            logger.error(
                f"{tag('reflection')} [{log_ref(session_id, 'session')}] 存储任务异常: {exc}",
                exc_info=exc,
            )

    def set_shutting_down(self, value: bool) -> None:
        self.shutting_down = value


__all__ = ["ReflectionService"]
