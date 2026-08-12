"""反思批次的幂等写入与游标提交。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ...log import log_ref, logger, tag
from .cursor_service import ReflectionCursor, ReflectionCursorService
from .extraction_service import ReflectionExtractionService

if TYPE_CHECKING:
    from ..managers.conversation_manager import ConversationManager
    from ..managers.memory_engine import MemoryEngine


class ReflectionBatchWriter:
    """将一个确定 CM 窗口写成零至多条长期记忆，并在成功后提交游标。"""

    def __init__(
        self,
        memory_engine: "MemoryEngine",
        conversation_manager: "ConversationManager",
        cursor_service: ReflectionCursorService,
        extraction_service: ReflectionExtractionService,
    ) -> None:
        self.memory_engine = memory_engine
        self.conversation_manager = conversation_manager
        self.cursor_service = cursor_service
        self.extraction_service = extraction_service

    async def write(
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
        from ..utils import OperationContext

        async with OperationContext("CM 记忆存储", session_id):
            try:
                session_ref = log_ref(session_id, "session")
                current_cursor, _ = await self.cursor_service.load(
                    session_id, cursor_key
                )
                if current_cursor and self.cursor_service.sort_key(
                    end_cursor
                ) <= self.cursor_service.sort_key(current_cursor):
                    logger.debug(
                        f"{tag('reflection')} [{session_ref}] CM 任务过期："
                        f"end={end_cursor.created_at}#{end_cursor.record_id} "
                        f"<= current={current_cursor.created_at}#{current_cursor.record_id}，跳过"
                    )
                    return

                logger.debug(
                    f"{tag('reflection')} [{session_ref}] CM 模式开始处理记忆，"
                    f"范围=[{start_cursor.created_at}#{start_cursor.record_id} -> "
                    f"{end_cursor.created_at}#{end_cursor.record_id}], "
                    f"消息数={len(cm_messages)}, 当前人格={persona_id or '未设置'}"
                )

                if not self.memory_engine:
                    logger.error(
                        f"{tag('reflection')} [{session_ref}] CM 任务：MemoryEngine 未初始化，跳过"
                    )
                    return

                candidates = await self.extraction_service.extract(
                    session_id=session_id,
                    cm_messages=cm_messages,
                    persona_id=persona_id,
                    current_user_id=current_user_id,
                )
                if candidates is None:
                    return

                batch_id = self.cursor_service.build_batch_id(
                    cursor_key, start_cursor, end_cursor
                )
                batch_size = len(candidates)
                for batch_index, candidate in enumerate(candidates, 1):
                    metadata = dict(candidate.metadata)
                    metadata["source_window"] = {
                        "session_id": session_id,
                        "mode": "cm_takeover",
                        "start_ts": start_cursor.created_at,
                        "start_record_id": start_cursor.record_id,
                        "end_ts": end_cursor.created_at,
                        "end_record_id": end_cursor.record_id,
                        "message_count": len(cm_messages),
                        "batch_id": batch_id,
                        "batch_index": batch_index,
                        "batch_size": batch_size,
                    }
                    await self.memory_engine.add_memory(
                        content=candidate.content,
                        session_id=session_id,
                        persona_id=persona_id,
                        importance=candidate.importance,
                        metadata=metadata,
                        atoms=candidate.atoms,
                        idempotency_key=f"reflection:{batch_id}:{batch_index}",
                    )
                    logger.debug(
                        f"{tag('reflection')} [{session_ref}] 已存储批次记忆 "
                        f"{batch_index}/{batch_size}，batch={batch_id[:10]}，"
                        f"主题数={len(metadata.get('topics', []))}，"
                        f"重要性={candidate.importance:.2f}"
                    )
                logger.info(
                    f"{tag('reflection')} [{session_ref}] CM 模式完成萃取："
                    f"{len(cm_messages)}条消息生成 {batch_size} 条长期记忆，"
                    f"batch={batch_id[:10]}"
                )

                try:
                    await self.cursor_service.store(
                        session_id, cursor_key, end_cursor
                    )
                    logger.debug(
                        f"{tag('reflection')} [{session_ref}] CM 模式：推进分区游标 = "
                        f"{end_cursor.created_at}#{end_cursor.record_id}"
                    )
                except Exception as meta_err:
                    logger.error(
                        f"{tag('reflection')} [{session_ref}] CM 模式：记忆已存储但游标更新失败: {meta_err}。"
                        "下次重试将由批次幂等键避免重复写入。",
                        exc_info=True,
                    )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    f"{tag('reflection')} [{log_ref(session_id, 'session')}] CM 模式存储失败: {exc}",
                    exc_info=True,
                )

__all__ = ["ReflectionBatchWriter"]
