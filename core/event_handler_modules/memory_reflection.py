"""AstrBot 反思 Hook：校验 CM 状态并委托反思领域服务。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from astrbot.api.event import AstrMessageEvent

from ...log import logger, tag
from ..reflection import (
    CMHistoryReader,
    ReflectionBatchWriter,
    ReflectionCursorService,
    ReflectionExtractionService,
    ReflectionService,
)
from ..utils import get_cm_plugin, get_cm_status

if TYPE_CHECKING:
    from ..base.config_manager import ConfigManager
    from ..managers.conversation_manager import ConversationManager
    from ..managers.memory_engine import MemoryEngine
    from ..processors.memory_processor import MemoryProcessor
    from .message_utils import MessageUtils


class MemoryReflection:
    """接收 AstrBot 事件并把 CM 反思交给领域服务。"""

    def __init__(
        self,
        context: Any,
        config_manager: "ConfigManager",
        memory_engine: "MemoryEngine",
        memory_processor: "MemoryProcessor",
        conversation_manager: "ConversationManager",
        message_utils: "MessageUtils",
        storage_tasks: set[asyncio.Task],
        storage_sessions_inflight: set[str],
        storage_state_lock: asyncio.Lock,
    ) -> None:
        self.context = context
        self.config_manager = config_manager
        self.message_utils = message_utils

        cursor_service = ReflectionCursorService(conversation_manager)
        extraction_service = ReflectionExtractionService(memory_processor)
        batch_writer = ReflectionBatchWriter(
            memory_engine,
            conversation_manager,
            cursor_service,
            extraction_service,
        )
        self._reflection_service = ReflectionService(
            context=context,
            conversation_manager=conversation_manager,
            cursor_service=cursor_service,
            history_reader=CMHistoryReader(),
            batch_writer=batch_writer,
            storage_tasks=storage_tasks,
            storage_sessions_inflight=storage_sessions_inflight,
            storage_state_lock=storage_state_lock,
        )

    async def handle_memory_reflection(self, event: AstrMessageEvent) -> None:
        """在 CM 写入 prepared assistant 后检查并派发反思批次。"""
        logger.debug(
            f"{tag('reflection')} 进入 handle_memory_reflection"
            "（on_decorating_result）"
        )
        try:
            session_id = event.unified_msg_origin
            if not session_id:
                logger.warning(f"{tag('reflection')} session_id 为空，跳过反思")
                return

            if "error:" in session_id.lower():
                logger.warning(
                    f"{tag('reflection')} 检测到异常 session_id，跳过详情记录"
                )

            cm_on, cm_limit = get_cm_status(self.context)
            if not cm_on or cm_limit <= 0:
                logger.warning(
                    f"{tag('reflection')} CM 未接管（异常状态），跳过反思。"
                    f"ct_enable={cm_on}, ct_limit_rounds={cm_limit}"
                )
                return

            extraction_mode = ReflectionService.get_extraction_mode(
                get_cm_plugin(self.context)
            )
            configured = int(
                self.config_manager.get(
                    "reflection_engine.trigger_count", 0
                )
                or 0
            )
            trigger_count = configured if configured > 0 else cm_limit
            unit = "轮" if extraction_mode == "rounds" else "条消息"
            logger.debug(
                f"{tag('reflection')} 反思模式={extraction_mode}, "
                f"触发阈值={trigger_count}{unit}"
            )

            await self._reflection_service.dispatch(
                event=event,
                session_id=session_id,
                trigger_count=trigger_count,
                cm_limit=cm_limit,
                extraction_mode=extraction_mode,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"{tag('reflection')} 处理记忆反思时发生错误: {exc}",
                exc_info=True,
            )

    def set_shutting_down(self, value: bool) -> None:
        """阻止关闭过程继续派发新的反思任务。"""
        self._reflection_service.set_shutting_down(value)
