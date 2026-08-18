"""将 ChatMemory 记录转换为可持久化的长期记忆候选。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ...log import log_ref, logger, tag
from ..models.conversation_models import Message
from ..processors.memory_processor import LLMExtractionSkip

if TYPE_CHECKING:
    from ..processors.memory_processor import MemoryProcessor


@dataclass(frozen=True)
class ReflectionMemoryCandidate:
    """一次 LLM 萃取产生的单条记忆及其原子。"""

    content: str
    metadata: dict[str, Any]
    importance: float
    atoms: list[Any]


@dataclass(frozen=True)
class ReflectionExtractionSkip:
    """模型明确拒绝当前窗口；调用方按配置跳过有限 CM 单位。"""

    reason: str = ""


class ReflectionExtractionService:
    """负责 CM 消息身份转换、LLM 萃取和原子分类。"""

    def __init__(self, memory_processor: "MemoryProcessor") -> None:
        self.memory_processor = memory_processor

    async def extract(
        self,
        *,
        session_id: str,
        cm_messages: list[dict],
        persona_id: str,
        current_user_id: str,
    ) -> list[ReflectionMemoryCandidate] | ReflectionExtractionSkip | None:
        """区分成功、合法空结果、结构化跳过和可重试失败。"""
        session_ref = log_ref(session_id, "session")
        if not self.memory_processor:
            logger.error(
                f"{tag('reflection')} [{session_ref}] CM 任务：MemoryProcessor 未初始化，跳过"
            )
            return None

        is_group_chat = bool(
            cm_messages[0].get("group_id") if cm_messages else False
        )
        if not is_group_chat and "GroupMessage" in session_id:
            is_group_chat = True

        history_messages = [
            self.convert_cm_dict_to_message(message, session_id, current_user_id)
            for message in cm_messages
        ]
        try:
            generated_memories = (
                await self.memory_processor.process_conversation_batch(
                    messages=history_messages,
                    is_group_chat=is_group_chat,
                    persona_id=persona_id,
                )
            )
        except LLMExtractionSkip as exc:
            logger.warning(
                f"{tag('reflection')} [{session_ref}] CM 模式收到结构化跳过结果"
                + (f"，原因={exc.reason}" if exc.reason else "")
            )
            return ReflectionExtractionSkip(exc.reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"{tag('reflection')} [{session_ref}] CM 模式 LLM 处理失败: {exc}",
                exc_info=True,
            )
            return None

        candidates: list[ReflectionMemoryCandidate] = []
        for content, metadata, importance in generated_memories:
            normalized_metadata = dict(metadata or {})
            atoms = self.memory_processor.classify_atoms_from_metadata(
                metadata=normalized_metadata,
                parent_importance=importance,
                session_id=session_id,
                persona_id=persona_id,
            )
            if atoms:
                # 把本批原子的类型集合写入主记忆 metadata，让
                # recall_engine.memory_type_filter=event_only 对新反思记忆生效。
                # atoms 为空/None 时不写，不编造类型；旧记忆与 Agent 主动
                # memorize 工具（无 atoms）仍走 event_only 的保守兼容分支。
                normalized_metadata["atom_types"] = sorted(
                    {atom.atom_type.value for atom in atoms}
                )
            candidates.append(
                ReflectionMemoryCandidate(
                    content=content,
                    metadata=normalized_metadata,
                    importance=importance,
                    atoms=atoms,
                )
            )
        return candidates

    @staticmethod
    def convert_cm_dict_to_message(
        cm_dict: dict, session_id: str, current_user_id: str = ""
    ) -> Message:
        """把 CM 查询结果转成 LM Message，并保留真实发言者关系。"""
        created_at = cm_dict.get("created_at_utc") or cm_dict.get("created_at")
        timestamp = 0.0
        if created_at:
            try:
                parsed = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                timestamp = parsed.timestamp()
            except (ValueError, TypeError):
                timestamp = 0.0

        is_bot = cm_dict.get("role") == "assistant"
        if is_bot:
            sender_id = str(cm_dict.get("self_id") or "bot")
            # CM 最新结构把 Bot 昵称存在 assistant 行的 sender_nickname；
            # bot_nickname 仅作为旧版 CM 兼容回退。
            sender_name = (
                cm_dict.get("sender_nickname")
                or cm_dict.get("bot_nickname")
                or "Bot"
            )
            speaker_relation = "bot"
        else:
            sender_id = str(
                cm_dict.get("user_id") or cm_dict.get("sender_id") or ""
            )
            sender_name = (
                cm_dict.get("sender_nickname")
                or cm_dict.get("sender_name")
                or sender_id
                or None
            )
            speaker_relation = (
                "current_user"
                if current_user_id and sender_id == current_user_id
                else "other_user"
            )

        return Message(
            id=0,
            session_id=session_id,
            role=cm_dict.get("role") or "user",
            content=cm_dict.get("content") or "",
            sender_id=sender_id,
            sender_name=sender_name,
            group_id=cm_dict.get("group_id"),
            platform=cm_dict.get("platform_name") or cm_dict.get("platform_id"),
            timestamp=timestamp,
            metadata={
                "is_bot_message": is_bot,
                "speaker_relation": speaker_relation,
                "turn_id": cm_dict.get("turn_id"),
            },
        )


__all__ = [
    "ReflectionExtractionService",
    "ReflectionExtractionSkip",
    "ReflectionMemoryCandidate",
]
