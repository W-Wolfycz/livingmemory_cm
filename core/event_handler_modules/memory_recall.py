"""
记忆召回模块
负责长期记忆的检索和注入
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from ...log import log_ref, logger, tag, tag_event
from ..reflection import ReflectionService
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

from ..utils import (
    OperationContext,
    format_memories_for_fake_tool_call,
    format_memories_for_injection,
    get_cm_plugin,
    get_persona_id,
)

if TYPE_CHECKING:
    from ..base.config_manager import ConfigManager
    from ..managers.conversation_manager import ConversationManager
    from ..managers.memory_engine import MemoryEngine
    from ..utils.injection_adapter import InjectionAdapter
    from .message_utils import MessageUtils


class MemoryRecall:
    """记忆召回类"""

    def __init__(
        self,
        context,
        config_manager: "ConfigManager",
        memory_engine: "MemoryEngine",
        conversation_manager: "ConversationManager",
        message_utils: "MessageUtils",
        injection_adapter: "InjectionAdapter",
    ):
        """
        初始化记忆召回模块

        Args:
            context: AstrBot上下文
            config_manager: 配置管理器
            memory_engine: 记忆引擎
            conversation_manager: 会话管理器
            message_utils: 消息处理工具
            injection_adapter: 注入适配器
        """
        self.context = context
        self.config_manager = config_manager
        self.memory_engine = memory_engine
        self.conversation_manager = conversation_manager
        self.message_utils = message_utils
        self.injection_adapter = injection_adapter

    async def handle_memory_recall(
        self, event: AstrMessageEvent, req: ProviderRequest
    ):
        """Query and inject long-term memory before LLM request"""
        try:
            session_id = event.unified_msg_origin
            session_ref = log_ref(session_id, "session")
            logger.debug(f"{tag_event('recall', event)} 获取到会话引用: {session_ref}")

            # 检测异常session_id
            if session_id and (
                "Error:" in session_id or "error:" in session_id.lower()
            ):
                logger.warning(
                    f"{tag_event('recall', event)} [{session_ref}] 检测到异常会话引用，"
                    "这可能导致记忆功能异常。"
                )

            async with OperationContext("记忆召回", session_id):
                prompt_text = getattr(req, "prompt", "")
                extra_parts = getattr(req, "extra_user_content_parts", [])
                has_prompt_text = isinstance(prompt_text, str) and bool(
                    prompt_text.strip()
                )
                has_extra_parts = bool(extra_parts)

                if not has_prompt_text and not has_extra_parts:
                    logger.debug(f"{tag('recall')} [{session_ref}] 请求中无可用用户内容，跳过记忆召回")
                    return

                normalized = self._normalize_text_only_context_parts(req, session_id)
                if normalized > 0:
                    logger.debug(f"{tag('recall')} [{session_ref}] 已归一化 {normalized} 条纯文本历史消息")

                # 自动删除旧的注入记忆（恒开）
                # extra_user_content 路径通过 mark_as_temp(_no_save=True) 已天然不持久化，
                # 此处对它无副作用；user_message_*/fake_tool_call 路径需要主动清理。
                removed = self._remove_injected_memories_from_context(req, session_id)
                removed += self._remove_fake_tool_call_from_context(req, session_id)
                if removed > 0:
                    logger.debug(
                        f"{tag('recall')} [{session_ref}] 已清理 {removed} 处历史记忆注入片段"
                    )

                # 先提取用户消息（召回查询需要）
                actual_query = await self.message_utils.get_event_message_str(event)

                # 若 top_k <= 0，跳过记忆检索和注入，但上述清理已执行
                top_k = self.config_manager.get("recall_engine.top_k", 5)
                if top_k <= 0:
                    logger.debug(
                        f"{tag('recall')} [{session_ref}] top_k={top_k} <= 0，跳过记忆检索和注入"
                    )
                    return

                if not actual_query:
                    logger.warning(f"{tag('recall')} [{session_ref}] 原始用户消息为空，跳过记忆召回")
                    return

                # 获取 persona_id，与 AstrBot 主流程保持一致的三级优先级：
                # 1. session_service_config（最高）
                # 2. req.conversation.persona_id（会话级）
                # 3. 全局默认人格（最低）
                # 注意：on_llm_request 钩子在 _ensure_persona_and_skills 之前触发，
                # 因此不能直接依赖 req.system_prompt 已注入人格，需自行走完整优先级。
                persona_id = await get_persona_id(self.context, event)

                # 人格/会话过滤恒开（CM-only 单路径：CM 已隔离短期窗口，
                # LM 长期记忆同样按 session/persona 隔离，无需额外开关）
                recall_session_id = session_id
                recall_persona_id = persona_id

                # 当前发言始终是召回主体。历史只从 CM 公开 API 读取
                # 当前用户最近的完整 user/assistant 问答，用于指代消歧。
                query_for_search = actual_query
                try:
                    query_for_search = await self._build_recall_query(
                        event=event,
                        actual_query=actual_query,
                        persona_id=persona_id or "",
                    )
                except Exception as e:
                    logger.warning(
                        f"{tag('recall')} [{session_ref}] 构建 CM 用户历史召回查询失败，"
                        f"回退当前发言: {e}"
                    )

                # 执行记忆召回
                logger.debug(
                    f"{tag('recall')} [{session_ref}] 开始记忆召回，"
                    f"查询长度={len(query_for_search)}"
                )

                recalled_memories = await self.memory_engine.search_memories(
                    query=query_for_search,
                    k=self.config_manager.get("recall_engine.top_k", 5),
                    session_id=recall_session_id,
                    persona_id=recall_persona_id,
                )

                if recalled_memories:
                    logger.debug(
                        f"{tag('recall')} [{session_ref}] 检索到 {len(recalled_memories)} 条记忆"
                    )

                    # 格式化并注入记忆
                    memory_list = [
                        {
                            "id": getattr(mem, "doc_id", None),
                            "content": mem.content,
                            "score": mem.final_score,
                            "metadata": mem.metadata,
                            "timestamp": mem.metadata.get("create_time"),
                        }
                        for mem in recalled_memories
                    ]

                    # 输出详细记忆信息
                    for i, mem in enumerate(recalled_memories, 1):
                        logger.debug(
                            f"{tag('recall')} [{session_ref}] 记忆 #{i}: 得分={mem.final_score:.3f}, "
                            f"重要性={mem.metadata.get('importance', 0.5):.2f}, "
                            f"内容长度={len(mem.content or '')}"
                        )

                    # 根据配置选择注入方式（含 Provider 兼容降级）
                    configured_method = self.config_manager.get(
                        "recall_engine.injection_method", "extra_user_content"
                    )
                    provider = None
                    if configured_method in ("fake_tool_call",):
                        try:
                            provider = self.context.get_using_provider(session_id)
                        except Exception as e:
                            logger.warning(
                                f"{tag('recall')} [{session_ref}] 获取当前 Provider 失败，"
                                f"将按无 Provider 继续解析注入模式: {e}"
                            )
                    injection_method, fallback_reason = (
                        self.injection_adapter.resolve(provider, configured_method)
                    )
                    if fallback_reason:
                        logger.warning(
                        f"{tag('recall')} [{session_ref}] 注入模式从 {configured_method} 降级为 "
                            f"{injection_method}: {fallback_reason}"
                        )

                    memory_str = format_memories_for_injection(
                        memory_list,
                        max_memories=self.config_manager.get(
                            "recall_engine.injection_max_memories", 3
                        ),
                        max_chars=self.config_manager.get(
                            "recall_engine.injection_max_chars", 3200
                        ),
                    )
                    if not memory_str:
                        logger.debug(
                            f"{tag('recall')} [{session_ref}] 候选记忆去重/预算筛选后为空"
                        )
                        return

                    if injection_method == "user_message_before":
                        req.prompt = memory_str + "\n\n" + (req.prompt or "")
                        logger.info(
                            f"{tag('recall')} [{session_ref}] 成功向用户消息前注入 {len(recalled_memories)} 条记忆"
                        )
                    elif injection_method == "user_message_after":
                        req.prompt = (req.prompt or "") + "\n\n" + memory_str
                        logger.info(
                            f"{tag('recall')} [{session_ref}] 成功向用户消息后注入 {len(recalled_memories)} 条记忆"
                        )
                    elif injection_method == "fake_tool_call":
                        fake_messages = format_memories_for_fake_tool_call(
                            memory_list,
                            query=actual_query,
                            k=self.config_manager.get("recall_engine.top_k", 5),
                            session_filtered=True,
                            persona_filtered=True,
                        )
                        if fake_messages:
                            req.contexts.extend(fake_messages)
                            logger.info(
                                f"{tag('recall')} [{session_ref}] 成功以伪造工具调用方式注入 "
                                f"{len(recalled_memories)} 条记忆"
                            )
                    else:
                        # extra_user_content（推荐）：追加到用户消息末尾，
                        # 不影响前缀缓存且 mark_as_temp 后不污染对话历史
                        req.extra_user_content_parts.append(
                            TextPart(text=memory_str).mark_as_temp()
                        )
                        logger.info(
                            f"{tag('recall')} [{session_ref}] 成功向用户消息末尾注入 "
                            f"{len(recalled_memories)} 条记忆"
                        )
                else:
                    logger.debug(f"{tag('recall')} [{session_ref}] 未找到相关记忆")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"{tag('recall')} 处理 on_llm_request 钩子时发生错误: {e}", exc_info=True)

    async def _build_recall_query(
        self,
        *,
        event: AstrMessageEvent,
        actual_query: str,
        persona_id: str,
    ) -> str:
        """以当前发言为主体，附加当前用户最近的 CM 完整问答作为消歧上下文。"""
        rounds_limit = int(
            self.config_manager.get("recall_engine.query_context_rounds", 2) or 0
        )
        max_chars = int(
            self.config_manager.get("recall_engine.query_context_max_chars", 800)
            or 0
        )
        if rounds_limit <= 0 or max_chars <= 0:
            return actual_query

        cm_plugin = get_cm_plugin(self.context)
        if cm_plugin is None:
            return actual_query
        extraction_mode = ReflectionService.get_extraction_mode(cm_plugin)
        if extraction_mode == "rounds" and not hasattr(cm_plugin, "query_rounds"):
            return actual_query
        if extraction_mode == "messages" and not hasattr(cm_plugin, "query_history"):
            return actual_query

        umo = event.unified_msg_origin
        user_id = str(event.get_sender_id() or "")
        if not umo or not user_id:
            return actual_query
        cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
        if not cid:
            return actual_query

        max_age_seconds = max(
            0,
            int(
                self.config_manager.get(
                    "recall_engine.query_context_max_age_seconds",
                    0,
                )
                or 0
            ),
        )
        since = (
            datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
            if max_age_seconds > 0
            else None
        )

        # 召回历史单位跟随 CM llm_status_filter（与反思萃取语义一致）：
        # 仅 llm_success → 前 N 轮配对问答（query_rounds）；
        # 包含其他状态 → 前 N 条消息（query_history）。
        # 召回消歧只需要“该用户发言 + Bot 对该发言的完整回复”，
        # 不跟随 full_group/cross_session 的混合消息窗口。
        llm_status = getattr(cm_plugin, "ct_llm_status_filter", None)
        if extraction_mode == "rounds":
            rounds = await cm_plugin.query_rounds(
                umo=umo,
                conversation_id=cid,
                user_id=user_id,
                limit_rounds=rounds_limit,
                llm_status=llm_status,
                persona_id=persona_id,
                since=since,
            )
            units = rounds or []
        else:
            messages = await cm_plugin.query_history(
                umo=umo,
                conversation_id=cid,
                user_id=user_id,
                limit=rounds_limit,
                llm_status=llm_status,
                persona_id=persona_id,
                since=since,
            )
            units = [[message] for message in (messages or [])]
        if not units:
            return actual_query

        history_lines: list[str] = []
        used = 0
        for round_messages in reversed(units):
            round_lines: list[str] = []
            for message in round_messages:
                role = message.get("role")
                if role not in ("user", "assistant"):
                    continue
                text = str(message.get("content") or "").strip()
                if text:
                    label = "用户" if role == "user" else "助手"
                    round_lines.append(f"{label}：{text}")
            round_text = "\n".join(round_lines)
            if not round_text:
                continue
            separator_length = 1 if history_lines else 0
            remaining = max_chars - used - separator_length
            if remaining <= 0:
                break
            if len(round_text) > remaining:
                if remaining > 40:
                    history_lines.append(round_text[-remaining:])
                break
            history_lines.append(round_text)
            used += separator_length + len(round_text)

        if not history_lines:
            return actual_query
        history_lines.reverse()
        history = "\n".join(history_lines)
        unit_label = "轮" if extraction_mode == "rounds" else "条消息"
        logger.debug(
            f"{tag('recall')} [{log_ref(umo, 'umo')}] 召回查询包含当前发言 + "
            f"{min(len(units), rounds_limit)}{unit_label} 当前用户 CM 历史"
            f"（{len(history)}字符，模式={extraction_mode}）"
        )
        # 当前发言置于首尾各一次，明确其为 embedding 查询主体。
        return (
            f"当前用户发言：{actual_query}\n"
            f"最近相关问答（仅用于指代消歧）：\n{history}\n"
            f"需要检索的当前发言：{actual_query}"
        )

    def _remove_injected_memories_from_context(
        self, req: ProviderRequest, session_id: str
    ) -> int:
        """从请求上下文中移除临时注入的记忆片段"""
        import re
        from ..base.constants import MEMORY_INJECTION_FOOTER, MEMORY_INJECTION_HEADER

        removed = 0

        # 清理 system_prompt（兼容旧版本注入残留）
        if hasattr(req, "system_prompt") and req.system_prompt:
            if isinstance(req.system_prompt, str):
                original_prompt = req.system_prompt
                if (
                    MEMORY_INJECTION_HEADER in original_prompt
                    and MEMORY_INJECTION_FOOTER in original_prompt
                ):
                    # 使用正则清理记忆片段
                    pattern = re.compile(
                        re.escape(MEMORY_INJECTION_HEADER)
                        + r".*?"
                        + re.escape(MEMORY_INJECTION_FOOTER),
                        re.DOTALL,
                    )
                    cleaned_prompt = pattern.sub("", original_prompt)
                    cleaned_prompt = re.sub(r"\n{3,}", "\n\n", cleaned_prompt).strip()
                    req.system_prompt = cleaned_prompt
                    if cleaned_prompt != original_prompt:
                        removed += 1

        # 清理 extra_user_content_parts（通过 mark_as_temp/_no_save 标记）
        parts_before = len(getattr(req, "extra_user_content_parts", []))
        if parts_before > 0:
            req.extra_user_content_parts = [
                part
                for part in req.extra_user_content_parts
                if not self._is_livingmemory_temp_part(part)
            ]
            parts_after = len(req.extra_user_content_parts)
            removed += parts_before - parts_after

        return removed

    def _is_livingmemory_temp_part(self, part) -> bool:
        """判断是否为 LivingMemory 本轮临时注入的 extra_user_content part"""
        from ..base.constants import MEMORY_INJECTION_FOOTER, MEMORY_INJECTION_HEADER

        text = getattr(part, "text", "")
        return (
            getattr(part, "_no_save", False)
            and isinstance(text, str)
            and MEMORY_INJECTION_HEADER in text
            and MEMORY_INJECTION_FOOTER in text
        )

    def _normalize_text_only_context_parts(
        self, req: ProviderRequest, session_id: str
    ) -> int:
        """把历史中的纯文本 content parts 折叠回字符串，避免污染长期上下文格式"""
        contexts = getattr(req, "contexts", None)
        if not isinstance(contexts, list):
            return 0

        normalized = 0
        for msg in contexts:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list) or not content:
                continue

            text_parts = []
            text_only = True
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    text_only = False
                    break
                text_parts.append(str(part.get("text", "") or ""))

            if not text_only:
                continue

            msg["content"] = "".join(text_parts)
            normalized += 1

        if normalized:
            logger.debug(
                f"{tag('recall')} [{log_ref(session_id, 'session')}] 已归一化 "
                f"{normalized} 条纯文本历史 content parts"
            )
        return normalized

    def _remove_fake_tool_call_from_context(
        self, req: ProviderRequest, session_id: str
    ) -> int:
        """从请求上下文中移除伪造的工具调用记忆（fake_tool_call 注入方式）

        识别并移除以 FAKE_TOOL_CALL_ID_PREFIX 为 ID 前缀的
        assistant(tool_calls) + tool(result) 消息对。
        """
        from ..base.constants import FAKE_TOOL_CALL_ID_PREFIX

        if not hasattr(req, "contexts") or not req.contexts:
            return 0

        removed = 0
        indices_to_remove: set[int] = set()
        fake_call_ids: set[str] = set()

        try:
            # 单轮扫描：同时收集伪造 assistant(tool_calls) 和对应 tool(result) 消息
            for i, msg in enumerate(req.contexts):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tc_id = (
                            tc.get("id", "")
                            if isinstance(tc, dict)
                            else getattr(tc, "id", "")
                        )
                        if tc_id.startswith(FAKE_TOOL_CALL_ID_PREFIX):
                            fake_call_ids.add(tc_id)
                            indices_to_remove.add(i)
                elif role == "tool":
                    tc_id = msg.get("tool_call_id", "")
                    if tc_id in fake_call_ids:
                        indices_to_remove.add(i)

            # 从后往前删除，避免索引偏移
            for i in sorted(indices_to_remove, reverse=True):
                req.contexts.pop(i)
                removed += 1

        except Exception:
            pass

        return removed
