"""
记忆处理器 - 使用LLM将对话历史处理为结构化记忆
"""

import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ...log import logger, tag

from ..models.conversation_models import Message
from ..models.memory_atom import MemoryAtom
from .atom_classifier import classify_atoms


class LLMExtractionSkip(Exception):
    """模型按 JSON 协议明确表示本批内容无法处理。"""

    def __init__(self, reason: str = "") -> None:
        self.reason = str(reason or "").strip()[:120]
        super().__init__(self.reason or "LLM requested structured skip")


class MemoryProcessor:
    """
    记忆处理器

    使用LLM将对话历史转换为结构化记忆。
    支持私聊和群聊两种场景的不同处理策略。
    """

    def __init__(
        self,
        context=None,
        llm_provider: Any = None,
        config: dict[str, Any] | None = None,
    ):
        """
        初始化记忆处理器

        Args:
            context: AstrBot 上下文，用于动态解析 LLM Provider；不读取 Persona Prompt
            llm_provider: LLM Provider 实例或 Provider ID 字符串。
                          传入实例时直接使用（测试用）；传入字符串时动态解析。
                          留空则使用AstrBot默认Provider。
            config: 记忆处理器配置。
        """
        self.context = context
        self._llm_provider = llm_provider
        self.config = config or {}

        # 加载提示词模板
        self._load_prompts()

    def _get_current_llm_provider(self):
        """动态解析LLM Provider以避免持有过期引用

        AstrBot可能在运行期间重新创建Provider实例（例如配置变更后），
        旧的Provider实例内部的httpx client会被关闭，导致
        RuntimeError: Cannot send a request, as the client has been closed.
        因此每次调用前都从AstrBot上下文重新获取当前有效的Provider。
        """
        if not self.context:
            # 无 context 时直接返回传入的 provider 实例（测试路径）
            if self._llm_provider is not None and not isinstance(
                self._llm_provider, str
            ):
                return self._llm_provider
            return None

        # 如果传入的是 provider 实例（非字符串），直接使用（测试路径）
        if self._llm_provider is not None and not isinstance(self._llm_provider, str):
            return self._llm_provider

        # 优先使用配置中指定的Provider ID（字符串）
        if isinstance(self._llm_provider, str) and self._llm_provider:
            try:
                provider = self.context.get_provider_by_id(self._llm_provider)
                if provider:
                    return provider
            except Exception:
                pass

        # 回退到AstrBot当前默认Provider
        try:
            provider = self.context.get_using_provider()
            if provider:
                return provider
        except Exception:
            pass

        return None

    def _load_prompts(self) -> None:
        """从外部文件加载提示词模板"""
        prompt_dir = Path(__file__).parent.parent / "prompts"

        try:
            # 加载私聊提示词
            private_prompt_file = prompt_dir / "private_chat_prompt.txt"
            with open(private_prompt_file, encoding="utf-8") as f:
                self.private_chat_prompt = f.read()

            # 加载群聊提示词
            group_prompt_file = prompt_dir / "group_chat_prompt.txt"
            with open(group_prompt_file, encoding="utf-8") as f:
                self.group_chat_prompt = f.read()

            logger.info(f"{tag('processor')} 提示词模板加载成功")

        except Exception as e:
            logger.error(f"{tag('processor')} 加载提示词模板失败: {e}")
            # 使用简单的后备提示词（注意：使用 replace 替换，无需转义大括号）
            self.private_chat_prompt = """分析以下对话并生成JSON格式的长期记忆:
{conversation}

输出格式:
{"status": "success", "memories": [{"summary": "中性摘要", "topics": ["主题"], "key_facts": ["事实"], "event_time": "", "sentiment": "neutral", "importance": 0.5}]}
"""
            self.group_chat_prompt = """分析以下群聊对话并生成JSON格式的长期记忆:
{conversation}

输出格式:
{"status": "success", "memories": [{"summary": "中性摘要", "topics": ["主题"], "key_facts": ["事实"], "participants": ["参与者"], "event_time": "", "sentiment": "neutral", "importance": 0.5}]}
"""

    async def _build_system_prompt_with_persona(self, persona_id: str | None) -> str:
        """构建中性的事实萃取提示词。

        ``persona_id`` 只用于存储和召回分区。保留参数与方法名是为了兼容旧
        调用方；萃取阶段不读取 Persona Prompt，也不模仿角色语气。
        """
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        if persona_id:
            logger.debug(
                f"{tag('processor')} persona_id 仅用于分区，不注入萃取提示词"
            )
        return (
            "你是长期记忆事实萃取器。只能输出一个标准 JSON 对象，禁止输出 Markdown、"
            "解释、道歉或其他自然语言。\n"
            f"当前日期时间: {current_date}\n"
            "只提取对未来有用、能够由对话支持的事实。使用中性、简洁、第三人称或"
            "直接事实表述；不得模仿任何角色人格、语气或文风，不得文学化补写。\n"
            "必须区分消息前缀中的具体发言者和 Bot，不得把不同人的行为合并。\n"
            "将今天、明天、昨天、下周等相对时间转换为可长期理解的绝对日期。\n"
            "一次输入可按主题或连续事件拆成 0 至 5 条记忆。正常完成时输出"
            " {\"status\": \"success\", \"memories\": [...]}；没有持久价值时输出"
            " {\"status\": \"success\", \"memories\": []}；如果内容安全或政策原因"
            "导致无法处理，也必须输出"
            " {\"status\": \"skip\", \"reason\": \"简短原因\", \"memories\": []}。"
        )

    @staticmethod
    def _exception_chain(error: BaseException) -> list[BaseException]:
        """有限展开异常链，兼容 Provider 对底层 HTTP 错误的包装。"""
        chain: list[BaseException] = []
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen and len(chain) < 5:
            seen.add(id(current))
            chain.append(current)
            current = current.__cause__ or current.__context__
        return chain

    @classmethod
    def _is_content_policy_rejection(cls, error: BaseException) -> bool:
        """只识别高置信度的 Provider 内容安全拒绝，不匹配普通 4xx。"""
        statuses: set[int] = set()
        text_parts: list[str] = []

        for current in cls._exception_chain(error):
            try:
                text_parts.append(str(current))
            except Exception:
                pass

            for attribute in ("message", "code", "type", "body"):
                try:
                    value = getattr(current, attribute, None)
                except Exception:
                    value = None
                if value is not None:
                    try:
                        text_parts.append(
                            json.dumps(value, ensure_ascii=False, default=str)
                            if isinstance(value, (dict, list, tuple))
                            else str(value)
                        )
                    except Exception:
                        pass

            try:
                status_code = getattr(current, "status_code", None)
                response = getattr(current, "response", None)
                if status_code is None and response is not None:
                    status_code = getattr(response, "status_code", None)
                if status_code is not None:
                    statuses.add(int(status_code))
            except (TypeError, ValueError, AttributeError):
                pass

        # 已知是其他状态码时不因偶然包含敏感词而误判；部分 Provider 包装会丢失状态码，
        # 此时仍允许下方高置信度错误签名命中。
        if statuses and not statuses.intersection({400, 403, 422}):
            return False

        normalized = " ".join(text_parts).casefold()
        signatures = (
            "input data may contain inappropriate content",
            "content_policy_violation",
            "content policy violation",
            "responsibleaipolicyviolation",
            "request was rejected as a result of the content filter",
            "blocked due to safety",
            "prompt was blocked for safety",
            "输入数据可能包含不当内容",
            "输入内容可能包含不适当内容",
            "api 返回的 completion 由于内容安全过滤被拒绝",
            "内容安全过滤被拒绝",
        )
        return any(signature in normalized for signature in signatures)

    async def _call_llm_with_retry(
        self,
        prompt: str,
        system_prompt: str,
        max_retries: int = 3,
        response_validator: Callable[[str], Any] | None = None,
    ) -> str:
        """
        带指数退避的 LLM 调用

        Args:
            prompt: 提示词
            system_prompt: 系统提示词
            max_retries: 最大重试次数

        Returns:
            LLM 响应文本
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                provider = self._get_current_llm_provider()
                if not provider:
                    raise RuntimeError("LLM Provider 不可用")
                response = await provider.text_chat(
                    prompt=prompt, system_prompt=system_prompt
                )
                completion_text = str(response.completion_text or "")
                if response_validator is not None:
                    response_validator(completion_text)
                return completion_text
            except LLMExtractionSkip:
                raise
            except Exception as e:
                if self._is_content_policy_rejection(e):
                    raise LLMExtractionSkip("provider_content_policy") from e
                last_error = e
                if attempt == max_retries - 1:
                    raise
                wait_time = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"{tag('processor')} LLM 调用或 JSON 校验失败，{wait_time:.1f}s 后重试 "
                    f"({attempt + 1}/{max_retries}): {e}"
                )
                await asyncio.sleep(wait_time)
        if last_error:
            raise last_error
        raise RuntimeError("LLM 调用失败，未捕获到具体异常")

    async def process_conversation(
        self,
        messages: list[Message],
        is_group_chat: bool = False,
        persona_id: str | None = None,
    ) -> tuple[str, dict[str, Any], float]:
        """
        兼容旧调用：处理对话历史并返回第一条结构化记忆。

        Args:
            messages: 消息列表(Message对象)
            is_group_chat: 是否为群聊
            persona_id: 人格分区 ID，不注入萃取提示词

        Returns:
            tuple: (content, metadata, importance)
                - content: 格式化的记忆内容字符串
                - metadata: 包含结构化信息的字典
                - importance: 重要性评分(0-1)

        Raises:
            Exception: 处理失败时抛出异常
        """
        memories = await self.process_conversation_batch(
            messages=messages,
            is_group_chat=is_group_chat,
            persona_id=persona_id,
        )
        if not memories:
            raise ValueError("本批对话没有可持久化的长期记忆")
        return memories[0]

    async def process_conversation_batch(
        self,
        messages: list[Message],
        is_group_chat: bool = False,
        persona_id: str | None = None,
    ) -> list[tuple[str, dict[str, Any], float]]:
        """一次 LLM 萃取生成 0 至 5 条主题记忆，每条可包含多条事实。"""
        if not messages:
            raise ValueError("消息列表不能为空")

        conversation_text = self._format_conversation(messages)
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        template = self.group_chat_prompt if is_group_chat else self.private_chat_prompt
        prompt = template.replace("{conversation}", conversation_text).replace(
            "{current_date}", current_date
        )
        conversation_type = "群聊" if is_group_chat else "私聊"

        try:
            logger.debug(
                f"{tag('processor')} 准备调用 LLM，对话类型={conversation_type}, "
                f"消息数={len(messages)}"
            )
            logger.debug(
                f"{tag('processor')} Prompt 长度={len(prompt)}，"
                f"对话文本长度={len(conversation_text)}"
            )

            system_prompt = await self._build_system_prompt_with_persona(persona_id)
            logger.debug(
                f"{tag('processor')} System Prompt 已生成，长度={len(system_prompt)}"
            )
            llm_response_text = await self._call_llm_with_retry(
                prompt=prompt,
                system_prompt=system_prompt,
                response_validator=lambda text: self._parse_llm_response_batch(
                    text, is_group_chat
                ),
            )
            logger.debug(
                f"{tag('processor')} LLM 响应成功，响应长度={len(llm_response_text)}"
            )

            structured_memories = self._parse_llm_response_batch(
                llm_response_text, is_group_chat
            )
            participant_identities = self._extract_participant_identities(messages)
            results: list[tuple[str, dict[str, Any], float]] = []
            for index, structured_data in enumerate(structured_memories, 1):
                quality = self._validate_summary_quality(structured_data)
                if quality == "low":
                    logger.warning(
                        f"{tag('processor')} 第 {index} 条记忆质量不达标（low），"
                        "将标记但仍写入"
                    )
                structured_data["_quality"] = quality
                content, metadata = self._build_storage_format(
                    conversation_text, structured_data, is_group_chat
                )
                metadata["participant_identities"] = participant_identities
                metadata["summary_quality"] = quality
                importance = float(structured_data.get("importance", 0.5))
                results.append((content, metadata, importance))

            logger.debug(
                f"{tag('processor')} {conversation_type}批次生成 {len(results)} 条长期记忆"
            )
            return results
        except LLMExtractionSkip:
            raise
        except Exception as e:
            logger.error(f"{tag('processor')} 处理对话历史失败: {e}", exc_info=True)
            raise

    def _format_conversation(self, messages: list[Message]) -> str:
        """
        格式化对话历史为文本

        Args:
            messages: 消息列表(Message对象)

        Returns:
            格式化后的对话文本
        """

        formatted_lines = []
        for i, msg in enumerate(messages):
            content_text = self._message_content_to_text(msg.content)
            logger.debug(
                f"{tag('processor')} [_format_conversation] 消息#{i}: "
                f"role={msg.role}, group={bool(msg.group_id)}, "
                f"content_length={len(content_text)}"
            )

            sender_info = self._format_sender_info(msg)
            formatted_line = f"{sender_info} {content_text}".rstrip()
            formatted_lines.append(formatted_line)
        return "\n".join(formatted_lines)

    @staticmethod
    def _extract_participant_identities(
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Build stable graph identities from CM sender IDs and aliases."""
        identities: dict[str, dict[str, Any]] = {}
        for message in messages:
            if message.role == "system":
                continue
            sender_id = str(message.sender_id or "").strip()
            if not sender_id:
                continue
            platform = str(message.platform or "unknown").strip().lower() or "unknown"
            identity_key = f"{platform}:{sender_id}"
            display_name = str(message.sender_name or sender_id).strip() or sender_id
            is_bot = bool(
                message.metadata.get("is_bot_message", False)
                or message.role == "assistant"
            )
            identity = identities.setdefault(
                identity_key,
                {
                    "identity_key": identity_key,
                    "sender_id": sender_id,
                    "platform": platform,
                    "display_name": display_name,
                    "aliases": [],
                    "is_bot": is_bot,
                },
            )
            identity["display_name"] = display_name
            identity["is_bot"] = bool(identity["is_bot"] or is_bot)
            if display_name not in identity["aliases"]:
                identity["aliases"].append(display_name)
        return list(identities.values())

    @staticmethod
    def _format_sender_info(msg: Message) -> str:
        """按 CM 最新上下文结构生成消息前缀（``<cm_time>/<cm_speaker>/<cm_nickname>``）。

        与 chat_memory 注入格式对齐：昵称缺失时输出 ``?``，不回退账号 ID（CM 隐私
        承诺：ID 不进入 LLM 上下文）。Bot 发言额外用 ``<cm_speaker bot="1"/>`` 标记，
        因为萃取文本没有 role 维度，需要显式区分 Bot 与用户。
        """
        from xml.sax.saxutils import escape

        time_str = datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        display_name = escape(
            (msg.sender_name if msg.sender_name else "?").strip() or "?"
        )
        is_bot = msg.metadata.get("is_bot_message", False) or msg.role == "assistant"
        if is_bot:
            speaker_tag = '<cm_speaker bot="1"/>'
        else:
            relation = msg.metadata.get("speaker_relation")
            if msg.group_id and relation == "current_user":
                speaker_tag = '<cm_speaker current="1"/>'
            elif msg.group_id and relation == "other_user":
                speaker_tag = "<cm_speaker/>"
            else:
                # 私聊：与 CM 一致，不加 speaker 标签
                speaker_tag = ""
        parts = [f"<cm_time>{time_str}</cm_time>"]
        if speaker_tag:
            parts.append(speaker_tag)
        parts.append(f"<cm_nickname>{display_name}</cm_nickname>")
        return " ".join(parts)

    @classmethod
    def _message_content_to_text(cls, content: Any) -> str:
        return Message.content_to_text(content)

    def _parse_llm_response_batch(
        self, response_text: str, is_group_chat: bool
    ) -> list[dict[str, Any]]:
        """严格解析萃取 JSON；非 JSON 响应不得降级为长期记忆。"""
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        if not cleaned_text:
            raise ValueError("LLM 萃取响应为空")
        try:
            payload = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM 萃取响应不是合法 JSON") from exc
        return self._normalize_batch_payload(payload, is_group_chat)

    def _normalize_batch_payload(
        self, payload: Any, is_group_chat: bool
    ) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ValueError("LLM 萃取响应根节点必须是 JSON 对象")

        if "memories" in payload:
            status = payload.get("status", "success")
            if status not in {"success", "skip"}:
                raise ValueError("status 必须是 success 或 skip")
            raw_memories = payload.get("memories")
            if not isinstance(raw_memories, list):
                raise ValueError("memories 必须是数组")
            if status == "skip":
                if raw_memories:
                    raise ValueError("status=skip 时 memories 必须为空")
                reason = str(payload.get("reason", "") or "").strip()
                raise LLMExtractionSkip(reason)
        elif {"summary", "topics", "key_facts"}.issubset(payload):
            # 兼容 cm.2 及更早 Prompt 的单记忆 JSON。
            raw_memories = [payload]
        else:
            raise ValueError("LLM 萃取响应缺少 memories 数组")

        if len(raw_memories) > 5:
            raise ValueError("memories 最多允许 5 条")

        normalized: list[dict[str, Any]] = []
        for item in raw_memories:
            if not isinstance(item, dict):
                raise ValueError("memories 中的每一项都必须是 JSON 对象")
            self._validate_memory_payload(item, is_group_chat)
            normalized.append(self._normalize_parsed_data(dict(item), is_group_chat))
        return normalized

    @staticmethod
    def _validate_memory_payload(item: dict[str, Any], is_group_chat: bool) -> None:
        """校验决定是否允许持久化的核心字段，拒绝拒答伪装成记忆。"""
        summary = item.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("memory.summary 必须是非空字符串")

        for field, maximum in (("topics", 4), ("key_facts", 5)):
            value = item.get(field)
            if (
                not isinstance(value, list)
                or not value
                or len(value) > maximum
                or any(not isinstance(entry, str) or not entry.strip() for entry in value)
            ):
                raise ValueError(
                    f"memory.{field} 必须是包含 1 至 {maximum} 个非空字符串的数组"
                )

        event_time = item.get("event_time", "")
        if not isinstance(event_time, str):
            raise ValueError("memory.event_time 必须是字符串")

        sentiment = item.get("sentiment")
        if sentiment not in {"positive", "neutral", "negative"}:
            raise ValueError(
                "memory.sentiment 必须是 positive、neutral 或 negative"
            )

        importance = item.get("importance")
        if (
            isinstance(importance, bool)
            or not isinstance(importance, (int, float))
            or not 0.0 <= float(importance) <= 1.0
        ):
            raise ValueError("memory.importance 必须是 0.0 至 1.0 的数字")

        if is_group_chat and "participants" in item:
            participants = item["participants"]
            if not isinstance(participants, list) or any(
                not isinstance(entry, str) or not entry.strip()
                for entry in participants
            ):
                raise ValueError("memory.participants 必须是字符串数组")

    def _build_storage_format(
        self,
        fallback_excerpt: str,
        structured_data: dict[str, Any],
        is_group_chat: bool,
    ) -> tuple[str, dict[str, Any]]:
        """
        构建存储格式

        Args:
            fallback_excerpt: 当摘要为空时使用的对话摘录
            structured_data: 结构化数据
            is_group_chat: 是否为群聊

        Returns:
            (content, metadata) 元组
        """
        summary = str(structured_data.get("summary", "") or "").strip()
        key_facts = structured_data.get("key_facts", [])
        topics = structured_data.get("topics", [])
        participants = structured_data.get("participants", [])
        event_time = str(structured_data.get("event_time", "") or "").strip()

        # v3 canonical 完全由中性结构字段组成，不再混入 Persona 文风。
        canonical_parts: list[str] = []
        if event_time:
            canonical_parts.append(f"事件时间：{event_time}")
        if participants:
            canonical_parts.append(
                "参与者：" + "、".join(str(item) for item in participants[:8])
            )
        if topics:
            canonical_parts.append("主题：" + "、".join(str(item) for item in topics[:5]))
        if key_facts:
            canonical_parts.append(
                "事实：" + "；".join(str(fact) for fact in key_facts[:5])
            )
        if summary:
            canonical_parts.append(f"摘要：{summary}")
        canonical_summary = " | ".join(canonical_parts)

        # content 字段使用 canonical_summary，提升检索稳定性
        if canonical_summary:
            content = canonical_summary
        else:
            content = fallback_excerpt

        # metadata字段:存储结构化信息
        # 注意：不要在这里设置 create_time 和 last_access_time
        # 这些字段会由 MemoryEngine.add_memory() 自动添加
        metadata = {
            "topics": topics,
            "key_facts": key_facts,
            "event_time": event_time,
            "sentiment": structured_data.get("sentiment", "neutral"),
            "interaction_type": "group_chat" if is_group_chat else "private_chat",
            "neutral_summary": summary,
            "canonical_summary": canonical_summary,
            # persona_summary 仅作为旧记录兼容字段；v3 新记录不再生成。
            "summary_schema_version": "v3",
            # summary_quality 由 process_conversation 中的 SummaryValidator 覆盖写入
        }

        if is_group_chat and "participants" in structured_data:
            metadata["participants"] = participants

        return content, metadata

    def _normalize_parsed_data(self, data: dict, is_group_chat: bool) -> dict[str, Any]:
        """
        规范化解析后的数据（补充缺失字段、类型转换）

        Args:
            data: 解析后的原始字典
            is_group_chat: 是否为群聊

        Returns:
            规范化后的字典
        """
        required_fields = [
            "summary",
            "topics",
            "key_facts",
            "event_time",
            "sentiment",
            "importance",
        ]
        if is_group_chat:
            required_fields.append("participants")

        for field in required_fields:
            if field not in data:
                data[field] = self._get_default_value(field)

        data["summary"] = str(data.get("summary", ""))
        data["topics"] = self._ensure_list(data.get("topics", []))[:5]
        data["key_facts"] = self._ensure_list(data.get("key_facts", []))[:5]
        data["event_time"] = str(data.get("event_time", "") or "").strip()
        data["sentiment"] = self._validate_sentiment(data.get("sentiment", "neutral"))
        data["importance"] = self._validate_importance(data.get("importance", 0.5))

        if is_group_chat:
            data["participants"] = self._ensure_list(data.get("participants", []))

        return data

    def _ensure_list(self, value: Any) -> list[str]:
        """确保值是字符串列表"""
        if isinstance(value, list):
            return [str(item) for item in value if item]
        elif isinstance(value, str):
            return [value] if value else []
        else:
            return []

    def _validate_sentiment(self, sentiment: str) -> str:
        """验证情感值"""
        valid_sentiments = ["positive", "neutral", "negative"]
        sentiment = sentiment.lower()
        return sentiment if sentiment in valid_sentiments else "neutral"

    def _validate_importance(self, importance: Any) -> float:
        """验证重要性评分"""
        try:
            score = float(importance)
            return max(0.0, min(1.0, score))  # 限制在0-1之间
        except (ValueError, TypeError):
            return 0.5

    def build_memory_from_structured_data(
        self,
        structured_data: dict[str, Any],
        is_group_chat: bool = False,
        fallback_excerpt: str = "",
    ) -> tuple[str, dict[str, Any], float]:
        """复用自动总结流程，将结构化数据转换为标准记忆存储格式。"""
        # 与自动总结路径保持一致：先校验质量，再规范化。
        # 这样原始 importance 越界等异常仍会被判为 low quality。
        quality = self._validate_summary_quality(structured_data)
        normalized = self._normalize_parsed_data(structured_data, is_group_chat)
        normalized["_quality"] = quality

        content, metadata = self._build_storage_format(
            fallback_excerpt or normalized.get("summary", ""),
            normalized,
            is_group_chat,
        )
        metadata["summary_quality"] = quality
        return (
            content,
            metadata,
            self._validate_importance(normalized.get("importance")),
        )

    def _get_default_value(self, field: str) -> Any:
        """获取字段的默认值"""
        defaults = {
            "summary": "",
            "topics": [],
            "key_facts": [],
            "event_time": "",
            "participants": [],
            "sentiment": "neutral",
            "importance": 0.5,
        }
        return defaults.get(field, "")

    def _validate_summary_quality(self, structured_data: dict[str, Any]) -> str:
        """
        校验总结质量，返回质量等级。

        检查规则：
        1. summary 不能为空或过短（< 10 字符）
        2. key_facts 至少有 1 条
        3. importance 在合法范围内
        4. summary 不含泛化词（"某用户"、"有人"等）

        Returns:
            "normal" 或 "low"
        """
        summary = structured_data.get("summary", "")
        key_facts = structured_data.get("key_facts", [])
        importance = structured_data.get("importance", 0.5)

        if not summary or len(summary.strip()) < 10:
            return "low"
        if not key_facts:
            return "low"
        if not isinstance(importance, (int, float)) or not (0.0 <= importance <= 1.0):
            return "low"

        # 泛化词检测
        generic_terms = [
            "某用户",
            "有人",
            "某人",
            "用户说",
            "对方说",
            "群成员",
            "某群成员",
        ]
        if any(term in summary for term in generic_terms):
            return "low"

        return "normal"

    def classify_atoms_from_metadata(
        self,
        metadata: dict[str, Any],
        parent_importance: float = 0.5,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[MemoryAtom]:
        """Generate time-aware memory atoms from key_facts in metadata.

        This is a post-processing step after process_conversation().
        It does NOT make additional LLM calls — classification is rule-based.
        """
        if not self.config.get("atom_enabled", True):
            return []
        key_facts: list[str] = metadata.get("key_facts", [])
        if not key_facts:
            return []
        topics = metadata.get("topics", [])
        participants = metadata.get("participants", [])
        return classify_atoms(
            key_facts=key_facts,
            topics=topics,
            participants=participants,
            parent_importance=parent_importance,
            session_id=session_id,
            persona_id=persona_id,
        )
