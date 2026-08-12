"""
记忆处理器 - 使用LLM将对话历史处理为结构化记忆
"""

import asyncio
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ...log import logger, tag

from ..models.conversation_models import Message
from ..models.memory_atom import MemoryAtom
from .atom_classifier import classify_atoms


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
{"memories": [{"summary": "中性摘要", "topics": ["主题"], "key_facts": ["事实"], "event_time": "", "sentiment": "neutral", "importance": 0.5}]}
"""
            self.group_chat_prompt = """分析以下群聊对话并生成JSON格式的长期记忆:
{conversation}

输出格式:
{"memories": [{"summary": "中性摘要", "topics": ["主题"], "key_facts": ["事实"], "participants": ["参与者"], "event_time": "", "sentiment": "neutral", "importance": 0.5}]}
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
            "你是长期记忆事实萃取器。请严格按照 JSON 格式输出。\n"
            f"当前日期时间: {current_date}\n"
            "只提取对未来有用、能够由对话支持的事实。使用中性、简洁、第三人称或"
            "直接事实表述；不得模仿任何角色人格、语气或文风，不得文学化补写。\n"
            "必须区分消息前缀中的具体发言者和 Bot，不得把不同人的行为合并。\n"
            "将今天、明天、昨天、下周等相对时间转换为可长期理解的绝对日期。\n"
            "一次输入可按主题或连续事件拆成 0 至 5 条记忆；没有持久价值时输出"
            " {\"memories\": []}。"
        )

    async def _call_llm_with_retry(
        self, prompt: str, system_prompt: str, max_retries: int = 3
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
                return response.completion_text
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                wait_time = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"{tag('processor')} LLM 调用失败，{wait_time:.1f}s 后重试 "
                    f"({attempt + 1}/{max_retries}): {e}"
                )
                await asyncio.sleep(wait_time)
        if last_error:
            raise last_error
        raise RuntimeError("LLM 调用失败，未捕获到具体异常")

    def _try_fix_json(self, text: str) -> str:
        """
        尝试修复损坏的 JSON 字符串

        Args:
            text: 可能损坏的 JSON 字符串

        Returns:
            修复后的 JSON 字符串
        """
        fixed = text.strip()

        # 移除 markdown 代码块标记
        if fixed.startswith("```json"):
            fixed = fixed[7:]
        elif fixed.startswith("```"):
            fixed = fixed[3:]
        if fixed.endswith("```"):
            fixed = fixed[:-3]
        fixed = fixed.strip()

        # 修复未闭合的字符串（截断的 JSON）
        open_quotes = fixed.count('"') - fixed.count('\\"')
        if open_quotes % 2 != 0:
            fixed += '"'

        # 修复未闭合的数组
        open_brackets = fixed.count("[") - fixed.count("]")
        if open_brackets > 0:
            fixed += "]" * open_brackets

        # 修复未闭合的对象
        open_braces = fixed.count("{") - fixed.count("}")
        if open_braces > 0:
            fixed += "}" * open_braces

        # 移除尾部逗号（JSON 不允许）
        fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

        # 修复常见的转义问题
        fixed = fixed.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

        return fixed

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
        time_str = datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        display_name = msg.sender_name if msg.sender_name else msg.sender_id or "未知"
        is_bot = msg.metadata.get("is_bot_message", False) or msg.role == "assistant"
        if is_bot:
            return f"[Bot: {display_name} | ID: {msg.sender_id} | {time_str}]"
        relation = msg.metadata.get("speaker_relation")
        if msg.group_id and relation == "current_user":
            return f"[当前发言者: {display_name} | ID: {msg.sender_id} | {time_str}]"
        if msg.group_id and relation == "other_user":
            return f"[其他发言者: {display_name} | ID: {msg.sender_id} | {time_str}]"
        return f"[{display_name} | ID: {msg.sender_id} | {time_str}]"

    @classmethod
    def _message_content_to_text(cls, content: Any) -> str:
        return Message.content_to_text(content)

    def _parse_llm_response_batch(
        self, response_text: str, is_group_chat: bool
    ) -> list[dict[str, Any]]:
        """解析新版 ``{"memories": [...]}``，并兼容旧版单记忆 JSON。"""
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            payload = json.loads(cleaned_text)
            return self._normalize_batch_payload(payload, is_group_chat)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"{tag('processor')} 批量 JSON 解析失败: {exc}")

        try:
            payload = json.loads(self._try_fix_json(response_text))
            return self._normalize_batch_payload(payload, is_group_chat)
        except (json.JSONDecodeError, ValueError, TypeError):
            legacy = self._extract_by_regex(response_text, is_group_chat)
            return [self._normalize_parsed_data(legacy, is_group_chat)]

    def _normalize_batch_payload(
        self, payload: Any, is_group_chat: bool
    ) -> list[dict[str, Any]]:
        if isinstance(payload, dict) and "memories" in payload:
            raw_memories = payload.get("memories")
            if not isinstance(raw_memories, list):
                raise ValueError("memories 必须是数组")
        elif isinstance(payload, dict):
            # 兼容 cm.2 及更早 Prompt 的单记忆 JSON。
            raw_memories = [payload]
        elif isinstance(payload, list):
            raw_memories = payload
        else:
            raise ValueError("LLM 响应必须是 JSON 对象或数组")

        normalized: list[dict[str, Any]] = []
        for item in raw_memories[:5]:
            if not isinstance(item, dict):
                logger.warning(f"{tag('processor')} 跳过非对象记忆项")
                continue
            normalized.append(self._normalize_parsed_data(dict(item), is_group_chat))
        return normalized

    def _parse_llm_response(
        self, response_text: str, is_group_chat: bool
    ) -> dict[str, Any]:
        """
        解析LLM响应,提取JSON数据

        Args:
            response_text: LLM响应文本
            is_group_chat: 是否为群聊

        Returns:
            解析后的字典数据
        """
        logger.debug(f"{tag('processor')} 开始解析 LLM 响应，长度={len(response_text)}")

        try:
            # 尝试直接解析JSON
            # 先清理可能的markdown代码块标记
            cleaned_text = response_text.strip()
            logger.debug(
                f"{tag('processor')} 清理前响应长度={len(response_text)}"
            )

            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
                logger.debug(f"{tag('processor')} 移除了 ```json 标记")
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
                logger.debug(f"{tag('processor')} 移除了 ``` 标记")
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
                logger.debug(f"{tag('processor')} 移除了结尾 ``` 标记")
            cleaned_text = cleaned_text.strip()

            logger.debug(
                f"{tag('processor')} 清理后 JSON 长度={len(cleaned_text)}"
            )

            # 解析JSON
            data = json.loads(cleaned_text)

            # 类型检查：确保解析结果是 dict
            if not isinstance(data, dict):
                logger.warning(
                    f"{tag('processor')} JSON 解析结果不是 dict，类型为 {type(data).__name__}"
                )
                raise ValueError(f"期望 dict 类型，实际为 {type(data).__name__}")

            logger.debug(f"{tag('processor')} JSON 解析成功")
            logger.debug(f"{tag('processor')} 解析得到的字段: {list(data.keys())}")

            # 验证必需字段 - 简化后的字段列表
            required_fields = [
                "summary",
                "topics",
                "key_facts",
                "sentiment",
                "importance",
            ]
            if is_group_chat:
                required_fields.append("participants")

            for field in required_fields:
                if field not in data:
                    logger.warning(
                        f"{tag('processor')} LLM 响应缺少字段: {field}, 使用默认值"
                    )
                    data[field] = self._get_default_value(field)

            # 数据类型校验和规范化
            data["summary"] = str(data.get("summary", ""))
            logger.debug(f"{tag('processor')} 提取 summary: {data['summary']}")

            data["topics"] = self._ensure_list(data.get("topics", []))[:5]
            logger.debug(
                f"{tag('processor')} 提取 topics ({len(data['topics'])} 个): {data['topics']}"
            )

            data["key_facts"] = self._ensure_list(data.get("key_facts", []))[:5]
            logger.debug(
                f"{tag('processor')} 提取 key_facts ({len(data['key_facts'])} 个): {data['key_facts']}"
            )

            data["sentiment"] = self._validate_sentiment(
                data.get("sentiment", "neutral")
            )
            logger.debug(f"{tag('processor')} 提取 sentiment: {data['sentiment']}")

            data["importance"] = self._validate_importance(data.get("importance", 0.5))
            logger.debug(f"{tag('processor')} 提取 importance: {data['importance']}")

            if is_group_chat:
                data["participants"] = self._ensure_list(data.get("participants", []))
                logger.debug(
                    f"{tag('processor')} 提取 participants ({len(data['participants'])} 个): {data['participants']}"
                )

            return data

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"{tag('processor')} JSON 解析失败: {e}")
            logger.debug(
                f"{tag('processor')} 解析失败的内容:\n{response_text}"
            )

            # 尝试修复 JSON 后再解析
            logger.debug(f"{tag('processor')} 尝试修复 JSON 后重新解析")
            try:
                fixed_text = self._try_fix_json(response_text)
                data = json.loads(fixed_text)
                if isinstance(data, dict):
                    logger.debug(f"{tag('processor')} JSON 修复后解析成功")
                    return self._normalize_parsed_data(data, is_group_chat)
            except (json.JSONDecodeError, ValueError) as fix_err:
                logger.debug(f"{tag('processor')} JSON 修复后仍无法解析: {fix_err}")

            logger.debug(f"{tag('processor')} 尝试使用正则表达式提取 JSON")
            # 尝试正则提取
            return self._extract_by_regex(response_text, is_group_chat)
        except Exception as e:
            logger.error(
                f"{tag('processor')} 解析 LLM 响应时发生异常: {e}", exc_info=True
            )
            logger.debug(
                f"{tag('processor')} 异常发生时的响应内容:\n{response_text}"
            )
            return self._get_default_structured_data(is_group_chat)

    def _extract_by_regex(self, text: str, is_group_chat: bool) -> dict[str, Any]:
        """
        使用正则表达式从文本中提取结构化数据(备用方案)

        Args:
            text: 响应文本
            is_group_chat: 是否为群聊

        Returns:
            提取的结构化数据
        """
        logger.debug(f"{tag('processor')} 开始使用正则表达式提取结构化数据")
        data = self._get_default_structured_data(is_group_chat)

        try:
            # 先尝试找到完整的 JSON 块
            json_matches = re.findall(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL
            )
            logger.debug(
                f"{tag('processor')} 正则匹配到 {len(json_matches)} 个可能的 JSON 块"
            )

            for i, match in enumerate(json_matches):
                logger.debug(
                    f"{tag('processor')} JSON 块 #{i + 1}:\n{match}"
                )
                try:
                    # 尝试解析每个匹配的块
                    parsed = json.loads(match)
                    if "summary" in parsed:
                        logger.debug(
                            f"{tag('processor')} 成功从第 {i + 1} 个 JSON 块中解析数据"
                        )
                        data = parsed
                        break
                except json.JSONDecodeError:
                    continue

            # 如果没有找到完整的 JSON，尝试单独提取字段
            if data == self._get_default_structured_data(is_group_chat):
                logger.debug(f"{tag('processor')} 未找到完整 JSON，尝试提取单独字段")

                # 提取summary
                summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
                if summary_match:
                    data["summary"] = summary_match.group(1)
                    logger.debug(
                        f"{tag('processor')} 正则提取 summary: {data['summary']}"
                    )

                # 提取importance
                importance_match = re.search(r'"importance"\s*:\s*([0-9.]+)', text)
                if importance_match:
                    data["importance"] = float(importance_match.group(1))
                    logger.debug(
                        f"{tag('processor')} 正则提取 importance: {data['importance']}"
                    )

                # 提取sentiment
                sentiment_match = re.search(r'"sentiment"\s*:\s*"(\w+)"', text)
                if sentiment_match:
                    data["sentiment"] = sentiment_match.group(1)
                    logger.debug(
                        f"{tag('processor')} 正则提取 sentiment: {data['sentiment']}"
                    )

                # 提取 topics 数组
                topics_match = re.search(r'"topics"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                if topics_match:
                    topics_str = topics_match.group(1)
                    topics = re.findall(r'"([^"]+)"', topics_str)
                    data["topics"] = topics[:5]
                    logger.debug(f"{tag('processor')} 正则提取 topics: {data['topics']}")

                # 提取 key_facts 数组
                facts_match = re.search(r'"key_facts"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                if facts_match:
                    facts_str = facts_match.group(1)
                    facts = re.findall(r'"([^"]+)"', facts_str)
                    data["key_facts"] = facts[:5]
                    logger.debug(
                        f"{tag('processor')} 正则提取 key_facts: {data['key_facts']}"
                    )

            logger.debug(
                f"{tag('processor')} 正则提取完成，提取到的字段: {list(data.keys())}"
            )

        except Exception as e:
            logger.error(f"{tag('processor')} 正则提取失败: {e}", exc_info=True)

        return data

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

    def _get_default_structured_data(self, is_group_chat: bool) -> dict[str, Any]:
        """获取默认的结构化数据"""
        data = {
            "summary": "对话记录",
            "topics": [],
            "key_facts": [],
            "event_time": "",
            "sentiment": "neutral",
            "importance": 0.5,
        }
        if is_group_chat:
            data["participants"] = []
        return data

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
