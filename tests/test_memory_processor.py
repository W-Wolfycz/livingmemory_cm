"""
Tests for MemoryProcessor.
"""

import asyncio
import json
import re
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import livingmemory_cm.core.processors.text_processor as text_processor_module
import pytest
from livingmemory_cm.core.models.conversation_models import Message
from livingmemory_cm.core.processors.memory_processor import (
    LLMExtractionSkip,
    MemoryProcessor,
)
from livingmemory_cm.core.processors.text_processor import TextProcessor
from livingmemory_cm.core.utils.stopwords_manager import StopwordsManager


class _DummyLLMProvider:
    def __init__(self, completion_text: str):
        self._completion_text = completion_text
        self.text_chat = AsyncMock(side_effect=self._chat)

    async def _chat(self, prompt: str, system_prompt: str):
        return SimpleNamespace(completion_text=self._completion_text)


class _FailingLLMProvider:
    def __init__(self, error: BaseException):
        self._error = error
        self.text_chat = AsyncMock(side_effect=self._chat)

    async def _chat(self, prompt: str, system_prompt: str):
        raise self._error


def _make_messages():
    return [
        Message(
            id=1,
            session_id="s1",
            role="user",
            content="明天下午三点开会",
            sender_id="u1",
            sender_name="张三",
            group_id=None,
            platform="test",
            metadata={},
        ),
        Message(
            id=2,
            session_id="s1",
            role="assistant",
            content="收到，我会提醒你",
            sender_id="bot",
            sender_name="Bot",
            group_id=None,
            platform="test",
            metadata={"is_bot_message": True},
        ),
    ]


def test_participant_identities_keep_aliases_for_one_sender() -> None:
    messages = [
        Message(
            id=1,
            session_id="s1",
            role="user",
            content="第一条",
            sender_id="10001",
            sender_name="旧昵称",
            platform="qq",
        ),
        Message(
            id=2,
            session_id="s1",
            role="user",
            content="第二条",
            sender_id="10001",
            sender_name="新昵称",
            platform="qq",
        ),
    ]

    identities = MemoryProcessor._extract_participant_identities(messages)

    assert len(identities) == 1
    assert identities[0]["identity_key"] == "qq:10001"
    assert identities[0]["display_name"] == "新昵称"
    assert identities[0]["aliases"] == ["旧昵称", "新昵称"]


@pytest.mark.asyncio
async def test_persona_prompt_is_not_included_in_extraction():
    llm = _DummyLLMProvider(
        """{
            "summary":"我愉快地记录了这次交流",
            "topics":["闲聊"],
            "key_facts":["用户问候"],
            "sentiment":"positive",
            "importance":0.5
        }"""
    )
    context = Mock()
    context.persona_manager = Mock()
    context.persona_manager.get_persona = AsyncMock(
        return_value=SimpleNamespace(system_prompt="你是活泼助手")
    )

    processor = MemoryProcessor(llm_provider=llm, context=context)

    system_prompt = await processor._build_system_prompt_with_persona("persona_1")
    assert "长期记忆事实萃取器" in system_prompt
    assert "活泼助手" not in system_prompt
    context.persona_manager.get_persona.assert_not_awaited()


@pytest.mark.asyncio
async def test_extraction_prompt_static_prefix_is_cache_friendly():
    """守的静默失效：静态前缀里一旦含逐次变化的日期，提供商的自动前缀缓存会整体
    失效——没有任何报错，只是每次请求都全量计费。"""
    processor = MemoryProcessor(llm_provider=_DummyLLMProvider("{}"), context=Mock())

    system_prompt = await processor._build_system_prompt_with_persona(None)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", system_prompt), (
        "system prompt 必须逐字节稳定，不能带日期/时间"
    )

    prompts_dir = Path(__file__).resolve().parents[1] / "core" / "prompts"
    for name in ("group_chat_prompt.txt", "private_chat_prompt.txt"):
        text = (prompts_dir / name).read_text(encoding="utf-8")
        static_head, _, dynamic_tail = text.partition("{conversation}")
        assert "{current_date}" not in static_head, f"{name} 的静态段不能带日期"
        assert "{current_date}" in dynamic_tail, f"{name} 末尾仍需给出当前日期"


# ── 中性摘要与质量校验 ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_neutral_summary_stores_canonical_without_persona_summary():
    llm = _DummyLLMProvider(
        """{
            "summary":"我记录了张三明天下午三点开会，并给出提醒",
            "topics":["会议提醒"],
            "key_facts":["张三明天下午三点开会"],
            "sentiment":"neutral",
            "importance":0.8
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, importance = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    # canonical_summary 应存在且包含事实内容
    assert "canonical_summary" in metadata
    assert len(metadata["canonical_summary"]) > 0

    assert metadata["neutral_summary"]
    assert "persona_summary" not in metadata

    # content 应使用 canonical_summary（事实导向）
    assert content == metadata["canonical_summary"]

    # schema 版本标记
    assert metadata.get("summary_schema_version") == "v3"


@pytest.mark.asyncio
async def test_process_conversation_batch_returns_multiple_topic_memories():
    llm = _DummyLLMProvider(
        """{
          "memories": [
            {
              "summary": "张三安排项目会议",
              "topics": ["会议"],
              "key_facts": ["张三安排 2026-07-20 15:00 开会", "会议需准备项目文档"],
              "event_time": "2026-07-20 15:00",
              "sentiment": "neutral",
              "importance": 0.8
            },
            {
              "summary": "张三偏好黑咖啡",
              "topics": ["饮食偏好"],
              "key_facts": ["张三喝黑咖啡时不加糖"],
              "event_time": "",
              "sentiment": "neutral",
              "importance": 0.6
            }
          ]
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    memories = await processor.process_conversation_batch(_make_messages())

    assert len(memories) == 2
    assert "项目文档" in memories[0][0]
    assert "黑咖啡" in memories[1][0]
    assert all(item[1]["summary_schema_version"] == "v3" for item in memories)


@pytest.mark.asyncio
async def test_canonical_summary_includes_key_facts():
    """canonical_summary 应将 key_facts 拼接到摘要中，提升检索覆盖率。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"用户提到了一个重要事项",
            "topics":["备忘"],
            "key_facts":["明天下午三点开会", "需要准备PPT"],
            "sentiment":"neutral",
            "importance":0.7
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    # canonical_summary 应包含 key_facts 内容
    assert "明天下午三点开会" in metadata["canonical_summary"]
    assert "需要准备PPT" in metadata["canonical_summary"]


def test_validate_summary_quality_directly():
    """直接测试 _validate_summary_quality 的各种边界情况。"""
    from unittest.mock import MagicMock

    processor = MemoryProcessor(llm_provider=MagicMock(), context=None)

    # 正常情况
    assert (
        processor._validate_summary_quality(
            {
                "summary": "用户明确表示喜欢吃寿司",
                "key_facts": ["用户喜欢寿司"],
                "importance": 0.7,
            }
        )
        == "normal"
    )

    # summary 过短
    assert (
        processor._validate_summary_quality(
            {
                "summary": "短",
                "key_facts": ["fact"],
                "importance": 0.5,
            }
        )
        == "low"
    )

    # importance 超出范围
    assert (
        processor._validate_summary_quality(
            {
                "summary": "用户明确表示喜欢吃寿司",
                "key_facts": ["用户喜欢寿司"],
                "importance": 1.5,
            }
        )
        == "low"
    )

    # 泛化词检测
    assert (
        processor._validate_summary_quality(
            {
                "summary": "有人提到了一些事情",
                "key_facts": ["有人说话"],
                "importance": 0.5,
            }
        )
        == "low"
    )


def test_quality_failure_reason_reports_specific_rule():
    """低质量警告应携带具体触发规则（不输出记忆正文）。"""
    processor = MemoryProcessor(llm_provider=Mock(), context=None)

    assert processor._quality_failure_reason(
        {"summary": "", "key_facts": ["事实"], "importance": 0.5}
    ) == "summary 为空"
    assert processor._quality_failure_reason(
        {"summary": "短", "key_facts": ["事实"], "importance": 0.5}
    ) == "summary 过短（1 字符）"
    assert processor._quality_failure_reason(
        {"summary": "用户进行了一次普通对话", "key_facts": [], "importance": 0.5}
    ) == "key_facts 为空"
    assert processor._quality_failure_reason(
        {"summary": "用户明确表示喜欢吃寿司", "key_facts": ["事实"], "importance": 1.5}
    ) == "importance 越界（1.5）"
    assert processor._quality_failure_reason(
        {"summary": "某用户在群里说了一些事情", "key_facts": ["事实"], "importance": 0.5}
    ) == "summary 含泛化词「某用户」"
    assert (
        processor._quality_failure_reason(
            {"summary": "用户明确表示喜欢吃寿司", "key_facts": ["事实"], "importance": 0.7}
        )
        is None
    )


# ── 群聊路径测试 ──────────────────────────────────────────────────────────────


def _make_group_messages():
    """构造一组群聊消息（含 group_id）"""
    return [
        Message(
            id=1,
            session_id="aiocqhttp:GroupMessage:88888",
            role="user",
            content="大家觉得 AI 工具怎么样？",
            sender_id="10001",
            sender_name="张三",
            group_id="88888",
            platform="aiocqhttp",
            metadata={},
        ),
        Message(
            id=2,
            session_id="aiocqhttp:GroupMessage:88888",
            role="user",
            content="我觉得 ChatGPT 写代码效率提升了 30%",
            sender_id="10002",
            sender_name="李四",
            group_id="88888",
            platform="aiocqhttp",
            metadata={},
        ),
        Message(
            id=3,
            session_id="aiocqhttp:GroupMessage:88888",
            role="assistant",
            content="AI 工具确实能提升效率，但需要仔细审查生成的代码",
            sender_id="bot",
            sender_name="Bot",
            group_id="88888",
            platform="aiocqhttp",
            metadata={"is_bot_message": True},
        ),
    ]


@pytest.mark.asyncio
async def test_process_group_chat_sets_interaction_type():
    """群聊路径应将 interaction_type 设置为 group_chat。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"群聊讨论了 AI 工具的使用效果",
            "topics":["AI工具","工作效率"],
            "key_facts":["张三认为 ChatGPT 效率提升 30%","需要仔细审查 AI 生成代码"],
            "participants":["张三","李四"],
            "sentiment":"positive",
            "importance":0.75
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, importance = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    assert metadata["interaction_type"] == "group_chat"
    assert importance == 0.75


@pytest.mark.asyncio
async def test_process_group_chat_extracts_participants():
    """群聊路径应正确提取 participants 字段。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"群聊讨论了 AI 工具的使用效果",
            "topics":["AI工具"],
            "key_facts":["张三认为 ChatGPT 效率提升 30%"],
            "participants":["张三","李四","王五"],
            "sentiment":"positive",
            "importance":0.7
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    assert "participants" in metadata
    assert "张三" in metadata["participants"]
    assert "李四" in metadata["participants"]
    assert "王五" in metadata["participants"]


def test_format_conversation_sanitizes_multimodal_private_message():
    processor = MemoryProcessor(llm_provider=None, context=None)
    message = Message(
        id=1,
        session_id="s1",
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            {"type": "text", "text": "这张图里有会议安排"},
        ],
        sender_id="u1",
        sender_name="张三",
        group_id=None,
        platform="test",
        metadata={},
    )

    formatted = processor._format_conversation([message])

    assert "这张图里有会议安排" in formatted
    assert "image_url" not in formatted
    assert "example.test" not in formatted


def test_text_tokenize_falls_back_when_jieba_runtime_fails(monkeypatch) -> None:
    class BrokenJieba:
        @staticmethod
        def cut_for_search(_text):
            raise AttributeError(
                "module 'pkg_resources' has no attribute 'resource_stream'"
            )

    monkeypatch.setattr(text_processor_module, "JIEBA_AVAILABLE", True)
    monkeypatch.setattr(text_processor_module, "JIEBA_RUNTIME_DISABLED", False)
    monkeypatch.setattr(text_processor_module, "jieba", BrokenJieba)

    with pytest.warns(UserWarning, match="jieba 分词初始化失败"):
        tokens = TextProcessor().tokenize("编程快乐")

    assert tokens
    assert "编" in tokens
    assert text_processor_module.JIEBA_RUNTIME_DISABLED is True


# ── 严格 JSON 协议 ───────────────────────────────────────────────────────────


def _new_processor() -> MemoryProcessor:
    return MemoryProcessor(llm_provider=Mock(), context=None)


def _memory_payload(
    *,
    summary: str = "张三明天下午三点开会",
    topics: list | None = None,
    key_facts: list | None = None,
    event_time: str = "",
    sentiment: str = "neutral",
    importance: float = 0.8,
    extra: dict | None = None,
) -> dict:
    payload = {
        "summary": summary,
        "topics": topics if topics is not None else ["会议"],
        "key_facts": key_facts if key_facts is not None else ["张三明天下午三点开会"],
        "event_time": event_time,
        "sentiment": sentiment,
        "importance": importance,
    }
    if extra:
        payload.update(extra)
    return payload


def test_strict_json_protocol_skip_empty_array_raises_llmextractionskip():
    processor = _new_processor()

    with pytest.raises(LLMExtractionSkip) as exc_info:
        processor._parse_llm_response_batch(
            '{"status":"skip","reason":"content_policy","memories":[]}',
            is_group_chat=False,
        )

    assert exc_info.value.reason == "content_policy"


def test_strict_json_protocol_rejects_more_than_five_memories():
    processor = _new_processor()
    payload = json.dumps(
        {
            "status": "success",
            "memories": [
                _memory_payload(summary=f"记忆 {index}") for index in range(6)
            ],
        },
        ensure_ascii=False,
    )

    with pytest.raises(ValueError, match="最多允许 5 条"):
        processor._parse_llm_response_batch(payload, is_group_chat=False)


@pytest.mark.parametrize(
    "field, bad_value, match",
    [
        pytest.param("topics", [], "topics", id="topics-empty"),
        pytest.param("topics", ["a"] * 5, "topics", id="topics-overflow"),
        pytest.param("topics", "会议", "topics", id="topics-not-list"),
        pytest.param("key_facts", [], "key_facts", id="key-facts-empty"),
        pytest.param("key_facts", ["f"] * 6, "key_facts", id="key-facts-overflow"),
        pytest.param("key_facts", "事实", "key_facts", id="key-facts-not-list"),
        pytest.param("event_time", 20260720, "event_time", id="event-time-non-string"),
        pytest.param("sentiment", "angry", "sentiment", id="sentiment-invalid"),
        pytest.param("importance", True, "importance", id="importance-bool"),
        pytest.param("importance", 1.5, "importance", id="importance-out-of-range"),
        pytest.param(
            "participants", ["张三", 42], "participants", id="participants-non-string"
        ),
    ],
)
def test_strict_json_protocol_rejects_invalid_memory_fields(field, bad_value, match):
    """记忆条目核心字段的非法值一律拒绝（不静默规范化）。"""
    processor = _new_processor()

    kwargs = {field: bad_value}
    extra = None
    if field == "participants":
        kwargs = {}
        extra = {"participants": bad_value}
    payload = json.dumps(
        {
            "status": "success",
            "memories": [_memory_payload(**kwargs, extra=extra)],
        },
        ensure_ascii=False,
    )
    with pytest.raises(ValueError, match=match):
        processor._parse_llm_response_batch(
            payload, is_group_chat=field == "participants"
        )


@pytest.mark.parametrize(
    "signature",
    [
        pytest.param("Input data may contain inappropriate content", id="openai"),
        pytest.param("content_policy_violation", id="signature"),
        pytest.param("ResponsibleAIPolicyViolation", id="azure"),
        pytest.param("request was rejected as a result of the content filter", id="filter"),
        pytest.param("输入数据可能包含不当内容", id="zh"),
    ],
)
def test_content_policy_rejection_detects_known_signatures(signature):
    error = RuntimeError(f"provider: {signature}")
    assert MemoryProcessor._is_content_policy_rejection(error)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            RuntimeError("request was rejected as a result of the content filter"),
            id="unrelated-status-code",
        ),
        pytest.param(RuntimeError("Invalid model parameter: max_tokens"), id="plain-400"),
    ],
)
def test_content_policy_rejection_guards_against_false_positives(error):
    """普通 400/无关状态码不得误判为内容安全拒绝。"""
    error.status_code = 400 if "Invalid" in str(error) else 500
    assert not MemoryProcessor._is_content_policy_rejection(error)


@pytest.mark.asyncio
async def test_call_llm_with_retry_propagates_structured_skip_without_retry(
    monkeypatch,
):
    """status=skip 的结构化拒绝不得被当作可重试失败。"""
    processor = _new_processor()
    processor._llm_provider = _DummyLLMProvider(
        '{"status":"skip","reason":"content_policy","memories":[]}'
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    with pytest.raises(LLMExtractionSkip):
        await processor._call_llm_with_retry(
            prompt="p",
            system_prompt="s",
            response_validator=lambda text: processor._parse_llm_response_batch(
                text, False
            ),
        )

    assert processor._llm_provider.text_chat.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_llm_with_retry_converts_provider_content_policy_to_skip(
    monkeypatch,
):
    """Provider 高置信内容安全拒绝转为 LLMExtractionSkip，不进入普通重试。"""
    processor = _new_processor()
    processor._llm_provider = _FailingLLMProvider(
        RuntimeError("Input data may contain inappropriate content")
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    with pytest.raises(LLMExtractionSkip) as exc_info:
        await processor._call_llm_with_retry(prompt="p", system_prompt="s")

    assert exc_info.value.reason == "provider_content_policy"
    assert processor._llm_provider.text_chat.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_max_retries_config_controls_attempt_count(monkeypatch):
    """config.llm_max_retries 控制最大尝试次数；缺省为 5，钳制不低于 1。"""
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    configured = MemoryProcessor(
        llm_provider=_FailingLLMProvider(RuntimeError("provider down")),
        context=None,
        config={"llm_max_retries": 5},
    )
    with pytest.raises(RuntimeError):
        await configured._call_llm_with_retry(prompt="p", system_prompt="s")
    assert configured._llm_provider.text_chat.await_count == 5

    defaulted = MemoryProcessor(
        llm_provider=_FailingLLMProvider(RuntimeError("provider down")),
        context=None,
    )
    with pytest.raises(RuntimeError):
        await defaulted._call_llm_with_retry(prompt="p", system_prompt="s")
    assert defaulted._llm_provider.text_chat.await_count == 5

    clamped = MemoryProcessor(
        llm_provider=_FailingLLMProvider(RuntimeError("provider down")),
        context=None,
        config={"llm_max_retries": 0},
    )
    with pytest.raises(RuntimeError):
        await clamped._call_llm_with_retry(prompt="p", system_prompt="s")
    assert clamped._llm_provider.text_chat.await_count == 1


class _MaxRetriesAwareProvider:
    """模拟 AstrBot 4.27.x 的 text_chat 签名（支持 request_max_retries）。"""

    def __init__(self, completion_text: str):
        self._completion_text = completion_text
        self.last_request_max_retries = None

    async def text_chat(self, prompt: str, system_prompt: str, request_max_retries=None):
        self.last_request_max_retries = request_max_retries
        return SimpleNamespace(completion_text=self._completion_text)


