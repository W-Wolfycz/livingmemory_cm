"""
Tests for MemoryProcessor.
"""

import asyncio
import json
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
async def test_process_conversation_success():
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

    assert "张三" in content
    assert metadata["interaction_type"] == "private_chat"
    assert "会议提醒" in metadata["topics"]
    assert importance == 0.8


@pytest.mark.asyncio
async def test_process_conversation_non_json_response_retries_then_fails(monkeypatch):
    """非 JSON 响应属于严格校验失败：重试 3 次后失败，不产生任何记忆。"""
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    llm = _DummyLLMProvider("summary=测试, importance=0.6")
    processor = MemoryProcessor(llm_provider=llm, context=None)

    with pytest.raises(ValueError):
        await processor.process_conversation(
            messages=_make_messages(),
            is_group_chat=False,
            persona_id=None,
        )

    # 严格协议：3 次尝试全部失败，未降级为记忆
    assert llm.text_chat.await_count == 3


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
async def test_process_conversation_batch_allows_empty_memories():
    processor = MemoryProcessor(
        llm_provider=_DummyLLMProvider('{"memories": []}'), context=None
    )

    assert await processor.process_conversation_batch(_make_messages()) == []


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


@pytest.mark.asyncio
async def test_summary_quality_normal_for_valid_response():
    """有效的 LLM 响应应标记为 summary_quality=normal。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"用户告知明天下午三点有重要会议需要参加",
            "topics":["会议"],
            "key_facts":["明天下午三点开会"],
            "sentiment":"neutral",
            "importance":0.8
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert metadata.get("summary_quality") == "normal"


@pytest.mark.asyncio
async def test_process_conversation_empty_summary_fails_strict_validation(monkeypatch):
    """自动萃取路径空 summary 属于严格校验失败，不得降级写入记忆。"""
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    llm = _DummyLLMProvider(
        """{
            "summary":"",
            "topics":["闲聊"],
            "key_facts":["用户问候"],
            "sentiment":"neutral",
            "importance":0.5
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    with pytest.raises(ValueError):
        await processor.process_conversation(
            messages=_make_messages(),
            is_group_chat=False,
            persona_id=None,
        )
    assert llm.text_chat.await_count == 3


@pytest.mark.asyncio
async def test_process_conversation_empty_key_facts_fails_strict_validation(monkeypatch):
    """自动萃取路径空 key_facts 属于严格校验失败，不得降级写入记忆。"""
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    llm = _DummyLLMProvider(
        """{
            "summary":"用户进行了一次普通对话",
            "topics":["闲聊"],
            "key_facts":[],
            "sentiment":"neutral",
            "importance":0.5
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    with pytest.raises(ValueError):
        await processor.process_conversation(
            messages=_make_messages(),
            is_group_chat=False,
            persona_id=None,
        )
    assert llm.text_chat.await_count == 3


def test_build_memory_from_structured_data_flags_low_quality_for_empty_summary():
    """手动结构化写入仍可保留 low quality 判定（不经过严格协议）。"""
    processor = MemoryProcessor(llm_provider=Mock(), context=None)

    _, metadata, _ = processor.build_memory_from_structured_data(
        {
            "summary": "",
            "topics": ["闲聊"],
            "key_facts": ["用户问候"],
            "sentiment": "neutral",
            "importance": 0.5,
        },
        is_group_chat=False,
        fallback_excerpt="fallback",
    )

    assert metadata["summary_quality"] == "low"


def test_build_memory_from_structured_data_flags_low_quality_for_missing_key_facts():
    """手动结构化写入仍可保留 low quality 判定（不经过严格协议）。"""
    processor = MemoryProcessor(llm_provider=Mock(), context=None)

    _, metadata, _ = processor.build_memory_from_structured_data(
        {
            "summary": "用户进行了一次普通对话",
            "topics": ["闲聊"],
            "key_facts": [],
            "sentiment": "neutral",
            "importance": 0.5,
        },
        is_group_chat=False,
        fallback_excerpt="fallback",
    )

    assert metadata["summary_quality"] == "low"


@pytest.mark.asyncio
async def test_summary_quality_low_for_generic_terms():
    """summary 包含泛化词（某用户、有人等）时应标记为 summary_quality=low。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"某用户提到了一些事情",
            "topics":["闲聊"],
            "key_facts":["某用户说了话"],
            "sentiment":"neutral",
            "importance":0.5
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert metadata.get("summary_quality") == "low"


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


def test_build_memory_from_structured_data_uses_standard_storage_format():
    processor = MemoryProcessor(llm_provider=Mock(), context=None)

    content, metadata, importance = processor.build_memory_from_structured_data(
        {
            "summary": "用户希望主动记忆工具复用自动总结格式",
            "topics": ["LivingMemory", "主动记忆"],
            "key_facts": ["主动记忆应复用 MemoryProcessor 格式化流程"],
            "event_time": "2026-07-19",
            "sentiment": "neutral",
            "importance": 0.8,
        },
        is_group_chat=False,
        fallback_excerpt="fallback",
    )

    assert content == metadata["canonical_summary"]
    assert metadata["neutral_summary"] == "用户希望主动记忆工具复用自动总结格式"
    assert "persona_summary" not in metadata
    assert metadata["topics"] == ["LivingMemory", "主动记忆"]
    assert metadata["key_facts"] == ["主动记忆应复用 MemoryProcessor 格式化流程"]
    assert metadata["event_time"] == "2026-07-19"
    assert metadata["sentiment"] == "neutral"
    assert metadata["interaction_type"] == "private_chat"
    assert metadata["summary_schema_version"] == "v3"
    assert metadata["summary_quality"] == "normal"
    assert importance == 0.8


def test_build_memory_from_structured_data_flags_low_quality_for_out_of_range_importance():
    """与自动总结路径一致：原始 importance 越界时应判为 low quality。"""
    processor = MemoryProcessor(llm_provider=Mock(), context=None)

    _, metadata, importance = processor.build_memory_from_structured_data(
        {
            "summary": "用户希望主动记忆工具复用自动总结格式",
            "topics": ["测试"],
            "key_facts": ["importance 越界"],
            "sentiment": "neutral",
            "importance": 1.5,
        },
        is_group_chat=False,
        fallback_excerpt="fallback",
    )

    assert metadata["summary_quality"] == "low"
    assert importance == 1.0


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


@pytest.mark.asyncio
async def test_process_group_chat_uses_neutral_summary_schema():
    llm = _DummyLLMProvider(
        """{
            "summary":"群聊讨论了 AI 工具的使用效果，建议内部部署私有化 LLM",
            "topics":["AI工具","数据安全"],
            "key_facts":["建议公司内部部署私有化 LLM","注意数据安全"],
            "participants":["张三","李四"],
            "sentiment":"positive",
            "importance":0.8
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, _ = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    assert "canonical_summary" in metadata
    assert "neutral_summary" in metadata
    assert "persona_summary" not in metadata
    assert metadata.get("summary_schema_version") == "v3"
    # canonical_summary 应包含 key_facts
    assert "私有化 LLM" in metadata["canonical_summary"]
    # content 应等于 canonical_summary
    assert content == metadata["canonical_summary"]


@pytest.mark.asyncio
async def test_process_group_chat_missing_participants_uses_default():
    """群聊 LLM 响应缺少 participants 字段时，应使用空列表默认值。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"群聊讨论了一些话题",
            "topics":["闲聊"],
            "key_facts":["大家聊了很多"],
            "sentiment":"neutral",
            "importance":0.5
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    # 缺少 participants 时应补充默认空列表
    assert "participants" in metadata
    assert isinstance(metadata["participants"], list)


@pytest.mark.asyncio
async def test_process_private_chat_no_participants_field():
    """私聊路径不应在 metadata 中包含 participants 字段。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"用户告知明天下午三点有重要会议",
            "topics":["会议"],
            "key_facts":["明天下午三点开会"],
            "sentiment":"neutral",
            "importance":0.8
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_messages(),
        is_group_chat=False,
        persona_id=None,
    )

    assert "participants" not in metadata
    assert metadata["interaction_type"] == "private_chat"


@pytest.mark.asyncio
async def test_process_group_chat_long_content():
    """群聊长内容（多条消息）应正常处理，不崩溃。"""
    long_messages = []
    for i in range(20):
        long_messages.append(
            Message(
                id=i + 1,
                session_id="aiocqhttp:GroupMessage:99999",
                role="user",
                content=f"成员{i % 5} 说：这是第 {i + 1} 条消息，内容比较详细，包含了很多信息。"
                * 3,
                sender_id=str(10000 + i % 5),
                sender_name=f"成员{i % 5}",
                group_id="99999",
                platform="aiocqhttp",
                metadata={},
            )
        )

    llm = _DummyLLMProvider(
        """{
            "summary":"群聊成员进行了多轮讨论，涉及多个话题",
            "topics":["群聊","讨论"],
            "key_facts":["多名成员参与讨论","讨论内容丰富"],
            "participants":["成员0","成员1","成员2","成员3","成员4"],
            "sentiment":"neutral",
            "importance":0.6
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    content, metadata, importance = await processor.process_conversation(
        messages=long_messages,
        is_group_chat=True,
        persona_id=None,
    )

    assert isinstance(content, str) and len(content) > 0
    assert metadata["interaction_type"] == "group_chat"
    assert len(metadata["participants"]) == 5
    assert 0.0 <= importance <= 1.0


@pytest.mark.asyncio
async def test_process_group_chat_quality_low_for_generic_terms():
    """群聊总结包含泛化词时，summary_quality 应为 low。"""
    llm = _DummyLLMProvider(
        """{
            "summary":"某用户在群里说了一些话",
            "topics":["闲聊"],
            "key_facts":["有人说话了"],
            "participants":["某用户"],
            "sentiment":"neutral",
            "importance":0.4
        }"""
    )
    processor = MemoryProcessor(llm_provider=llm, context=None)

    _, metadata, _ = await processor.process_conversation(
        messages=_make_group_messages(),
        is_group_chat=True,
        persona_id=None,
    )

    assert metadata.get("summary_quality") == "low"


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


def test_format_conversation_uses_placeholder_for_image_only_group_message():
    processor = MemoryProcessor(llm_provider=None, context=None)
    message = Message(
        id=1,
        session_id="g1",
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
        ],
        sender_id="u1",
        sender_name="张三",
        group_id="group1",
        platform="test",
        metadata={},
    )

    formatted = processor._format_conversation([message])

    assert "张三" in formatted
    assert "[图片消息]" in formatted
    assert "image_url" not in formatted


def test_text_tokenize_handles_empty_and_basic_cleaning() -> None:
    processor = TextProcessor()

    assert processor.tokenize("") == []
    assert processor.tokenize("   ") == []
    tokens = processor.tokenize("Visit https://example.com now!!!")
    assert "Visit" in tokens or "visit" in [token.lower() for token in tokens]


def test_text_tokenize_removes_common_stopwords() -> None:
    processor = TextProcessor()
    processor.add_stopwords(["我"])

    tokens = processor.tokenize("我 今天 去 图书馆", remove_stopwords=True)

    assert "我" not in tokens
    assert tokens


@pytest.mark.asyncio
async def test_text_load_stopwords_and_custom_words(tmp_path: Path) -> None:
    processor = TextProcessor()
    path = tmp_path / "stopwords.txt"
    path.write_text("# comment\nalpha\nbeta\n", encoding="utf-8")

    loaded = await processor.load_stopwords(str(path))
    processor.add_stopwords(["gamma"])

    assert "alpha" in loaded
    assert "alpha" in processor.stopwords
    assert "gamma" in processor.stopwords
    processor.remove_stopwords(["gamma"])
    assert "gamma" not in processor.stopwords


def test_text_word_frequency() -> None:
    frequency = TextProcessor().get_word_freq(["我 爱 编程", "编程 很 有趣"])

    assert isinstance(frequency, dict)
    assert frequency


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


@pytest.mark.asyncio
async def test_stopwords_manager_materializes_builtin_fallback(tmp_path: Path) -> None:
    manager = StopwordsManager(str(tmp_path))
    manager.builtin_stopwords_dir = tmp_path / "missing"

    stopwords_path = await manager.get_stopwords()
    loaded = await manager.load_stopwords()

    assert stopwords_path is not None
    assert Path(stopwords_path).exists()
    assert "的" in loaded


@pytest.mark.asyncio
async def test_text_processor_async_init_loads_builtin_stopwords(
    tmp_path: Path,
) -> None:
    processor = TextProcessor(str(tmp_path))

    await processor.async_init()

    assert "的" in processor.stopwords
    assert not (tmp_path / "stopwords_hit.txt").exists()


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


def test_strict_json_protocol_accepts_legacy_single_memory_json():
    """旧版单记忆 JSON（无 status/memories 包裹）仍兼容。"""
    processor = _new_processor()

    result = processor._parse_llm_response_batch(
        """{
            "summary":"张三明天下午三点开会",
            "topics":["会议"],
            "key_facts":["张三明天下午三点开会"],
            "event_time":"",
            "sentiment":"neutral",
            "importance":0.8
        }""",
        is_group_chat=False,
    )

    assert len(result) == 1
    assert result[0]["summary"] == "张三明天下午三点开会"
    assert result[0]["importance"] == 0.8


def test_strict_json_protocol_success_empty_array():
    processor = _new_processor()

    result = processor._parse_llm_response_batch(
        '{"status":"success","memories":[]}', is_group_chat=False
    )

    assert result == []


def test_strict_json_protocol_skip_empty_array_raises_llmextractionskip():
    processor = _new_processor()

    with pytest.raises(LLMExtractionSkip) as exc_info:
        processor._parse_llm_response_batch(
            '{"status":"skip","reason":"content_policy","memories":[]}',
            is_group_chat=False,
        )

    assert exc_info.value.reason == "content_policy"


def test_strict_json_protocol_rejects_invalid_status():
    processor = _new_processor()

    with pytest.raises(ValueError):
        processor._parse_llm_response_batch(
            '{"status":"error","memories":[]}', is_group_chat=False
        )


def test_strict_json_protocol_rejects_skip_with_non_empty_memories():
    processor = _new_processor()

    with pytest.raises(ValueError):
        processor._parse_llm_response_batch(
            """{"status":"skip","reason":"content_policy","memories":[
                {"summary":"s","topics":["t"],"key_facts":["f"],
                 "event_time":"","sentiment":"neutral","importance":0.5}]}""",
            is_group_chat=False,
        )


def test_strict_json_protocol_accepts_markdown_wrapped_json():
    """Markdown 代码块包裹仍应被剥离后解析。"""
    processor = _new_processor()

    result = processor._parse_llm_response_batch(
        """```json
{"status":"success","memories":[]}
```""",
        is_group_chat=False,
    )

    assert result == []


def test_strict_json_protocol_rejects_empty_response():
    processor = _new_processor()

    with pytest.raises(ValueError, match="为空"):
        processor._parse_llm_response_batch("   ", is_group_chat=False)


def test_strict_json_protocol_rejects_non_json_response():
    processor = _new_processor()

    with pytest.raises(ValueError, match="不是合法 JSON"):
        processor._parse_llm_response_batch("抱歉，我无法处理", is_group_chat=False)


def test_strict_json_protocol_rejects_root_array():
    processor = _new_processor()

    with pytest.raises(ValueError):
        processor._parse_llm_response_batch("[]", is_group_chat=False)


def test_strict_json_protocol_rejects_object_without_memories():
    processor = _new_processor()

    with pytest.raises(ValueError, match="缺少 memories"):
        processor._parse_llm_response_batch(
            '{"status":"success"}', is_group_chat=False
        )


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


def test_strict_json_protocol_rejects_non_object_memory_item():
    processor = _new_processor()

    with pytest.raises(ValueError, match="JSON 对象"):
        processor._parse_llm_response_batch(
            '{"status":"success","memories":["文本"]}', is_group_chat=False
        )


def test_strict_json_protocol_rejects_empty_or_non_string_summary():
    processor = _new_processor()

    for bad_summary in ("", "   ", 42):
        payload = json.dumps(
            {
                "status": "success",
                "memories": [_memory_payload(summary=bad_summary)],
            },
            ensure_ascii=False,
        )
        with pytest.raises(ValueError, match="summary"):
            processor._parse_llm_response_batch(payload, is_group_chat=False)


def test_strict_json_protocol_rejects_invalid_topics():
    processor = _new_processor()

    for bad_topics in ([], ["a"] * 5, [""], ["a", 1], "会议"):
        payload = json.dumps(
            {
                "status": "success",
                "memories": [_memory_payload(topics=bad_topics)],
            },
            ensure_ascii=False,
        )
        with pytest.raises(ValueError, match="topics"):
            processor._parse_llm_response_batch(payload, is_group_chat=False)


def test_strict_json_protocol_rejects_invalid_key_facts():
    processor = _new_processor()

    for bad_facts in ([], ["f"] * 6, [""], ["f", None], "事实"):
        payload = json.dumps(
            {
                "status": "success",
                "memories": [_memory_payload(key_facts=bad_facts)],
            },
            ensure_ascii=False,
        )
        with pytest.raises(ValueError, match="key_facts"):
            processor._parse_llm_response_batch(payload, is_group_chat=False)


def test_strict_json_protocol_rejects_non_string_event_time():
    processor = _new_processor()

    payload = json.dumps(
        {
            "status": "success",
            "memories": [_memory_payload(event_time=20260720)],
        },
        ensure_ascii=False,
    )
    with pytest.raises(ValueError, match="event_time"):
        processor._parse_llm_response_batch(payload, is_group_chat=False)


def test_strict_json_protocol_rejects_invalid_sentiment():
    processor = _new_processor()

    payload = json.dumps(
        {
            "status": "success",
            "memories": [_memory_payload(sentiment="angry")],
        },
        ensure_ascii=False,
    )
    with pytest.raises(ValueError, match="sentiment"):
        processor._parse_llm_response_batch(payload, is_group_chat=False)


def test_strict_json_protocol_rejects_invalid_importance():
    processor = _new_processor()

    for bad_importance in (True, 1.5, -0.1, "high"):
        payload = json.dumps(
            {
                "status": "success",
                "memories": [_memory_payload(importance=bad_importance)],
            },
            ensure_ascii=False,
        )
        with pytest.raises(ValueError, match="importance"):
            processor._parse_llm_response_batch(payload, is_group_chat=False)


def test_strict_json_protocol_rejects_invalid_group_participants():
    processor = _new_processor()

    payload = json.dumps(
        {
            "status": "success",
            "memories": [
                _memory_payload(extra={"participants": ["张三", 42]})
            ],
        },
        ensure_ascii=False,
    )
    with pytest.raises(ValueError, match="participants"):
        processor._parse_llm_response_batch(payload, is_group_chat=True)


def test_strict_json_protocol_accepts_valid_group_participants():
    processor = _new_processor()

    payload = json.dumps(
        {
            "status": "success",
            "memories": [
                _memory_payload(extra={"participants": ["张三", "李四"]})
            ],
        },
        ensure_ascii=False,
    )
    result = processor._parse_llm_response_batch(payload, is_group_chat=True)

    assert result[0]["participants"] == ["张三", "李四"]


def test_content_policy_rejection_detects_known_signatures():
    for signature in (
        "Input data may contain inappropriate content",
        "content_policy_violation",
        "ResponsibleAIPolicyViolation",
        "request was rejected as a result of the content filter",
        "输入数据可能包含不当内容",
    ):
        error = RuntimeError(f"provider: {signature}")
        assert MemoryProcessor._is_content_policy_rejection(error)


def test_content_policy_rejection_with_unrelated_status_code_returns_false():
    error = RuntimeError("request was rejected as a result of the content filter")
    error.status_code = 500
    assert not MemoryProcessor._is_content_policy_rejection(error)


def test_content_policy_rejection_plain_400_is_not_rejection():
    error = RuntimeError("Invalid model parameter: max_tokens")
    error.status_code = 400
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
async def test_process_conversation_batch_raises_skip_for_policy_response():
    """萃取批次收到 status=skip 时整体抛 LLMExtractionSkip，不产生记忆。"""
    processor = MemoryProcessor(
        llm_provider=_DummyLLMProvider(
            '{"status":"skip","reason":"content_policy","memories":[]}'
        ),
        context=None,
    )

    with pytest.raises(LLMExtractionSkip):
        await processor.process_conversation_batch(_make_messages())
