"""
Tests for PluginPageApi — WebUI REST API endpoints and helpers.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
from livingmemory_cm.core.page_api import (
    PAGE_API_PREFIX,
    PLUGIN_NAME,
    PluginPageApi,
)

# ---------------------------------------------------------------------------
# Fake / stub helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeMemoryEngine:
    db_path: str = ":memory:"
    graph_store: Any = None
    stats: dict = field(
        default_factory=lambda: {
            "total_memories": 42,
            "status_breakdown": {"active": 30, "archived": 8, "deleted": 4},
            "sessions": {"s1": 10, "s2": 5},
        }
    )

    async def get_statistics(self, persona_id=None):
        return self.stats

    async def search_memories(self, query, k=5, session_id=None, persona_id=None):
        return []

    async def get_memory(self, memory_id: int):
        return None

    async def add_memory(self, **kwargs):
        return 999

    async def delete_memory(self, memory_id: int):
        return True

    async def update_memory(self, memory_id: int, updates: dict):
        return True

    async def batch_delete_memories(self, memory_ids: list[int]):
        return len(memory_ids)

    async def close(self):
        pass


class FakeInitializer:
    def __init__(self):
        self.memory_engine = FakeMemoryEngine()
        self.conversation_manager = None
        self.data_dir = "/tmp/test_plugin"


class FakePlugin:
    def __init__(self, *, ready=True, memory_engine=None):
        self._ready = ready
        self._fail_message = "" if ready else "插件尚未就绪"
        self.initializer = FakeInitializer()
        if memory_engine:
            self.initializer.memory_engine = memory_engine
        self._api_routes = []

    async def _ensure_plugin_ready(self):
        return self._ready, self._fail_message

    @property
    def context(self):
        ctx = MagicMock()

        def _register(route, handler, methods, desc):
            self._api_routes.append((route, handler, methods, desc))

        ctx.register_web_api = _register
        return ctx


# ---------------------------------------------------------------------------
# Safe request mocking — Quart's request is a LocalProxy that throws
# RuntimeError when accessed outside a request context. Use patch.dict
# on the module's __dict__ to replace it without triggering the proxy.
# ---------------------------------------------------------------------------


def _mock_page_request(**overrides):
    """Build a MagicMock suitable for standing in as ``quart.request``.

    *overrides* can include ``args`` (dict), ``get_json`` (return value),
    and ``method`` (str).  All keys are optional.
    """
    req = MagicMock()

    args_mock = MagicMock()
    args_dict = overrides.get("args", {})
    args_mock.get.side_effect = lambda key, default=None: args_dict.get(key, default)
    req.args = args_mock

    json_value = overrides.get("get_json", {})
    req.get_json = AsyncMock(return_value=json_value)

    req.method = overrides.get("method", "GET")
    return req


@contextmanager
def _patch_page_request(req: MagicMock):
    """Temporarily replace ``page_api.request`` with *req*."""
    import livingmemory_cm.core.page_api as mod
    import livingmemory_cm.core.page_api_modules.graph_handler as graph_mod
    import livingmemory_cm.core.page_api_modules.memory_handler as memory_mod
    import livingmemory_cm.core.page_api_modules.recall_handler as recall_mod
    import livingmemory_cm.core.page_api_modules.stats_handler as stats_mod

    # Patch all modules that use request
    modules = [mod, memory_mod, recall_mod, graph_mod, stats_mod]
    old_values = []

    for module in modules:
        ns = vars(module)
        old_values.append((ns, ns.get("request")))
        ns["request"] = req

    try:
        yield
    finally:
        for ns, old in old_values:
            if old is not None:
                ns["request"] = old
            else:
                ns.pop("request", None)


# Alias for brevity in tests
def _qp(req=None, **kw):
    """Quick-patch: ``with _qp(mock_req): ...``"""
    if req is None:
        req = _mock_page_request(**kw)
    return _patch_page_request(req)


# ---------------------------------------------------------------------------
# Helper method unit tests (no plugin needed)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Endpoint tests (with mocked plugin)
# ---------------------------------------------------------------------------


@pytest.fixture
def api():
    plugin = FakePlugin()
    return PluginPageApi(plugin)


@pytest.fixture
def api_not_ready():
    plugin = FakePlugin(ready=False)
    return PluginPageApi(plugin)


class TestListMemories:
    @pytest.mark.asyncio
    async def test_type_filter_and_sort_are_applied_in_sql(self, api, tmp_path):
        db_path = tmp_path / "memories.db"
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY,
                    doc_id TEXT,
                    text TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            rows = [
                (
                    1,
                    "1",
                    "low preference",
                    {"memory_type": "PREFERENCE", "importance": 0.3, "create_time": 10},
                ),
                (
                    2,
                    "2",
                    "high preference",
                    {"memory_type": "PREFERENCE", "importance": 0.9, "create_time": 20},
                ),
                (
                    3,
                    "3",
                    "other fact",
                    {"memory_type": "FACT", "importance": 1.0, "create_time": 30},
                ),
            ]
            for memory_id, doc_id, text, metadata in rows:
                await db.execute(
                    """
                    INSERT INTO documents
                        (id, doc_id, text, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        doc_id,
                        text,
                        json.dumps(metadata),
                        "created",
                        "updated",
                    ),
                )
            await db.commit()

        api.plugin.initializer.memory_engine.db_path = str(db_path)
        req = _mock_page_request(
            args={
                "page": "1",
                "page_size": "20",
                "session_id": "",
                "keyword": "",
                "status": "all",
                "type": "PREFERENCE",
                "sort": "importance_desc",
            }
        )

        with _patch_page_request(req):
            result = await api.list_memories()

        assert result["status"] == "ok"
        assert result["data"]["total"] == 2
        assert result["data"]["filters"]["type"] == "PREFERENCE"
        assert result["data"]["sort"] == "importance_desc"
        assert [item["id"] for item in result["data"]["items"]] == [2, 1]

class TestExportMemories:
    """导出端点：守的是导出集合错位、静默截断、CSV 结构损坏三类看不见的错误。"""

    @staticmethod
    async def _prepare_db(tmp_path, rows):
        db_path = tmp_path / "export.db"
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY,
                    doc_id TEXT,
                    text TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            for memory_id, doc_id, text, metadata in rows:
                await db.execute(
                    """
                    INSERT INTO documents (id, doc_id, text, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (memory_id, doc_id, text, json.dumps(metadata), "c", "u"),
                )
            await db.commit()
        return str(db_path)

    @pytest.mark.asyncio
    async def test_export_applies_same_filters_as_list(self, api, tmp_path):
        """导出必须与列表同源过滤：否则文件里的记忆集合和界面所见不一致。"""
        rows = [
            (
                1,
                "1",
                "persona_a 的偏好",
                {
                    "persona_id": "persona_a",
                    "status": "active",
                    "memory_type": "PREFERENCE",
                    "create_time": 10,
                },
            ),
            (
                2,
                "2",
                "persona_b 的偏好",
                {
                    "persona_id": "persona_b",
                    "status": "active",
                    "memory_type": "PREFERENCE",
                    "create_time": 20,
                },
            ),
        ]
        api.plugin.initializer.memory_engine.db_path = await self._prepare_db(
            tmp_path, rows
        )

        args = {"persona_id": "persona_a", "status": "all", "sort": "created_desc"}
        with _patch_page_request(_mock_page_request(args=args)):
            listed = await api.list_memories()
        with _patch_page_request(_mock_page_request(args=args)):
            content, status, headers = await api.export_memories()

        assert status == 200
        assert "application/json" in headers["Content-Type"]
        payload = json.loads(content)
        assert payload["count"] == 1
        assert payload["truncated"] is False
        assert payload["filtered_by"]["persona_id"] == "persona_a"
        assert [item["id"] for item in payload["memories"]] == [
            item["id"] for item in listed["data"]["items"]
        ] == [1]
        # 文件名自带条数，页面据此提示；不含 _partial 表示未截断
        assert headers["Content-Disposition"].endswith('_1.json"')

    @pytest.mark.asyncio
    async def test_export_flags_truncation_and_zero_limit_means_max(self, api, tmp_path):
        """触达上限必须显式标记：静默少给记忆会被当成"库就这么多"。"""
        rows = [
            (
                idx,
                str(idx),
                f"记忆 {idx}",
                {"persona_id": "persona_a", "status": "active", "create_time": idx},
            )
            for idx in (1, 2, 3)
        ]
        api.plugin.initializer.memory_engine.db_path = await self._prepare_db(
            tmp_path, rows
        )

        args = {"persona_id": "persona_a", "limit": "1"}
        with _patch_page_request(_mock_page_request(args=args)):
            content, _, headers = await api.export_memories()
        payload = json.loads(content)
        assert payload["count"] == 1
        assert payload["total_matched"] == 3
        assert payload["truncated"] is True
        assert "_1_partial.json" in headers["Content-Disposition"]

        args = {"persona_id": "persona_a", "limit": "0"}
        with _patch_page_request(_mock_page_request(args=args)):
            content, _, headers = await api.export_memories()
        payload = json.loads(content)
        assert payload["count"] == 3
        assert payload["truncated"] is False
        assert "_3.json" in headers["Content-Disposition"]

    @pytest.mark.asyncio
    async def test_export_csv_escapes_cells_and_neutralizes_formulas(self, api, tmp_path):
        """CSV 结构损坏与公式注入都看不出来，必须守住。"""
        rows = [
            (
                1,
                "1",
                '带逗号,带引号"和换行\n的内容',
                {
                    "persona_id": "persona_a",
                    "status": "active",
                    "memory_type": "FACT",
                    "importance": 0.5,
                    "key_facts": ['=HYPERLINK("http://example.invalid")'],
                    "participants": ["群友A", "群友B"],
                },
            )
        ]
        api.plugin.initializer.memory_engine.db_path = await self._prepare_db(
            tmp_path, rows
        )

        args = {"format": "csv", "persona_id": "persona_a"}
        with _patch_page_request(_mock_page_request(args=args)):
            content, status, headers = await api.export_memories()

        assert status == 200
        assert "text/csv" in headers["Content-Type"]
        assert content.startswith("\ufeff"), "CSV 需带 BOM，否则 Excel 打开中文乱码"
        assert "ID,doc_id,类型" in content
        # 引号按 RFC 4180 双写，换行留在引号内 → 数据行仍是 1 行数据 + 结尾换行
        assert '"带逗号,带引号""和换行\n的内容"' in content
        # 前导 = 的单元格加单引号，避免表格软件当公式执行
        assert '''"'=HYPERLINK(""http://example.invalid"")"''' in content
        # 列表字段用顿号连接
        assert "群友A、群友B" in content

    @pytest.mark.asyncio
    async def test_export_plugin_not_ready_returns_non_2xx(self, api_not_ready):
        """未就绪必须是非 2xx：下载通道不解析错误包装，否则会存下错误 JSON 文件。"""
        with _patch_page_request(_mock_page_request()):
            content, status, headers = await api_not_ready.export_memories()
        assert status >= 400
        assert json.loads(content)["status"] == "error"
        assert "application/json" in headers["Content-Type"]


class TestUpdateMemory:
    @pytest.mark.asyncio
    async def test_importance_display_scale_preserves_one_point_zero(self, api):
        api.plugin.initializer.memory_engine.update_memory = AsyncMock(return_value=True)
        req = _mock_page_request(
            get_json={
                "memory_id": 1,
                "field": "importance",
                "value": 1.0,
                "value_scale": "display",
            }
        )
        with _patch_page_request(req):
            with patch(
                "livingmemory_cm.core.page_api_modules.memory_handler"
                ".MemoryHandler._get_memory_record",
                return_value={"id": 1, "text": "hello", "metadata": {}},
            ):
                result = await api.update_memory()

        assert result["status"] == "ok"
        updates = api.plugin.initializer.memory_engine.update_memory.call_args.args[1]
        assert updates["importance"] == 0.1

    @pytest.mark.asyncio
    async def test_content_update_uses_default_for_legacy_importance(self, api):
        req = _mock_page_request(
            get_json={
                "memory_id": 1,
                "field": "content",
                "value": "new content",
            }
        )
        with _patch_page_request(req):
            memory = {
                "id": 1,
                "text": "old content",
                "metadata": {
                    "session_id": "s1",
                    "persona_id": "p1",
                    "importance": "default",
                },
            }
            with patch(
                "livingmemory_cm.core.page_api_modules.memory_handler"
                ".MemoryHandler._get_memory_record",
                return_value=memory,
            ):
                api.plugin.initializer.memory_engine.add_memory = AsyncMock(
                    return_value=999
                )
                result = await api.update_memory()

        assert result["status"] == "ok"
        assert (
            api.plugin.initializer.memory_engine.add_memory.call_args.kwargs[
                "importance"
            ]
            == 0.5
        )


class TestGraphEndpoints:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("session_filter", [None, "None"])
    async def test_query_expands_node_hits_without_text_recall(self, session_filter):
        snapshot = {
            "nodes": [
                {
                    "id": 56,
                    "type": "person",
                    "weight": 1.0,
                    "degree": 0,
                    "label": "luna",
                }
            ],
            "edges": [],
            "entries": [
                {
                    "id": 501,
                    "memory_id": 123,
                    "entry_type": "summary",
                    "relation_type": "mentions",
                    "content": "Luna appears in this memory",
                    "metadata": {"session_id": "s1"},
                    "node_ids": [56],
                }
            ],
            "memories": [
                {
                    "memory_id": 123,
                    "summary": "Luna appears in this memory",
                    "importance": 0.7,
                    "entry_count": 1,
                    "node_count": 1,
                    "edge_count": 0,
                }
            ],
        }
        graph_store = SimpleNamespace(
            search_nodes_by_tokens=AsyncMock(
                return_value=[
                    {
                        "id": 56,
                        "node_key": "person:luna",
                        "node_type": "person",
                        "node_value": "luna",
                        "canonical_value": "luna",
                        "metadata": {},
                    }
                ]
            ),
            get_entries_for_node_ids=AsyncMock(
                return_value=[
                    {
                        "entry_id": 501,
                        "source_memory_id": 123,
                        "content": "Luna appears in this memory",
                        "metadata": {"session_id": "s1"},
                        "score": 0.85,
                    }
                ]
            ),
            get_subgraph_for_memories=AsyncMock(return_value=snapshot),
        )
        engine = FakeMemoryEngine(graph_store=graph_store)
        engine.search_memories = AsyncMock(return_value=[])
        api = PluginPageApi(FakePlugin(memory_engine=engine))
        req = _mock_page_request(
            get_json={
                "query": "luna",
                "session_id": session_filter,
                "persona_id": "undefined",
            }
        )

        with _patch_page_request(req):
            result = await api.query_graph()

        assert result["status"] == "ok"
        data = result["data"]
        assert data["filters"]["session_id"] is None
        assert data["filters"]["persona_id"] is None
        assert data["matched_node_ids"] == [56]
        assert data["matched_memory_ids"] == [123]
        assert data["summary"]["visible_node_count"] == 1
        assert data["snapshot"]["nodes"][0]["highlighted"] is True
        assert data["retrieval"]["items"][0]["source"] == "graph_node"
        engine.search_memories.assert_awaited_once()
        assert engine.search_memories.call_args.kwargs["session_id"] is None
        graph_store.get_entries_for_node_ids.assert_awaited_once()
        assert (
            graph_store.get_entries_for_node_ids.call_args.kwargs["session_id"] is None
        )
        graph_store.get_subgraph_for_memories.assert_awaited_once()
        assert graph_store.get_subgraph_for_memories.call_args.args[0] == [123]


# ---------------------------------------------------------------------------
# Ensure plugin ready helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


DASHBOARD = Path(__file__).parents[1] / "pages" / "dashboard"


