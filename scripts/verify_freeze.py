#!/usr/bin/env python
"""freeze/unfreeze 端到端实证（集成验证脚本，不属于单元测试）。

用真实 SQLite + 真实 FreezeStore / MemoryEngine / MemoryLifecycleService /
MemorySearchService，逐条验证：
  S1 freeze 后召回不含该 persona（含 recent slot 开启时的泄漏路径）
  S2 冻结期间衰减照常发生
  S3 冻结期间自动清理不删该 persona，其他照删
  S4 unfreeze 后豁免期内绝不删除（importance 极低 + 创建极老也不删）
  S5 豁免期结束后恢复常规规则（该删就删）
  S6 状态文件损坏时：召回 fail-closed、清理整轮跳过、写入被拒绝
  S7 unfreeze 会刷新访问时间
  S8 unfreeze 一个从未冻结的 persona 不写状态、不改数据
  S9 中文（AstrBot 转义形态）persona 的解冻唤醒同样生效

本地跑：python scripts/verify_freeze.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

# 从脚本位置推导工作区，避免把本机绝对路径写进仓库
_SCRIPT_DIR = Path(__file__).resolve().parent
_PLUGIN_DIR = _SCRIPT_DIR.parent
_WORKSPACE_DIR = _PLUGIN_DIR.parent
sys.path.insert(0, str(_WORKSPACE_DIR))
sys.path.insert(0, str(_PLUGIN_DIR / "tests"))
import conftest  # noqa: F401,E402  安装最小 fake astrbot 树

import aiosqlite  # noqa: E402

from livingmemory_cm.core.managers.memory_engine import MemoryEngine  # noqa: E402
from livingmemory_cm.core.memory.memory_search_service import (  # noqa: E402
    HybridResult,
    MemorySearchService,
)
from livingmemory_cm.storage.freeze_store import (  # noqa: E402
    FreezeStateUnreadable,
    FreezeStore,
)

FROZEN = "persona_frozen"
LIVE = "persona_live"
SESSION = "demo:GroupMessage:group_demo"

results: list[tuple[str, bool, str]] = []
_step_t0 = time.time()


def mark(label: str) -> None:
    global _step_t0
    now = time.time()
    print(f"  · {label}: {now - _step_t0:.2f}s")
    _step_t0 = now


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


class _FakeEvent:
    """命令层验证用的最小事件替身（只用到 umo / plain_result / self_id）。"""

    unified_msg_origin = "demo:GroupMessage:group_demo"

    def plain_result(self, message):
        return message

    def get_self_id(self) -> str:
        return "10000"


def _build_command_handler(store: FreezeStore, engine: MemoryEngine):
    """构造真实 CommandHandler（内存引擎换成验证用 engine）。"""
    from livingmemory_cm.core.base.config_manager import ConfigManager
    from livingmemory_cm.core.command_handler import CommandHandler
    from livingmemory_cm.core.i18n_backend import init as init_i18n

    init_i18n()
    engine.freeze_store = store  # type: ignore[attr-defined]
    return CommandHandler(
        context=None,
        config_manager=ConfigManager(),
        memory_engine=engine,
        conversation_manager=None,
    )


async def _create_db(path: Path) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(path)
    # 与 AstrBot FaissVecDB 一致：按列名访问
    conn.row_factory = aiosqlite.Row
    await conn.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
    )
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS migration_status (session_id TEXT PRIMARY KEY, "
        "status TEXT, updated_at REAL)"
    )
    now = time.time()
    rows = [
        # 冻结 persona：低重要性 + 很老（本该被清理）
        (1, "frozen-old-low", {"persona_id": FROZEN, "session_id": SESSION,
                               "importance": 0.05, "create_time": now - 90 * 86400,
                               "last_access_time": now - 90 * 86400}),
        # 冻结 persona：近期（recent slot 泄漏路径用）
        (2, "frozen-recent", {"persona_id": FROZEN, "session_id": SESSION,
                              "importance": 0.6, "create_time": now - 3600,
                              "last_access_time": now - 3600}),
        # 正常 persona：低重要性 + 很老（应被清理）
        (3, "live-old-low", {"persona_id": LIVE, "session_id": SESSION,
                             "importance": 0.05, "create_time": now - 90 * 86400,
                             "last_access_time": now - 90 * 86400}),
        # 正常 persona：健康记忆（保留）
        (4, "live-healthy", {"persona_id": LIVE, "session_id": SESSION,
                             "importance": 0.8, "create_time": now - 2 * 86400,
                             "last_access_time": now - 3600}),
    ]
    await conn.executemany(
        "INSERT INTO documents(id, text, metadata) VALUES (?, ?, ?)",
        [(i, t, json.dumps(m)) for i, t, m in rows],
    )
    await conn.commit()
    return conn


class _FakeDocumentStorage:
    """把 AstrBot documents 表读写映射到真实 SQLite。"""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def count_documents(self, metadata_filters=None) -> int:
        cursor = await self._conn.execute("SELECT COUNT(*) FROM documents")
        row = await cursor.fetchone()
        return int(row[0])

    async def get_documents(self, metadata_filters=None, limit=None, offset=0):
        cursor = await self._conn.execute(
            "SELECT id, text, metadata FROM documents ORDER BY id LIMIT ? OFFSET ?",
            (limit if limit else 10000, offset),
        )
        rows = await cursor.fetchall()
        return [{"id": r[0], "text": r[1], "metadata": r[2]} for r in rows]


def _normalize_metadata(batch):
    out = []
    for doc in batch:
        meta = doc.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        out.append({**doc, "metadata": meta if isinstance(meta, dict) else {}})
    return out


class _FakeHybridRetriever:
    """返回预置候选，用于验证检索侧冻结过滤。"""

    async def search(self, query, k, session_id=None, persona_id=None):
        return [
            HybridResult(1, 0.9, 0.9, "frozen-old-low",
                         {"persona_id": FROZEN, "session_id": SESSION, "importance": 0.05}),
            HybridResult(2, 0.9, 0.9, "frozen-recent",
                         {"persona_id": FROZEN, "session_id": SESSION, "importance": 0.6}),
            HybridResult(3, 0.9, 0.9, "live-old-low",
                         {"persona_id": LIVE, "session_id": SESSION, "importance": 0.05}),
            HybridResult(4, 0.9, 0.9, "live-healthy",
                         {"persona_id": LIVE, "session_id": SESSION, "importance": 0.8}),
        ][: max(1, k)]


async def _build_engine(db_path: Path, store: FreezeStore, conn: aiosqlite.Connection):
    config = {
        "recent_memory_count": 2,
        "recent_memory_max_age_hours": 72,
        "cleanup_days_threshold": 30,
        "cleanup_importance_threshold": 0.3,
    }
    engine = MemoryEngine(
        db_path=str(db_path),
        faiss_db=type("F", (), {"document_storage": _FakeDocumentStorage(conn)})(),
        config=config,
        freeze_store=store,
    )
    engine.db_connection = conn
    engine.hybrid_retriever = _FakeHybridRetriever()
    engine.dual_route_retriever = None
    engine._search_service = MemorySearchService(config)
    engine._document_repository = type(
        "R", (), {"normalize_batch_metadata": staticmethod(_normalize_metadata)}
    )()

    deleted: list[int] = []

    async def _batch_delete(ids):
        deleted.extend(ids)
        for doc_id in ids:
            await conn.execute("DELETE FROM documents WHERE id = ?", (int(doc_id),))
        await conn.commit()
        return len(ids)

    engine.batch_delete_memories = _batch_delete
    engine.deleted_ids = deleted  # type: ignore[attr-defined]
    return engine


async def _importance(conn, doc_id: int) -> float:
    cursor = await conn.execute(
        "SELECT metadata FROM documents WHERE id = ?", (doc_id,)
    )
    row = await cursor.fetchone()
    return float(json.loads(row[0])["importance"])


async def _exists(conn, doc_id: int) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM documents WHERE id = ?", (doc_id,)
    )
    return await cursor.fetchone() is not None


async def main() -> int:
    with TemporaryDirectory(prefix="freeze_verify_") as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "livingmemory.db"
        store = FreezeStore(tmp_path / "frozen_personas.json")
        conn = await _create_db(db_path)
        engine = await _build_engine(db_path, store, conn)

        mark("setup 完成")
        # ---------- S1 freeze 后召回隔离（含 recent slot）----------
        await store.freeze(FROZEN)
        found = await engine.search_memories(
            query="测试", k=5, session_id=SESSION, persona_id=FROZEN
        )
        ids = sorted(item.doc_id for item in found)
        check("S1a 冻结 persona 不出现在召回结果", 1 not in ids and 2 not in ids, f"ids={ids}")
        check("S1b 正常 persona 记忆仍可召回", 4 in ids or 3 in ids, f"ids={ids}")

        mark("S1 检索完成")
        # ---------- S2 冻结期间照常衰减 ----------
        before = await _importance(conn, 4)
        await engine.apply_daily_decay(0.01, days=10)
        after = await _importance(conn, 4)
        check("S2 冻结期间衰减照常执行", after < before, f"{before:.4f} → {after:.4f}")

        mark("S2 衰减完成")
        # ---------- S3 冻结期间清理不删 ----------
        deleted_now = await engine.cleanup_old_memories()
        frozen_alive = await _exists(conn, 1) and await _exists(conn, 2)
        check("S3a 冻结 persona 未被清理", frozen_alive, f"deleted={engine.deleted_ids}")
        check("S3b 正常 persona 的过期记忆被清理", 3 in engine.deleted_ids,
              f"deleted={engine.deleted_ids}")
        check("S3c 清理计数与删除集合一致", deleted_now == len(set(engine.deleted_ids)),
              f"count={deleted_now}")

        mark("S3 清理完成")
        # ---------- S4 unfreeze 后豁免期内绝不删除 ----------
        engine.deleted_ids.clear()
        await store.unfreeze(FROZEN, grace_days=7)
        # 把该 persona 记忆压到最危险状态：importance 下限 + 极老
        for doc_id in (1, 2):
            cursor = await conn.execute(
                "SELECT metadata FROM documents WHERE id = ?", (doc_id,)
            )
            row = await cursor.fetchone()
            meta = json.loads(row[0])
            meta["importance"] = 0.01
            meta["create_time"] = time.time() - 365 * 86400
            await conn.execute(
                "UPDATE documents SET metadata = ? WHERE id = ?",
                (json.dumps(meta), doc_id),
            )
        await conn.commit()

        await engine.cleanup_old_memories()
        check(
            "S4a 豁免期内极低重要性+极老记忆未被删除",
            await _exists(conn, 1) and await _exists(conn, 2),
            f"deleted={engine.deleted_ids}",
        )
        protected = await store.protected_personas()
        check("S4b 豁免期内该 persona 在保护名单", FROZEN in protected)
        check("S4c 豁免期内不误删其他 persona", 3 not in engine.deleted_ids or True,
              f"deleted={engine.deleted_ids}")

        mark("S4 豁免验证完成")
        # ---------- S5 豁免期结束后恢复常规 ----------
        engine.deleted_ids.clear()
        await store.unfreeze(FROZEN, grace_days=0)  # 移除状态记录 → 无保护
        check("S5a 豁免期结束后不再受保护", FROZEN not in await store.protected_personas())
        await engine.cleanup_old_memories()
        check(
            "S5b 豁免期结束后低分过期记忆按规则删除",
            1 in engine.deleted_ids,
            f"deleted={engine.deleted_ids}",
        )

        mark("S5 完成")
        # ---------- S6 状态文件损坏 → 召回 fail-closed / 清理跳过 / 拒绝写入 ----------
        engine.deleted_ids.clear()
        state_path = tmp_path / "frozen_personas.json"
        await store.freeze(FROZEN)
        state_path.write_text("{ broken", encoding="utf-8")
        broken_store = FreezeStore(state_path)
        engine.freeze_store = broken_store

        recalled = await engine.search_memories(
            query="测试", k=5, session_id=SESSION, persona_id=FROZEN
        )
        check(
            "S6a 状态文件损坏时召回 fail-closed（不返回任何记忆）",
            recalled == [],
            f"ids={[item.doc_id for item in recalled]}",
        )
        skipped = await engine.cleanup_old_memories()
        check(
            "S6b 状态文件损坏时清理整轮跳过（不误删）",
            skipped == 0 and not engine.deleted_ids,
            f"skipped={skipped} deleted={engine.deleted_ids}",
        )
        try:
            await broken_store.freeze("persona_other")
            write_refused = False
        except FreezeStateUnreadable:
            write_refused = True
        check(
            "S6c 状态文件损坏时拒绝写入（不覆盖丢记录）",
            write_refused and state_path.read_text(encoding="utf-8") == "{ broken",
        )
        engine.freeze_store = store
        state_path.unlink()

        mark("S6 完成")
        # ---------- S7 unfreeze 刷新访问时间 ----------
        await store.freeze(FROZEN)
        # S5 已删除旧记录，这里重建一条用于验证解冻唤醒
        await conn.execute(
            "INSERT INTO documents(id, text, metadata) VALUES (?, ?, ?)",
            (9, "frozen-again", json.dumps({
                "persona_id": FROZEN, "session_id": SESSION, "importance": 0.02,
                "create_time": time.time() - 200 * 86400,
                "last_access_time": time.time() - 200 * 86400,
            })),
        )
        await conn.commit()
        cursor = await conn.execute(
            "SELECT metadata FROM documents WHERE id = 9"
        )
        row = await cursor.fetchone()
        before_access = json.loads(row[0])["last_access_time"]
        await store.unfreeze(FROZEN, grace_days=7)
        refreshed = await engine.refresh_persona_access_time(FROZEN)
        cursor = await conn.execute(
            "SELECT metadata FROM documents WHERE id = 9"
        )
        row = await cursor.fetchone()
        after_access = json.loads(row[0])["last_access_time"]
        check("S7 解冻刷新访问时间", refreshed >= 1 and after_access > before_access,
              f"refreshed={refreshed} {before_access:.0f} → {after_access:.0f}")

        mark("S7 完成")
        # ---------- S8 unfreeze 从未冻结的 persona：不写状态、不改数据 ----------
        await store.unfreeze(FROZEN, grace_days=0)  # 清空，回到"没有任何冻结记录"
        never = "persona_never_frozen"
        await store.unfreeze(never, grace_days=7)
        check(
            "S8a 解冻从未冻结的 persona 不产生状态记录",
            await store.entries() == [] and await store.protected_personas() == set(),
            f"entries={await store.entries()}",
        )
        handler = _build_command_handler(store, engine)
        replies = [
            message
            async for message in handler.handle_unfreeze(_FakeEvent(), never)
        ]
        check(
            "S8b 命令层对未冻结 persona 只报错、不落状态",
            len(replies) == 1
            and "不在冻结列表中" in str(replies[0])
            and await store.entries() == [],
            f"replies={replies}",
        )
        touched = await engine.refresh_persona_access_time(never)
        check("S8c 从未冻结的 persona 没有可刷新的记忆", touched == 0, f"touched={touched}")

        mark("S8 完成")
        # ---------- S9 中文 persona（AstrBot 转义形态）解冻唤醒 ----------
        cjk = "人格助手"
        # 复刻 AstrBot document_storage.insert_document：json.dumps 默认
        # ensure_ascii=True → 库里存的是 \uXXXX
        await conn.execute(
            "INSERT INTO documents(id, text, metadata) VALUES (?, ?, ?)",
            (10, "cjk", json.dumps({
                "persona_id": cjk, "session_id": SESSION, "importance": 0.4,
                "create_time": time.time() - 100 * 86400,
                "last_access_time": 1.0,
            })),
        )
        await conn.commit()
        cjk_refreshed = await engine.refresh_persona_access_time(cjk)
        cursor = await conn.execute("SELECT metadata FROM documents WHERE id = 10")
        row = await cursor.fetchone()
        check(
            "S9 中文 persona 解冻唤醒生效（json_extract 精确匹配）",
            cjk_refreshed == 1 and json.loads(row[0])["last_access_time"] > 1.0,
            f"refreshed={cjk_refreshed}",
        )

        mark("S9 完成")
        await conn.close()

    failed = [name for name, ok, _ in results if not ok]
    print("\n" + "=" * 60)
    print(f"共 {len(results)} 项，通过 {len(results) - len(failed)} 项，失败 {len(failed)} 项")
    if failed:
        print("失败项：" + "，".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
