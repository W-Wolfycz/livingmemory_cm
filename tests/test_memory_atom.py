"""Tests for memory atom models, classifier, and lifecycle management."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from livingmemory_cm.core.models.memory_atom import (
    AtomStatus,
    AtomType,
    DecayType,
    MemoryAtom,
    compute_ttl,
)
from livingmemory_cm.core.processors.atom_classifier import (
    _classify_single,
    classify_atoms,
)
from livingmemory_cm.storage.atom_store import AtomStore

# ---------- TTL computation ----------


def test_compute_ttl_planned_with_event_time() -> None:
    future = time.time() + 7 * 86400.0  # 7 days from now
    ttl, decay = compute_ttl(AtomType.PLANNED, importance=0.8, event_time=future)
    # base=2, days_until=~7, ttl= (2+7) * (0.5+0.8) = 9 * 1.3 = 11.7
    assert 8 <= ttl <= 15
    assert decay == DecayType.STEP


# ---------- Decay functions ----------


def test_memory_atom_exponential_decay() -> None:
    atom = MemoryAtom(
        parent_memory_id=1,
        atom_type=AtomType.EPISODIC,
        content="test",
        ttl_days=10.0,
        decay_type=DecayType.EXPONENTIAL,
        last_accessed_at=time.time(),
    )
    score_now = atom.compute_temporal_score()
    assert score_now >= 0.99  # fresh

    score_old = atom.compute_temporal_score(time.time() + 10 * 86400.0)
    assert score_old < 0.3  # well past half-life


# ---------- Atom classifier ----------


# ---------- AtomStore persistence ----------


@pytest.mark.asyncio
async def test_atom_store_fts_search(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_atoms_fts.db")
    store = AtomStore(db_path)
    await store.initialize()

    await store.insert(MemoryAtom(parent_memory_id=1, content="张三喜欢喝咖啡"))
    await store.insert(MemoryAtom(parent_memory_id=1, content="李四喜欢吃火锅"))
    await store.insert(MemoryAtom(parent_memory_id=2, content="明天项目deadline"))

    results = await store.search_fts("咖啡", limit=5)
    assert len(results) >= 1
    assert any("咖啡" in r.content for r in results)


@pytest.mark.asyncio
async def test_atom_store_reinforce(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_atoms_reinforce.db")
    store = AtomStore(db_path)
    await store.initialize()

    atom = MemoryAtom(
        parent_memory_id=1,
        atom_type=AtomType.FACTUAL,
        content="test",
        importance=0.8,
        confidence=0.7,
    )
    atom_id = await store.insert(atom)
    original = await store.get(atom_id)
    assert original is not None
    original_ttl = original.ttl_days
    original_count = original.reinforcement_count

    await store.reinforce(atom_id, new_confidence=0.9)
    reinforced = await store.get(atom_id)
    assert reinforced is not None
    assert reinforced.reinforcement_count == original_count + 1
    # EMA: 0.7 * 0.7 + 0.9 * 0.3 = 0.49 + 0.27 = 0.76
    assert 0.7 < reinforced.confidence < 0.85
    assert reinforced.ttl_days > original_ttl


@pytest.mark.asyncio
async def test_atom_store_delete_by_parent(tmp_path: Path) -> None:
    db_path = str(tmp_path / "test_atoms_delete.db")
    store = AtomStore(db_path)
    await store.initialize()

    await store.insert(MemoryAtom(parent_memory_id=10, content="fact 1"))
    await store.insert(MemoryAtom(parent_memory_id=10, content="fact 2"))
    await store.insert(MemoryAtom(parent_memory_id=20, content="fact 3"))

    deleted = await store.delete_by_parent(10)
    assert deleted == 2

    remaining = await store.get_by_parent(20)
    assert len(remaining) == 1


def test_classify_planned_wins_over_preference() -> None:
    atom_type, confidence, event_time = _classify_single("明天下午去喝咖啡")
    assert atom_type == AtomType.PLANNED
    assert event_time is not None


# ---------- TTL boundary conditions ----------


def test_compute_ttl_max_reinforcement() -> None:
    ttl, decay = compute_ttl(AtomType.FACTUAL, importance=0.5, reinforcement_count=20)
    # 180 * 1.0 * min(1.5, 1.0+2.0) = 180 * 1.0 * 1.5 = 270
    assert ttl >= 180


def test_compute_ttl_minimum_ttl_clamp() -> None:
    ttl, _ = compute_ttl(AtomType.UNKNOWN, importance=0.0, reinforcement_count=0)
    assert ttl >= 1.0


# ---------- Decay edge cases ----------


def test_decay_zero_ttl_clamped_to_minimum_decay() -> None:
    atom = MemoryAtom(
        parent_memory_id=1,
        ttl_days=0.5,
        decay_type=DecayType.EXPONENTIAL,
        last_accessed_at=time.time(),
    )
    score = atom.compute_temporal_score(time.time() + 3 * 86400)
    assert 0.0 < score < 1.0


# ---------- AtomStore edge cases ----------


# ---------- AtomLifecycleManager ----------


@pytest.mark.asyncio
async def test_lifecycle_manager_run_maintenance(tmp_path: Path) -> None:
    from livingmemory_cm.core.managers.atom_lifecycle_manager import (
        AtomLifecycleManager,
    )

    db_path = str(tmp_path / "test_atoms_lifecycle_mgr.db")
    store = AtomStore(db_path)
    await store.initialize()

    mgr = AtomLifecycleManager(store)
    result = await mgr.run_maintenance()
    assert "expired" in result
    assert "forgotten" in result
    assert isinstance(result["expired"], int)
    assert isinstance(result["forgotten"], int)


@pytest.mark.asyncio
async def test_lifecycle_manager_logs_and_backs_off_on_error(monkeypatch):
    import livingmemory_cm.core.managers.atom_lifecycle_manager as lifecycle_module
    from livingmemory_cm.core.managers.atom_lifecycle_manager import (
        AtomLifecycleManager,
    )

    class FailingStore:
        async def expire_stale_atoms(self):
            raise RuntimeError("maintenance failed")

    sleep_calls: list[float] = []
    mgr = AtomLifecycleManager(FailingStore(), {"atom_maintenance_interval_hours": 1})
    mgr._running = True

    async def fake_sleep(seconds: float):
        sleep_calls.append(seconds)
        mgr._running = False

    error_mock = Mock()
    monkeypatch.setattr(lifecycle_module.logger, "error", error_mock)
    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    await mgr._maintenance_loop()

    error_mock.assert_called_once()
    assert error_mock.call_args.args[0] == "[livingmemory_cm][atom] 维护任务异常"
    assert error_mock.call_args.kwargs["exc_info"] is True
    assert sleep_calls == [60.0]


@pytest.mark.asyncio
async def test_graph_store_semantic_edge_merge(tmp_path: Path) -> None:
    from livingmemory_cm.core.models.graph_models import (
        GraphEdge,
        GraphNode,
    )
    from livingmemory_cm.storage.graph_store import GraphStore

    db_path = str(tmp_path / "test_semantic_edge.db")
    store = GraphStore(db_path)
    await store.initialize()

    # Create two nodes
    node_a = GraphNode(
        node_type="topic", value="项目Alpha", canonical_value="项目alpha"
    )
    node_b = GraphNode(node_type="fact", value="明天发布", canonical_value="明天发布")
    node_a_id = await store.upsert_node(node_a)
    node_b_id = await store.upsert_node(node_b)

    node_map = {node_a.node_key: node_a_id, node_b.node_key: node_b_id}

    # First edge from memory 1
    edge1 = GraphEdge(
        source_key=node_a.node_key,
        target_key=node_b.node_key,
        relation_type="describes",
        source_memory_id=1,
        confidence=0.8,
    )
    e1_id = await store.add_edge(edge1, node_map)

    # Second edge from memory 2 (same semantic relation)
    edge2 = GraphEdge(
        source_key=node_a.node_key,
        target_key=node_b.node_key,
        relation_type="describes",
        source_memory_id=2,
        confidence=0.9,
    )
    e2_id = await store.add_edge(edge2, node_map)

    # Should merge: same id returned
    assert e1_id == e2_id

    # Verify EMA-updated confidence: 0.8 * 0.7 + 0.9 * 0.3 = 0.56 + 0.27 = 0.83
    async with store._connect() as db:
        cursor = await db.execute(
            "SELECT confidence, weight FROM graph_edges WHERE id = ?", (e1_id,)
        )
        row = await cursor.fetchone()
    assert row is not None
    assert 0.8 < float(row[0]) <= 0.9  # confidence between old and new
    assert float(row[1]) > 1.0  # weight accumulated

    await store.delete_memory(1)
    await store.delete_memory(2)


# ---------- GraphExtractor with atoms ----------


def test_graph_extractor_atoms_no_entities_fallback() -> None:
    from livingmemory_cm.core.models.memory_atom import AtomType, MemoryAtom
    from livingmemory_cm.core.processors.graph_extractor import (
        GraphExtractor,
    )

    extractor = GraphExtractor()
    # Atom without entities — _add_node("fact") fails if canonicalize returns ""
    # so it should fall through to the "no entries" branch and create a summary entry
    atom = MemoryAtom(
        parent_memory_id=1,
        atom_type=AtomType.UNKNOWN,
        content="x",
        entities=[],
        confidence=0.6,
    )
    result = extractor.extract(1, "", {}, atoms=[atom])
    # The fact node key would be "fact:x" which should canonicalize to "x"
    # Actually let me check: if canonicalize("x") returns "x" (non-empty), then
    # we'll get a fact node + entry. Let me use a proper test instead.
    atom2 = MemoryAtom(
        parent_memory_id=1,
        atom_type=AtomType.UNKNOWN,
        content="测试内容",
        entities=[],
        confidence=0.6,
    )
    result = extractor.extract(1, "", {}, atoms=[atom2])
    assert len(result.nodes) >= 1
    assert len(result.entries) >= 1


# ---------- Backward compatibility ----------


def test_classify_atoms_from_metadata_atom_disabled() -> None:
    """When atom_enabled is False, classify_atoms_from_metadata returns empty list."""
    from livingmemory_cm.core.processors.memory_processor import (
        MemoryProcessor,
    )

    processor = MemoryProcessor(config={"atom_enabled": False})
    atoms = processor.classify_atoms_from_metadata(
        metadata={"key_facts": ["测试1", "测试2"]},
    )
    assert atoms == []


def test_legacy_extract_path_unchanged() -> None:
    """When no atoms provided, extract() uses legacy path (backward compatible)."""
    from livingmemory_cm.core.processors.graph_extractor import (
        GraphExtractor,
    )

    extractor = GraphExtractor()
    result = extractor.extract(
        1,
        content="讨论总结",
        metadata={
            "topics": ["项目进度"],
            "key_facts": ["今天讨论了Q3计划"],
            "participants": ["张三"],
            "importance": 0.7,
            "canonical_summary": "今天讨论了Q3计划",
        },
        atoms=None,
    )
    # Legacy path should produce nodes/edges/entries from metadata
    assert len(result.nodes) >= 2  # topic + fact + person
    assert len(result.entries) >= 1
    assert len(result.edges) >= 1


# ---------- event_time parsing ----------


def test_event_time_parsing_bare_weekday_uses_next_occurrence(monkeypatch) -> None:
    from livingmemory_cm.core.processors import atom_classifier

    # 2024-06-05 is Wednesday; bare "周二" should point to the next Tuesday.
    now = 1717588800.0
    monkeypatch.setattr(atom_classifier.time, "time", lambda: now)

    result = atom_classifier._parse_event_time("周二开会")

    assert result is not None
    assert result == now + 6 * 86400


# ---------- compute_ttl type coverage ----------


# ---------- AtomStore stats ----------


