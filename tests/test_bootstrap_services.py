"""初始化子服务的独立回归测试。"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from livingmemory_cm.core.base.exceptions import (
    ConfigurationError,
    DatabaseError,
    InitializationError,
    LivingMemoryException,
    MemoryProcessingError,
    ProviderNotReadyError,
    RetrievalError,
    ValidationError,
)
from livingmemory_cm.core.bootstrap import (
    EmbeddingFingerprintService,
    EmbeddingIndexBootstrapService,
    EmbeddingIndexSpec,
    FaissBootstrapService,
)
import livingmemory_cm.core.bootstrap.faiss_bootstrap as faiss_bootstrap_module
from livingmemory_cm.core.plugin_initializer import PluginInitializer
from livingmemory_cm.storage.db_migration import DBMigration


def test_faiss_bootstrap_sanitizes_non_ascii_path() -> None:
    sanitized = FaissBootstrapService.sanitize_path(
        r"C:\Users\测试用户\livingmemory.index"
    )

    assert "测试用户" not in sanitized
    assert "[***]" in sanitized
    assert sanitized.endswith("livingmemory.index")


class _FakeIndex:
    def __init__(self, dimension: int, ids: list[int] | None = None) -> None:
        self.d = dimension
        self.id_map = np.asarray(ids or [], dtype=np.int64)
        self.ntotal = len(self.id_map)

    def add_with_ids(self, vectors, ids) -> None:
        del vectors
        self.id_map = np.concatenate(
            [self.id_map, np.asarray(ids, dtype=np.int64)]
        )
        self.ntotal = len(self.id_map)


class _FakeFaiss:
    @staticmethod
    def IndexFlatL2(dimension: int):
        return SimpleNamespace(d=dimension)

    @staticmethod
    def IndexIDMap(base):
        return _FakeIndex(base.d)

    @staticmethod
    def vector_to_array(values):
        return np.asarray(values)


class _FakeIndexBootstrap:
    def __init__(self, index: _FakeIndex) -> None:
        self.index = index
        self.writes = 0

    def faiss_read_index_safe(self, path: str):
        del path
        return self.index

    def faiss_write_index_safe(self, index, path: str) -> None:
        self.index = _FakeIndex(index.d, list(index.id_map))
        self.writes += 1
        with open(path, "wb") as handle:
            handle.write(b"new-index")


class _FakeEmbeddingProvider:
    def __init__(
        self, model: str, *, fail: bool = False, max_batch: int | None = None
    ) -> None:
        self.model = model
        self.fail = fail
        self.max_batch = max_batch
        self.calls = 0
        self.batches: list[int] = []
        self.provider_config = {
            "id": "embedding_demo",
            "type": "openai_embedding",
            "embedding_model": model,
            "embedding_api_key": "must-not-be-written",
        }

    def get_dim(self) -> int:
        return 2

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider unavailable")
        if self.max_batch is not None and len(texts) > self.max_batch:
            raise RuntimeError(
                "Error code: 400 - InvalidParameter: Value error, batch size is "
                f"invalid, it should not be larger than {self.max_batch}.: input.contents"
            )
        self.batches.append(len(texts))
        return [[float(len(text)), 1.0] for text in texts]


def _make_documents_db(path) -> None:
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT)")
        db.executemany(
            "INSERT INTO documents(id, text) VALUES (?, ?)",
            [(1, "alpha"), (2, "beta")],
        )


@pytest.mark.asyncio
async def test_embedding_fingerprint_detects_same_dimension_model_change() -> None:
    first = _FakeEmbeddingProvider("model_a")
    second = _FakeEmbeddingProvider("model_b")
    assert EmbeddingFingerprintService.digest(first) != EmbeddingFingerprintService.digest(
        second
    )
    assert "must-not-be-written" not in json.dumps(
        EmbeddingFingerprintService.payload(first), ensure_ascii=True
    )


@pytest.mark.asyncio
async def test_shadow_rebuild_preserves_old_index_on_provider_failure(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)
    db_path = tmp_path / "documents.db"
    index_path = tmp_path / "memory.index"
    state_path = tmp_path / "embedding_index_state.json"
    _make_documents_db(db_path)
    index_path.write_bytes(b"old-index")
    bootstrap = _FakeIndexBootstrap(_FakeIndex(2, [1, 2]))
    service = EmbeddingIndexBootstrapService(
        state_path,
        faiss_bootstrap=bootstrap,
        batch_size=1,
    )
    original = _FakeEmbeddingProvider("model_a")
    await service.prepare(
        [EmbeddingIndexSpec("document", db_path, index_path)], original
    )
    original_state = state_path.read_text(encoding="utf-8")
    original_bytes = index_path.read_bytes()

    failing = _FakeEmbeddingProvider("model_b", fail=True)
    with pytest.raises(InitializationError, match="影子索引重建失败"):
        await service.prepare(
            [EmbeddingIndexSpec("document", db_path, index_path)], failing
        )

    assert index_path.read_bytes() == original_bytes
    assert state_path.read_text(encoding="utf-8") == original_state


@pytest.mark.asyncio
async def test_shadow_rebuild_atomically_switches_complete_index(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)
    db_path = tmp_path / "documents.db"
    index_path = tmp_path / "memory.index"
    state_path = tmp_path / "embedding_index_state.json"
    _make_documents_db(db_path)
    index_path.write_bytes(b"old-index")
    bootstrap = _FakeIndexBootstrap(_FakeIndex(2, [1, 2]))
    service = EmbeddingIndexBootstrapService(
        state_path,
        faiss_bootstrap=bootstrap,
        batch_size=1,
    )
    spec = EmbeddingIndexSpec("document", db_path, index_path)
    await service.prepare([spec], _FakeEmbeddingProvider("model_a"))

    replacement = _FakeEmbeddingProvider("model_b")
    result = await service.prepare([spec], replacement)

    assert result[0].action == "rebuilt"
    assert result[0].reason == "embedding_provider_changed"
    assert replacement.calls == 2
    assert index_path.read_bytes() == b"new-index"
    assert bootstrap.index.id_map.tolist() == [1, 2]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["indexes"]["document"]["vector_schema"] == "document-v1"


def _create_legacy_database(path: Path) -> None:
    with sqlite3.connect(path) as database:
        database.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL)"
        )
        database.execute("INSERT INTO documents(content) VALUES ('legacy data')")
        database.commit()


@pytest.mark.asyncio
async def test_unsupported_legacy_database_is_backed_up_without_source_mutation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "livingmemory.db"
    _create_legacy_database(db_path)

    result = await DBMigration(str(db_path)).migrate()

    assert result["success"] is False
    assert result["from_version"] == 1
    assert result["to_version"] == DBMigration.CURRENT_VERSION
    backup_path = Path(result["backup_path"])
    assert backup_path.exists()
    with sqlite3.connect(db_path) as database:
        row = database.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='db_version'"
        ).fetchone()
        assert row is None
        assert database.execute("SELECT content FROM documents").fetchone()[0] == (
            "legacy data"
        )
    with sqlite3.connect(backup_path) as backup:
        assert backup.execute("SELECT content FROM documents").fetchone()[0] == (
            "legacy data"
        )


@pytest.mark.asyncio
async def test_initializer_blocks_when_migration_returns_failure() -> None:
    initializer = PluginInitializer.__new__(PluginInitializer)
    initializer.db_migration = SimpleNamespace(
        needs_migration=AsyncMock(return_value=True),
        migrate=AsyncMock(
            return_value={
                "success": False,
                "message": "旧数据库版本不受支持",
                "backup_path": "backup_demo.db",
            }
        ),
    )

    with pytest.raises(InitializationError, match="旧数据库版本不受支持"):
        await initializer._check_and_migrate_database()

    initializer.db_migration.migrate.assert_awaited_once_with(progress_callback=None)


@pytest.mark.asyncio
async def test_initialize_runs_database_preflight_before_provider_wait(
    tmp_path: Path,
) -> None:
    initializer = PluginInitializer(
        context=SimpleNamespace(),
        config_manager=SimpleNamespace(),
        data_dir=str(tmp_path),
    )
    initializer._preflight_database_compatibility = AsyncMock(
        side_effect=InitializationError("旧库不受支持")
    )
    initializer._wait_for_providers_non_blocking = AsyncMock(return_value=True)

    result = await initializer.initialize()

    assert result is False
    assert initializer.is_failed is True
    assert "旧库不受支持" in str(initializer.error_message)
    initializer._wait_for_providers_non_blocking.assert_not_awaited()


@pytest.mark.asyncio
async def test_shadow_rebuild_downgrades_batch_size_on_provider_rejection(
    monkeypatch, tmp_path
) -> None:
    """Provider 拒绝超限批大小时自动降批并完成重建（不再整体失败）。"""
    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)
    db_path = tmp_path / "documents.db"
    index_path = tmp_path / "memory.index"
    state_path = tmp_path / "embedding_index_state.json"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT)")
        db.executemany(
            "INSERT INTO documents(id, text) VALUES (?, ?)",
            [(index, f"text-{index}") for index in range(1, 13)],
        )
    bootstrap = _FakeIndexBootstrap(_FakeIndex(2, list(range(1, 13))))
    service = EmbeddingIndexBootstrapService(
        state_path,
        faiss_bootstrap=bootstrap,
        batch_size=6,
    )
    provider = _FakeEmbeddingProvider("model_a", max_batch=2)

    result = await service.prepare(
        [EmbeddingIndexSpec("document", db_path, index_path)], provider
    )

    assert result[0].action == "rebuilt"
    # 6 → 3 → 1：降批后每个实际成功的批次都不超过 Provider 上限
    assert provider.batches
    assert all(size <= 2 for size in provider.batches)
    assert sum(provider.batches) == 12
    assert bootstrap.index.id_map.tolist() == list(range(1, 13))


