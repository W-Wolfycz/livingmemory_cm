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


def test_faiss_bootstrap_runtime_failure_is_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(
        faiss_bootstrap_module.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=-4, stderr="", stdout="")),
    )

    with pytest.raises(InitializationError, match="信号 4"):
        FaissBootstrapService().check_runtime()


def test_faiss_binding_mismatch_is_not_reported_as_provider_error(monkeypatch) -> None:
    result = SimpleNamespace(
        returncode=1,
        stderr="NameError: name 'SuperKMeans' is not defined",
        stdout="",
    )
    run = Mock(return_value=result)
    monkeypatch.setattr(faiss_bootstrap_module.subprocess, "run", run)
    monkeypatch.setattr(
        faiss_bootstrap_module.importlib_metadata,
        "version",
        Mock(return_value="1.14.2"),
    )

    with pytest.raises(InitializationError, match="封装与二进制扩展不匹配"):
        FaissBootstrapService().check_runtime()

    run.assert_called_once()


def test_faiss_generic_fallback_only_for_compatible_failure(monkeypatch) -> None:
    failed = SimpleNamespace(
        returncode=1,
        stderr="Illegal instruction while loading optimized extension",
        stdout="",
    )
    recovered = SimpleNamespace(returncode=0, stderr="", stdout="")
    run = Mock(side_effect=[failed, recovered])
    monkeypatch.setattr(faiss_bootstrap_module.subprocess, "run", run)
    monkeypatch.delenv("FAISS_OPT_LEVEL", raising=False)

    FaissBootstrapService().check_runtime()

    assert faiss_bootstrap_module.os.environ["FAISS_OPT_LEVEL"] == "generic"
    assert run.call_args_list[1].kwargs["env"]["FAISS_OPT_LEVEL"] == "generic"


@pytest.mark.asyncio
async def test_faiss_bootstrap_rejects_empty_index_without_deleting(tmp_path) -> None:
    index_path = tmp_path / "empty.index"
    index_path.write_bytes(b"")
    provider = SimpleNamespace(get_dim=Mock(return_value=128))

    with pytest.raises(InitializationError, match="为空"):
        await FaissBootstrapService().check_and_fix_dimension_mismatch(
            str(index_path), provider
        )

    assert index_path.exists()
    provider.get_dim.assert_not_called()


@pytest.mark.asyncio
async def test_faiss_bootstrap_keeps_matching_dimension(monkeypatch, tmp_path) -> None:
    # 该分支会执行 faiss_bootstrap 里的 ``import faiss``；本地不装 faiss，
    # 用命名空间占位即可通过（实际读取已被 mock 掉）。
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace())
    index_path = tmp_path / "matching.index"
    index_path.write_bytes(b"valid")
    service = FaissBootstrapService()
    monkeypatch.setattr(
        service,
        "faiss_read_index_safe",
        Mock(return_value=SimpleNamespace(d=256)),
    )
    provider = SimpleNamespace(get_dim=Mock(return_value=256))

    await service.check_and_fix_dimension_mismatch(str(index_path), provider)

    assert index_path.exists()
    provider.get_dim.assert_called_once_with()


@pytest.mark.asyncio
async def test_faiss_bootstrap_rejects_mismatched_dimension_without_deleting(
    monkeypatch,
    tmp_path,
) -> None:
    # 该分支同样会执行 ``import faiss``，用命名空间占位避免真实依赖。
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace())
    index_path = tmp_path / "mismatched.index"
    index_path.write_bytes(b"valid")
    service = FaissBootstrapService()
    monkeypatch.setattr(
        service,
        "faiss_read_index_safe",
        Mock(return_value=SimpleNamespace(d=128)),
    )
    provider = SimpleNamespace(get_dim=Mock(return_value=256))

    with pytest.raises(InitializationError, match="维度不匹配"):
        await service.check_and_fix_dimension_mismatch(str(index_path), provider)

    assert index_path.exists()


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
    def __init__(self, model: str, *, fail: bool = False) -> None:
        self.model = model
        self.fail = fail
        self.calls = 0
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


def test_living_memory_exception_fields() -> None:
    error = LivingMemoryException("boom", "E_TEST")

    assert str(error) == "boom"
    assert error.message == "boom"
    assert error.error_code == "E_TEST"


def test_specialized_exception_codes_and_inheritance() -> None:
    cases = [
        (InitializationError, InitializationError("x"), "INIT_ERROR"),
        (ProviderNotReadyError, ProviderNotReadyError(), "PROVIDER_NOT_READY"),
        (DatabaseError, DatabaseError("x"), "DATABASE_ERROR"),
        (RetrievalError, RetrievalError("x"), "RETRIEVAL_ERROR"),
        (MemoryProcessingError, MemoryProcessingError("x"), "MEMORY_PROCESSING_ERROR"),
        (ConfigurationError, ConfigurationError("x"), "CONFIG_ERROR"),
        (ValidationError, ValidationError("x"), "VALIDATION_ERROR"),
    ]

    for exception_type, error, expected_code in cases:
        assert issubclass(exception_type, LivingMemoryException)
        assert error.error_code == expected_code


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
async def test_initializer_blocks_without_migration_manager() -> None:
    initializer = PluginInitializer.__new__(PluginInitializer)
    initializer.db_migration = None

    with pytest.raises(InitializationError, match="迁移管理器未初始化"):
        await initializer._check_and_migrate_database()


@pytest.mark.asyncio
async def test_initializer_skips_migration_for_current_database() -> None:
    initializer = PluginInitializer.__new__(PluginInitializer)
    initializer.db_migration = SimpleNamespace(
        needs_migration=AsyncMock(return_value=False),
        migrate=AsyncMock(),
    )

    await initializer._check_and_migrate_database()

    initializer.db_migration.migrate.assert_not_awaited()


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
