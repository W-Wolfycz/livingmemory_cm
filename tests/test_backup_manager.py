"""Tests for version-triggered backup manager."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from livingmemory_cm.core.managers.backup_manager import (
    BackupError,
    PLUGIN_VERSION,
    BackupManager,
    _BACKUP_INFO_FILE,
)


def test_backup_if_needed_creates_backup_dir(tmp_path: Path) -> None:
    # Create mock data files
    (tmp_path / "livingmemory.db").write_text("mock db content")
    (tmp_path / "livingmemory.index").write_text("mock index")
    (tmp_path / "conversations.db").write_text("mock conversations")
    (tmp_path / "decay_state.json").write_text('{"last_run": "2025-01-01"}')

    mgr = BackupManager(str(tmp_path))
    # Simulate previous version
    mgr.version_file.write_text("2.0.0", encoding="utf-8")

    backup_dir = mgr.backup_if_needed()
    assert backup_dir is not None
    assert os.path.isdir(backup_dir)
    assert "v2.0.0" in backup_dir

    # Verify files were copied
    backup_path = Path(backup_dir)
    assert (backup_path / "livingmemory.db").exists()
    assert (backup_path / "conversations.db").exists()
    assert (backup_path / "decay_state.json").exists()

    # Verify backup info was written
    info_path = backup_path / _BACKUP_INFO_FILE
    assert info_path.exists()
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["plugin_version"] == PLUGIN_VERSION
    assert info["previous_version"] == "2.0.0"
    assert info["files_copied"] >= 3

    # Verify version file was updated
    assert mgr.get_stored_version() == PLUGIN_VERSION

def test_backup_covers_all_data_file_patterns(tmp_path: Path) -> None:
    """守的静默失效：`_BACKUP_PATTERNS` 漏一类文件时备份照样"成功"，
    直到真去恢复才发现图数据 / WAL / SHM 没备份。
    """
    for name in (
        "livingmemory_graph_documents.db",
        "livingmemory_graph.index",
        "livingmemory.db-wal",
        "livingmemory.db-shm",
    ):
        (tmp_path / name).write_text("payload")

    mgr = BackupManager(str(tmp_path))
    mgr.version_file.write_text("2.0.0", encoding="utf-8")
    backup_dir = mgr.backup_if_needed()

    backup_path = Path(backup_dir)
    missing = [
        name
        for name in (
            "livingmemory_graph_documents.db",
            "livingmemory_graph.index",
            "livingmemory.db-wal",
            "livingmemory.db-shm",
        )
        if not (backup_path / name).exists()
    ]
    assert missing == [], f"备份遗漏文件: {missing}"
    info = json.loads((backup_path / _BACKUP_INFO_FILE).read_text(encoding="utf-8"))
    assert info["files_copied"] >= 4

def test_backup_failure_does_not_mark_version_complete(monkeypatch, tmp_path: Path) -> None:
    """真实复制失败时保留旧版本标记，下一次启动必须可以重试。"""
    (tmp_path / "livingmemory.db").write_text("ok")
    mgr = BackupManager(str(tmp_path))
    mgr.version_file.write_text("1.0.0", encoding="utf-8")

    def _fail_copy(*_args, **_kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(
        "livingmemory_cm.core.managers.backup_manager.shutil.copy2",
        _fail_copy,
    )

    with pytest.raises(BackupError, match="备份文件失败"):
        mgr.backup_if_needed()

    assert mgr.get_stored_version() == "1.0.0"
    assert mgr.needs_backup() is True
    published = [
        path
        for path in (tmp_path / "backups").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert published == []

def test_list_backups_corrupt_info_json(tmp_path: Path) -> None:
    """损坏的 backup_info.json 不应导致 list_backups 崩溃，应回退到默认值。"""
    root = tmp_path / "backups" / "v1.0.0"
    root.mkdir(parents=True)
    (root / _BACKUP_INFO_FILE).write_text("{not valid json", encoding="utf-8")
    (root / "livingmemory.db").write_text("data")

    result = BackupManager.list_backups(str(tmp_path))
    assert len(result) == 1
    # 损坏 info 时回退到默认字段
    assert result[0]["name"] == "v1.0.0"
    assert "livingmemory.db" in result[0]["files"]

def test_pluigin_version_constant_matches_metadata() -> None:
    """Ensure PLUGIN_VERSION in backup_manager matches metadata.yaml."""
    import yaml

    metadata_path = Path(__file__).parent.parent / "metadata.yaml"
    if not metadata_path.exists():
        return  # skip if metadata.yaml not found (CI, etc.)

    with open(metadata_path, encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    assert PLUGIN_VERSION == metadata["version"], (
        f"PLUGIN_VERSION ({PLUGIN_VERSION}) must match metadata.yaml "
        f"({metadata['version']}). Update backup_manager.py."
    )
