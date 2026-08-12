"""Version-triggered data backup manager.

Automatically backs up all plugin data files when the plugin version changes,
storing each backup under a version-tagged directory for easy recovery.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from ...log import logger, tag
from ...version import PLUGIN_VERSION

_VERSION_FILE = ".plugin_version"
_BACKUP_INFO_FILE = "backup_info.json"

# Files/patterns to include in a full backup (relative to data_dir).
_BACKUP_PATTERNS: list[str] = [
    "livingmemory.db",
    "livingmemory.index",
    "livingmemory_graph_documents.db",
    "livingmemory_graph.index",
    "conversations.db",
    "decay_state.json",
    "*.db-wal",
    "*.db-shm",
]


class BackupError(RuntimeError):
    """版本备份未完整完成。"""


class BackupManager:
    """Detect version changes and create full data backups."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self.version_file = self.data_dir / _VERSION_FILE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stored_version(self) -> str | None:
        """Return the last-known plugin version, or None on first run."""
        if not self.version_file.exists():
            return None
        try:
            return self.version_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def write_current_version(self) -> None:
        """Persist the current plugin version."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.version_file.with_name(f".{self.version_file.name}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(PLUGIN_VERSION)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.version_file)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def needs_backup(self) -> bool:
        """Return True when the plugin version has changed (or is fresh)."""
        stored = self.get_stored_version()
        if stored is None:
            return True  # first install — backup for safety
        return stored != PLUGIN_VERSION

    def backup_if_needed(self) -> str | None:
        """Create a full backup when the version changed. Returns backup dir path or None."""
        if not self.needs_backup():
            return None

        stored = self.get_stored_version()
        old_label = stored or "unknown"
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", old_label).strip(".") or "unknown"
        backups_root = self.data_dir / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        backup_dir = backups_root / f"v{safe_label}"
        if backup_dir.exists():
            # 同一版本的残留目录可能来自进程在原子发布前崩溃；不覆盖已有备份，
            # 使用时间后缀保留可恢复性。
            backup_dir = backups_root / f"v{safe_label}-{int(time.time())}"
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".v{safe_label}.tmp-", dir=backups_root)
        )

        logger.info(
            f"{tag('backup')} 检测到版本变更 ({old_label} → {PLUGIN_VERSION})，"
            f"正在备份数据到 {backup_dir} ..."
        )

        try:
            copied_count = 0
            seen_paths: set[Path] = set()
            for pattern in _BACKUP_PATTERNS:
                for file_path in self.data_dir.glob(pattern):
                    if not file_path.is_file() or file_path in seen_paths:
                        continue
                    seen_paths.add(file_path)
                    dest = staging_dir / file_path.name
                    try:
                        shutil.copy2(file_path, dest)
                        copied_count += 1
                    except OSError as exc:
                        logger.error(
                            f"{tag('backup')} 备份文件失败 {file_path.name}: {exc}"
                        )
                        raise BackupError(
                            f"备份文件失败: {file_path.name}: {exc}"
                        ) from exc

            # 元数据先写入临时目录；只有整个目录准备完成后才对外可见。
            info = {
                "plugin_version": PLUGIN_VERSION,
                "previous_version": old_label,
                "backup_timestamp": datetime.now(timezone.utc).isoformat(),
                "backup_unix_time": time.time(),
                "files_copied": copied_count,
                "complete": True,
            }
            info_path = staging_dir / _BACKUP_INFO_FILE
            with open(info_path, "w", encoding="utf-8") as handle:
                json.dump(info, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(staging_dir, backup_dir)

            # 只有备份目录已经原子发布后才更新版本号；失败会让下次启动重试。
            self.write_current_version()
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        logger.info(f"{tag('backup')} 备份完成: {copied_count} 个文件 → {backup_dir}")
        return str(backup_dir)

    async def backup_if_needed_async(self) -> str | None:
        """异步版本：通过 asyncio.to_thread 将同步文件 I/O 卸载到线程池。"""
        return await asyncio.to_thread(self.backup_if_needed)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def list_backups(data_dir: str) -> list[dict]:
        """Enumerate existing backups with their metadata."""
        backups_path = Path(data_dir) / "backups"
        if not backups_path.exists():
            return []

        result: list[dict] = []
        for backup_dir in sorted(backups_path.iterdir(), reverse=True):
            if not backup_dir.is_dir():
                continue
            if backup_dir.name.startswith("."):
                # 隐藏临时目录表示上次备份在原子发布前中断，不视为可恢复备份。
                continue
            info_path = backup_dir / _BACKUP_INFO_FILE
            info: dict = {}
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            info.setdefault("directory", str(backup_dir))
            info.setdefault("name", backup_dir.name)
            files = [p.name for p in backup_dir.iterdir() if p.is_file()]
            info.setdefault("files", files)
            info.setdefault("file_count", len(files))
            result.append(info)

        return result


__all__ = ["BackupError", "BackupManager", "PLUGIN_VERSION"]
