"""
数据库迁移管理器 - 版本管理与备份

历史迁移函数（v1→v8）已移除：本仓库数据库始终处于 v8。
未来如需升级到 v9，请在此追加 _migrate_v8_to_v9 并在 migrate() 中注册。
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from ..log import logger, tag


class DBMigration:
    """数据库迁移管理器"""

    # 当前数据库版本
    CURRENT_VERSION = 8

    # 版本历史记录
    VERSION_HISTORY = {
        1: "初始版本 - 基础记忆存储",
        2: "FTS5索引预处理 - 添加分词和停用词支持",
        3: "会话ID迁移 - 标记需要session_id格式升级",
        4: "Schema v2 - 双通道总结字段 + source_window 溯源支持",
        5: "Graph memory - graph tables and dual-route retrieval metadata",
        6: "插件 FTS 表统一 livingmemory 前缀，旧 documents_fts 安全重命名备份",
        7: "Storage indexes and FTS optimization for graph and atom data",
        8: "Write-operation log and access-aware metadata indexes",
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.migration_lock = asyncio.Lock()

    async def get_db_version(self) -> int:
        """
        获取当前数据库版本

        Returns:
            int: 数据库版本号，如果不存在版本表则返回1（旧版本）
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 检查版本表是否存在
                cursor = await db.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='db_version'
                """)
                table_exists = await cursor.fetchone()

                if not table_exists or len(table_exists) == 0:
                    # 没有版本表，检查是否有documents表（判断是否为旧数据库）
                    cursor = await db.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='documents'
                    """)
                    has_documents = await cursor.fetchone()

                    if has_documents:
                        # 有documents表但没有版本表，检查是否有数据
                        cursor = await db.execute("SELECT COUNT(*) FROM documents")
                        doc_count_row = await cursor.fetchone()
                        doc_count = doc_count_row[0] if doc_count_row else 0

                        if doc_count > 0:
                            # 有数据但无版本表，判定为v1旧数据库
                            logger.info(
                                f"{tag('migrate')} 检测到旧版本数据库（无版本表，有{doc_count}条数据），当前版本: 1"
                            )
                            return 1
                        else:
                            # 空数据库，视为最新版本
                            logger.info(
                                f"{tag('migrate')} 检测到空数据库（已初始化但无数据），视为最新版本"
                            )
                            return self.CURRENT_VERSION
                    else:
                        # 全新数据库，没有任何表，视为最新版本
                        logger.info(f"{tag('migrate')} 检测到全新数据库，视为最新版本")
                        return self.CURRENT_VERSION

                # 读取版本号
                cursor = await db.execute(
                    "SELECT version FROM db_version ORDER BY id DESC LIMIT 1"
                )
                row = await cursor.fetchone()

                if row and len(row) > 0:
                    version = row[0]
                    logger.info(f"{tag('migrate')} 当前数据库版本: {version}")
                    return version
                else:
                    return 1

        except Exception as e:
            logger.error(f"{tag('migrate')} 获取数据库版本失败: {e}", exc_info=True)
            return 1

    async def initialize_version_table(self):
        """初始化版本管理表"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS db_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version INTEGER NOT NULL,
                        description TEXT,
                        migrated_at TEXT NOT NULL,
                        migration_duration_seconds REAL
                    )
                """)
                await db.commit()
                logger.info(f"{tag('migrate')} 数据库版本管理表初始化完成")
        except Exception as e:
            logger.error(f"{tag('migrate')} 初始化版本表失败: {e}", exc_info=True)
            raise

    async def set_db_version(
        self, version: int, description: str = "", duration: float = 0.0
    ):
        """
        设置数据库版本

        Args:
            version: 版本号
            description: 版本描述
            duration: 迁移耗时（秒）
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO db_version (version, description, migrated_at, migration_duration_seconds)
                    VALUES (?, ?, ?, ?)
                """,
                    (version, description, datetime.now(timezone.utc).isoformat(), duration),
                )
                await db.commit()
                logger.info(f"{tag('migrate')} 数据库版本已更新至: {version}")
        except Exception as e:
            logger.error(f"{tag('migrate')} 设置数据库版本失败: {e}", exc_info=True)
            raise

    async def needs_migration(self) -> bool:
        """
        检查是否需要迁移

        Returns:
            bool: True表示需要迁移
        """
        current_version = await self.get_db_version()
        needs_migration = current_version < self.CURRENT_VERSION

        if needs_migration:
            logger.warning(
                f"{tag('migrate')} 数据库需要迁移: v{current_version} -> v{self.CURRENT_VERSION}"
            )
        else:
            logger.info(f"{tag('migrate')} 数据库版本最新: v{current_version}")

        return needs_migration

    async def migrate(
        self,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        执行数据库迁移

        当前版本（v8）以下的所有迁移函数已移除。若数据库版本低于 CURRENT_VERSION，
        会创建备份并提示需要手动处理；新版本（v9+）发布时请在此注册新迁移步骤。

        Args:
            progress_callback: 进度回调函数 (message, current, total)

        Returns:
            Dict: 迁移结果
        """
        async with self.migration_lock:
            start_time = datetime.now()

            try:
                # 先读取版本，旧库在备份完成前不写入任何新表或版本记录。
                current_version = await self.get_db_version()

                if current_version >= self.CURRENT_VERSION:
                    return {
                        "success": True,
                        "message": "数据库已是最新版本，无需迁移",
                        "from_version": current_version,
                        "to_version": self.CURRENT_VERSION,
                        "duration": 0,
                    }

                logger.warning(
                    f"{tag('migrate')} 检测到旧版本数据库 v{current_version}，本仓库已移除历史迁移代码"
                )
                logger.warning(
                    f"{tag('migrate')} 已自动创建备份，如需升级到 v{self.CURRENT_VERSION} 请恢复历史迁移代码或手动处理"
                )

                # 迁移前自动备份，确保数据安全
                backup_path = await self.create_backup()
                if backup_path:
                    logger.info(f"{tag('migrate')} 迁移前备份已创建: {backup_path}")
                else:
                    logger.warning(
                        f"{tag('migrate')} 迁移前备份失败，请确认磁盘空间与文件权限。"
                    )

                duration = (datetime.now() - start_time).total_seconds()

                return {
                    "success": False,
                    "message": (
                        f"数据库版本 v{current_version} 低于当前版本 v{self.CURRENT_VERSION}，"
                        "但历史迁移代码已移除。"
                        + (
                            "已创建备份，需手动处理。"
                            if backup_path
                            else "备份创建失败，已阻止启动；请先人工备份并处理。"
                        )
                    ),
                    "from_version": current_version,
                    "to_version": self.CURRENT_VERSION,
                    "duration": duration,
                    "backup_path": backup_path,
                }

            except Exception as e:
                logger.error(f"{tag('migrate')} 数据库迁移失败: {e}", exc_info=True)
                return {
                    "success": False,
                    "message": f"数据库迁移失败: {str(e)}",
                    "error": str(e),
                }

    async def get_migration_info(self) -> dict[str, Any]:
        """
        获取迁移信息

        Returns:
            Dict: 迁移信息
        """
        try:
            current_version = await self.get_db_version()
            needs_migration = await self.needs_migration()

            # 获取迁移历史
            migration_history = []
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute("""
                        SELECT version, description, migrated_at, migration_duration_seconds
                        FROM db_version
                        ORDER BY id DESC
                        LIMIT 10
                    """)
                    rows = await cursor.fetchall()

                    for row in rows:
                        migration_history.append(
                            {
                                "version": row[0],
                                "description": row[1],
                                "migrated_at": row[2],
                                "duration": row[3],
                            }
                        )
            except Exception as e:
                logger.error(f"{tag('migrate')} 获取迁移历史失败: {e}", exc_info=True)

            return {
                "current_version": current_version,
                "latest_version": self.CURRENT_VERSION,
                "needs_migration": needs_migration,
                "version_history": self.VERSION_HISTORY,
                "migration_history": migration_history,
                "db_path": self.db_path,
            }

        except Exception as e:
            logger.error(f"{tag('migrate')} 获取迁移信息失败: {e}", exc_info=True)
            return {"error": str(e)}

    async def create_backup(self) -> str | None:
        """
        创建数据库备份

        Returns:
            Optional[str]: 备份文件路径，失败返回None
        """
        try:
            db_path = Path(self.db_path)
            backup_dir = db_path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = (
                backup_dir / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
            )

            logger.info(f"{tag('migrate')} 正在创建数据库备份: {backup_path}")

            # 使用SQLite的备份API
            async with aiosqlite.connect(self.db_path) as source:
                async with aiosqlite.connect(str(backup_path)) as dest:
                    await source.backup(dest)

            logger.info(f"{tag('migrate')} 数据库备份成功: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"{tag('migrate')} 数据库备份失败: {e}", exc_info=True)
            return None
