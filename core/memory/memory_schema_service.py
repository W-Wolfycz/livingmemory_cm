"""MemoryEngine 主 SQLite schema 的创建与兼容收敛。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ...log import logger, tag
from .metadata_expr import CREATED_EXPR, STATUS_EXPR, TYPE_EXPR


@dataclass(slots=True)
class MemorySchemaContext:
    """Schema 初始化所需的动态依赖。"""

    db_connection: Any
    create_write_ops_table: Callable[[], Awaitable[None]]


class MemorySchemaService:
    """创建主表、JSON 索引、版本表并清理旧 FTS trigger。"""

    async def create_tables(self, context: MemorySchemaContext) -> None:
        connection = context.db_connection
        if connection is None:
            return

        await self.drop_legacy_documents_fts_triggers(context)
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                text TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                updated_at TEXT
            )
        """)

        cursor = await connection.execute("PRAGMA table_info(documents)")
        column_rows = await cursor.fetchall()
        existing_columns = {row[1] for row in column_rows}
        missing_columns: list[str] = []
        for column_name, column_type in (
            ("doc_id", "TEXT"),
            ("created_at", "TEXT"),
            ("updated_at", "TEXT"),
        ):
            if column_name in existing_columns:
                continue
            await connection.execute(
                f"ALTER TABLE documents ADD COLUMN {column_name} {column_type}"
            )
            missing_columns.append(column_name)

        if missing_columns:
            logger.warning(
                f"{tag('schema')} 检测到旧版 documents 表结构，已补齐字段: "
                f"{', '.join(missing_columns)}"
            )

        await connection.execute("""
            UPDATE documents
            SET doc_id = 'legacy-' || id
            WHERE doc_id IS NULL OR TRIM(doc_id) = ''
        """)
        await connection.execute("""
            UPDATE documents
            SET created_at = datetime('now')
            WHERE created_at IS NULL OR TRIM(CAST(created_at AS TEXT)) = ''
        """)
        await connection.execute("""
            UPDATE documents
            SET updated_at = COALESCE(created_at, datetime('now'))
            WHERE updated_at IS NULL OR TRIM(CAST(updated_at AS TEXT)) = ''
        """)

        for index_sql in (
            """
            CREATE INDEX IF NOT EXISTS idx_doc_metadata
            ON documents(json_extract(metadata, '$.session_id'))
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_doc_persona_metadata
            ON documents(json_extract(metadata, '$.persona_id'))
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_doc_importance_metadata
            ON documents(json_extract(metadata, '$.importance'))
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_doc_last_access_metadata
            ON documents(json_extract(metadata, '$.last_access_time'))
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_doc_idempotency_key
            ON documents(json_extract(metadata, '$.idempotency_key'))
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_documents_doc_id
            ON documents(doc_id)
            """,
            # 表达式索引：表达式来自 metadata_expr 单一来源，与召回/列表查询里的
            # 表达式同形（含 json_valid 守卫与 COALESCE/CAST 包装）才会被命中。
            # 复合形式让同一个索引同时服务 "status 过滤 + create_time 倒序截断" 的
            # 列表/计数查询与近期记忆槽位；不用 INDEXED BY 钉计划——索引万一缺失时
            # 钉计划会让用户查询直接报错，宁可退化扫描并由 EXPLAIN 测试守住计划。
            f"""
            CREATE INDEX IF NOT EXISTS idx_doc_status_created
            ON documents(
                {STATUS_EXPR},
                {CREATED_EXPR} DESC,
                id DESC
            )
            """,
            # 按记忆类型过滤的列表查询（WebUI「类型」筛选）
            f"""
            CREATE INDEX IF NOT EXISTS idx_doc_type_created
            ON documents(
                {TYPE_EXPR},
                {CREATED_EXPR} DESC,
                id DESC
            )
            """,
            # 不带筛选的列表/导出：只有独立的 create_time 索引才能直接提供排序，
            # 否则 SQLite 要对全表建 TEMP B-TREE 再截断（数据量越大越慢）。
            f"""
            CREATE INDEX IF NOT EXISTS idx_doc_created
            ON documents(
                {CREATED_EXPR} DESC,
                id DESC
            )
            """,
        ):
            await connection.execute(index_sql)

        await self._assert_expression_indexes(connection)

        await context.create_write_ops_table()
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS db_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL,
                description TEXT,
                migrated_at TEXT NOT NULL,
                migration_duration_seconds REAL
            )
        """)
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS migration_status (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        await connection.commit()

        cursor = await connection.execute("SELECT COUNT(*) FROM db_version")
        version_result = await cursor.fetchone()
        version_count = version_result[0] if version_result else 0
        if version_count != 0:
            return

        from ...storage.db_migration import DBMigration

        await connection.execute(
            """
            INSERT INTO db_version(
                version, description, migrated_at, migration_duration_seconds
            ) VALUES (?, ?, ?, ?)
            """,
            (
                DBMigration.CURRENT_VERSION,
                "初始版本 - 当前架构",
                datetime.now(timezone.utc).isoformat(),
                0.0,
            ),
        )
        await connection.commit()
        logger.info(
            f"{tag('schema')} 已初始化数据库版本信息: "
            f"v{DBMigration.CURRENT_VERSION}"
        )

    @staticmethod
    async def _assert_expression_indexes(connection) -> None:
        """表达式索引缺失时大声报警。

        ``CREATE INDEX IF NOT EXISTS`` 在索引被手工删除、或迁移中断时会静默通过，
        此后热路径退化为全表扫描且没有任何症状——这里补一条可见的 ERROR。
        不抛异常：索引缺失仍能正常读写（只是慢），不该因此拦住插件启动。
        """
        expected = {
            "idx_doc_status_created",
            "idx_doc_type_created",
            "idx_doc_created",
        }
        cursor = await connection.execute("PRAGMA index_list(documents)")
        present = {row[1] for row in await cursor.fetchall()}
        missing = sorted(expected - present)
        if missing:
            logger.error(
                f"{tag('schema')} 表达式索引缺失: {', '.join(missing)}"
                "；列表/检索查询会退化为全表扫描，请检查数据库迁移是否完成"
            )

    async def drop_legacy_documents_fts_triggers(
        self,
        context: MemorySchemaContext,
    ) -> int:
        connection = context.db_connection
        if connection is None:
            return 0

        cursor = await connection.execute("""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND tbl_name='documents'
              AND sql LIKE '%documents_fts%'
        """)
        rows = await cursor.fetchall()
        for row in rows:
            trigger_name = str(row[0])
            escaped_name = trigger_name.replace('"', '""')
            await connection.execute(
                f'DROP TRIGGER IF EXISTS "{escaped_name}"'
            )
            logger.warning(
                f"{tag('schema')} 已清理旧 LivingMemory FTS trigger"
            )
        return len(rows)


__all__ = ["MemorySchemaContext", "MemorySchemaService"]
