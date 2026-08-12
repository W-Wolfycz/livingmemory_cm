"""长期记忆的衰减、清理、访问与兼容迁移。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ...log import log_ref, logger, tag
from ..utils.number_utils import clamp_float, safe_float


@dataclass(slots=True)
class MemoryLifecycleContext:
    """生命周期维护调用所需的动态引擎依赖。"""

    db_connection: Any
    faiss_db: Any
    graph_memory_manager: Any
    document_repository: Any
    config: dict[str, Any]
    batch_delete_memories: Callable[[list[int]], Awaitable[int]]
    invalidate_search_cache: Callable[[], None]


class MemoryLifecycleService:
    """执行图索引重建、访问计数、衰减、清理和旧会话迁移。"""

    async def rebuild_graph_index(
        self,
        context: MemoryLifecycleContext,
    ) -> dict[str, int]:
        if context.graph_memory_manager is None:
            return {"rebuilt": 0, "skipped": 0}

        async def memory_batches():
            yielded_from_sqlite = False
            if context.db_connection is not None:
                last_document_id = -1
                while True:
                    cursor = await context.db_connection.execute(
                        """
                        SELECT id, text, metadata
                        FROM documents
                        WHERE id > ?
                          AND COALESCE(
                              json_extract(metadata, '$.status'),
                              'active'
                          ) = 'active'
                        ORDER BY id ASC
                        LIMIT ?
                        """,
                        (last_document_id, 200),
                    )
                    rows = await cursor.fetchall()
                    if not rows:
                        break
                    yielded_from_sqlite = True
                    yield [
                        (
                            int(row[0]),
                            str(row[1] or ""),
                            self.safe_json_dict(row[2]),
                        )
                        for row in rows
                    ]
                    last_document_id = int(rows[-1][0])

            if yielded_from_sqlite:
                return

            document_storage = context.faiss_db.document_storage
            total_count = await document_storage.count_documents(
                metadata_filters={}
            )
            offset = 0
            while offset < total_count:
                documents = await document_storage.get_documents(
                    metadata_filters={},
                    limit=200,
                    offset=offset,
                )
                if not documents:
                    break
                yield [
                    (
                        int(document["id"]),
                        str(document.get("text") or ""),
                        self.safe_json_dict(document.get("metadata")),
                    )
                    for document in documents
                    if str(
                        self.safe_json_dict(document.get("metadata")).get(
                            "status"
                        )
                        or "active"
                    )
                    == "active"
                ]
                offset += len(documents)

        rebuild_batches = getattr(
            context.graph_memory_manager,
            "rebuild_memory_batches",
            None,
        )
        if callable(rebuild_batches):
            result = await rebuild_batches(memory_batches())
        else:
            rebuilt = 0
            skipped = 0
            async for batch in memory_batches():
                for memory_id, content, metadata in batch:
                    if not content.strip():
                        skipped += 1
                        continue
                    await context.graph_memory_manager.index_memory(
                        memory_id,
                        content,
                        metadata,
                    )
                    rebuilt += 1
            result = {"rebuilt": rebuilt, "skipped": skipped}

        context.invalidate_search_cache()
        return result

    async def apply_daily_decay(
        self,
        context: MemoryLifecycleContext,
        decay_rate: float,
        days: int = 1,
    ) -> int:
        if decay_rate <= 0 or days <= 0:
            return 0
        if context.db_connection is None:
            logger.error(f"{tag('engine')} [衰减] 数据库连接未初始化")
            return 0

        try:
            decay_rate = min(decay_rate, 1.0)
            access_window_days = float(
                context.config.get("access_decay_window_days", 30.0)
            )
            max_access_count = float(
                context.config.get("access_decay_max_count", 10.0)
            )
            access_decay_multiplier = float(
                context.config.get("access_count_decay_multiplier", 0.5)
            )
            protected_threshold = clamp_float(
                context.config.get("protected_importance_threshold"),
                default=1.0,
            )
            access_window_start = (
                time.time() - max(1.0, access_window_days) * 86400.0
            )
            access_decay_multiplier = max(
                0.0,
                min(1.0, access_decay_multiplier),
            )
            cursor = await context.db_connection.execute(
                "SELECT id, metadata FROM documents "
                "WHERE json_extract(metadata, '$.importance') IS NOT NULL "
                "OR metadata LIKE '%\"importance\"%'"
            )
            rows = await cursor.fetchall()
            updates: list[tuple[str, int]] = []

            for row in rows:
                metadata = self.safe_json_dict(row["metadata"])
                importance = clamp_float(
                    metadata.get("importance"),
                    default=0.5,
                )
                if importance >= protected_threshold:
                    continue
                access_count = safe_float(metadata.get("access_count"), 0.0)
                last_access_time = safe_float(
                    metadata.get("last_access_time"),
                    0.0,
                )
                recent_access_factor = (
                    1.0 if last_access_time >= access_window_start else 0.5
                )
                access_factor = min(
                    1.0,
                    access_count / max(1.0, max_access_count),
                )
                effective_decay_rate = decay_rate * (
                    1 - 0.5 * access_factor * recent_access_factor
                )
                decay_factor = (1 - effective_decay_rate) ** days
                metadata["importance"] = max(
                    0.01,
                    round(importance * decay_factor, 4),
                )
                metadata["access_count"] = int(
                    access_count * access_decay_multiplier
                )
                updates.append(
                    (json.dumps(metadata, ensure_ascii=False), int(row["id"]))
                )

            if not updates:
                return 0

            await context.db_connection.executemany(
                "UPDATE documents SET metadata = ? WHERE id = ?",
                updates,
            )
            await context.db_connection.commit()
            affected = len(updates)
            logger.info(
                f"{tag('engine')} [衰减] 批量衰减完成: "
                f"衰减率={decay_rate}, 天数={days}, "
                f"访问窗口={access_window_days:.1f}天, 影响记录={affected}"
            )
            context.invalidate_search_cache()
            return affected
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"{tag('engine')} [衰减] 批量衰减失败: {exc}",
                exc_info=True,
            )
            return 0

    async def update_access_time(
        self,
        context: MemoryLifecycleContext,
        memory_id: int,
    ) -> bool:
        current_time = time.time()
        try:
            if context.db_connection is None:
                return False

            cursor = await context.db_connection.execute(
                "SELECT metadata FROM documents WHERE id = ?",
                (memory_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return False

            metadata = self.safe_json_dict(row[0] if row[0] else "{}")
            metadata["last_access_time"] = current_time
            try:
                access_count = int(metadata.get("access_count", 0) or 0)
            except (TypeError, ValueError):
                access_count = 0
            metadata["access_count"] = min(access_count + 1, 1_000_000)

            await context.db_connection.execute(
                "UPDATE documents SET metadata = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), memory_id),
            )
            await context.db_connection.commit()
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                f"{tag('engine')} 更新访问时间失败 "
                f"(memory_id={memory_id}): {exc}",
                exc_info=True,
            )
            return False

    async def cleanup_old_memories(
        self,
        context: MemoryLifecycleContext,
        days_threshold: int | None = None,
        importance_threshold: float | None = None,
    ) -> int:
        days = (
            context.config.get("cleanup_days_threshold", 30)
            if days_threshold is None
            else days_threshold
        )
        importance = (
            context.config.get("cleanup_importance_threshold", 0.3)
            if importance_threshold is None
            else importance_threshold
        )
        try:
            days = int(days)
            importance = float(importance)
        except (TypeError, ValueError):
            logger.error(
                f"{tag('engine')} 清理参数格式错误: "
                f"days_threshold={days}, importance_threshold={importance}"
            )
            return 0

        if days < 0:
            logger.error(
                f"{tag('engine')} 清理参数无效: "
                f"days_threshold={days}（必须 >= 0）"
            )
            return 0

        cutoff_time = time.time() - days * 86400
        try:
            total_count = await context.faiss_db.document_storage.count_documents(
                metadata_filters={}
            )
            if total_count == 0:
                return 0

            batch_size = 500
            offset = 0
            to_delete_ids: list[int] = []
            while offset < total_count:
                batch_docs = (
                    await context.faiss_db.document_storage.get_documents(
                        metadata_filters={},
                        limit=batch_size,
                        offset=offset,
                    )
                )
                if not batch_docs:
                    break

                batch_docs = await asyncio.to_thread(
                    context.document_repository.normalize_batch_metadata,
                    batch_docs,
                )
                for doc in batch_docs:
                    metadata = doc["metadata"]
                    create_time = safe_float(
                        metadata.get("create_time"),
                        time.time(),
                    )
                    doc_importance = clamp_float(
                        metadata.get("importance"),
                        default=0.5,
                    )
                    if create_time < cutoff_time and doc_importance < importance:
                        to_delete_ids.append(doc["id"])

                offset += len(batch_docs)
                if len(batch_docs) < batch_size:
                    break

            if not to_delete_ids:
                return 0

            logger.debug(
                f"{tag('engine')} [清理] 发现 {len(to_delete_ids)} 条候选记忆，"
                "开始批量删除"
            )
            deleted_count = await context.batch_delete_memories(to_delete_ids)
            logger.info(
                f"{tag('engine')} [清理] 完成，已删除 {deleted_count} 条旧记忆"
            )
            return deleted_count
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                f"{tag('engine')} [清理] 清理旧记忆失败",
                exc_info=True,
            )
            return 0

    async def migrate_session_data_if_needed(
        self,
        context: MemoryLifecycleContext,
        unified_msg_origin: str,
    ) -> None:
        """把未带平台前缀的旧 session_id 迁移为完整 UMO。"""
        umo_ref = log_ref(unified_msg_origin, "umo")
        try:
            parts = unified_msg_origin.split(":", 2)
            if len(parts) != 3:
                logger.warning(
                    f"{tag('engine')} [自动迁移] UMO 格式不正确 "
                    f"({umo_ref})"
                )
                return

            full_session_id = parts[2]
            candidates = [full_session_id]
            if "!" in full_session_id:
                parts_by_bang = full_session_id.split("!")
                for index in range(1, len(parts_by_bang)):
                    candidates.append("!".join(parts_by_bang[index:]))

            logger.debug(
                f"{tag('engine')} [自动迁移] 开始检查会话 "
                f"({umo_ref}, candidates={len(candidates)})"
            )
            migration_key = f"migrated_umo_{unified_msg_origin}"
            if context.db_connection is None:
                return

            cursor = await context.db_connection.execute(
                "SELECT value FROM migration_status WHERE key = ?",
                (migration_key,),
            )
            row = await cursor.fetchone()
            if row and row[0] == "true":
                return

            placeholders = " OR ".join(
                [
                    "json_extract(metadata, '$.session_id') = ?"
                    for _ in candidates
                ]
            )
            query = f"""
                SELECT id, metadata FROM documents
                WHERE ({placeholders})
                AND json_extract(metadata, '$.session_id') NOT LIKE '%:%'
            """
            cursor = await context.db_connection.execute(query, tuple(candidates))
            rows = list(await cursor.fetchall())

            if not rows:
                logger.debug(
                    f"{tag('engine')} [自动迁移] 未找到旧数据 ({umo_ref})"
                )
                await context.db_connection.execute(
                    "INSERT OR REPLACE INTO migration_status "
                    "(key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    (migration_key, "true"),
                )
                await context.db_connection.commit()
                return

            logger.debug(
                f"{tag('engine')} [自动迁移] 找到 {len(rows)} 条旧数据 "
                f"({umo_ref})"
            )
            updated_count = 0
            for row in rows:
                doc_id = row[0]
                metadata = self.safe_json_dict(row[1])
                old_session_id = metadata.get("session_id", "unknown")
                metadata["session_id"] = unified_msg_origin
                metadata["migrated_at"] = time.time()
                metadata["old_session_id"] = old_session_id
                await context.db_connection.execute(
                    "UPDATE documents SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False), doc_id),
                )
                updated_count += 1

            await context.db_connection.commit()
            await context.db_connection.execute(
                "INSERT OR REPLACE INTO migration_status "
                "(key, value, updated_at) VALUES (?, ?, datetime('now'))",
                (migration_key, "true"),
            )
            await context.db_connection.commit()
            logger.info(
                f"{tag('engine')} [自动迁移] 完成，已更新 "
                f"{updated_count} 条记录 ({umo_ref})"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"{tag('engine')} [自动迁移] 迁移失败 "
                f"({umo_ref}): {exc}",
                exc_info=True,
            )

    @staticmethod
    def safe_json_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not value:
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}


__all__ = ["MemoryLifecycleContext", "MemoryLifecycleService"]
