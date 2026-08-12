"""长期记忆统计聚合与 SQLite 存储维护。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...log import logger, tag
from ..utils.number_utils import clamp_float, safe_float
from .document_repository import DocumentRepository


class MemoryStatisticsService:
    """从 DocumentStorage 聚合统计并执行底层数据库维护。"""

    def __init__(self, db_path: str, faiss_db) -> None:
        self.db_path = db_path
        self.faiss_db = faiss_db

    async def get_statistics(
        self,
        persona_id: str | None = None,
        *,
        graph_store=None,
    ) -> dict[str, Any]:
        """按可选 persona 过滤聚合主记忆统计。"""
        try:
            metadata_filters = {"persona_id": persona_id} if persona_id else {}
            total_count = await self.faiss_db.document_storage.count_documents(
                metadata_filters=metadata_filters
            )

            session_counts: dict[str, int] = {}
            status_breakdown = {"active": 0, "archived": 0, "deleted": 0}
            importance_sum = 0.0
            importance_count = 0
            importance_distribution = {
                "0-1": 0,
                "1-2": 0,
                "2-3": 0,
                "3-4": 0,
                "4-5": 0,
                "5-6": 0,
                "6-7": 0,
                "7-8": 0,
                "8-9": 0,
                "9-10": 0,
            }
            bucket_keys = list(importance_distribution)
            oldest_time = None
            newest_time = None

            batch_size = 500
            offset = 0
            while offset < total_count:
                batch_docs = await self.faiss_db.document_storage.get_documents(
                    metadata_filters=metadata_filters,
                    limit=batch_size,
                    offset=offset,
                )
                if not batch_docs:
                    break

                batch_docs = await asyncio.to_thread(
                    DocumentRepository.normalize_batch_metadata,
                    batch_docs,
                )
                for doc in batch_docs:
                    metadata = doc["metadata"]
                    session_id = metadata.get("session_id")
                    if session_id:
                        session_counts[session_id] = (
                            session_counts.get(session_id, 0) + 1
                        )

                    status = metadata.get("status", "active")
                    if status in status_breakdown:
                        status_breakdown[status] += 1
                    else:
                        status_breakdown["active"] += 1

                    importance = metadata.get("importance")
                    if importance is not None:
                        clamped = clamp_float(importance, default=0.5)
                        importance_sum += clamped
                        importance_count += 1
                        display_importance = (
                            clamped * 10 if clamped <= 1 else clamped
                        )
                        bucket_idx = min(9, max(0, int(display_importance)))
                        importance_distribution[bucket_keys[bucket_idx]] += 1

                    create_time = metadata.get("create_time")
                    if create_time:
                        normalized_time = safe_float(create_time, 0.0)
                        if oldest_time is None or normalized_time < oldest_time:
                            oldest_time = normalized_time
                        if newest_time is None or normalized_time > newest_time:
                            newest_time = normalized_time

                offset += batch_size

            stats: dict[str, Any] = {
                "total_memories": total_count,
                "sessions": session_counts,
                "status_breakdown": status_breakdown,
                "avg_importance": (
                    importance_sum / importance_count
                    if importance_count > 0
                    else 0.0
                ),
                "importance_distribution": importance_distribution,
                "oldest_memory": oldest_time,
                "newest_memory": newest_time,
            }
            if graph_store is not None:
                stats.update(await graph_store.get_memory_entry_stats())
                stats["graph_memory_enabled"] = True
            else:
                stats["graph_memory_enabled"] = False
            return stats
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                f"{tag('memory-stats')} 获取统计信息失败",
                exc_info=True,
            )
            return {
                "total_memories": 0,
                "sessions": {},
                "status_breakdown": {
                    "active": 0,
                    "archived": 0,
                    "deleted": 0,
                },
                "avg_importance": 0.0,
                "oldest_memory": None,
                "newest_memory": None,
                "graph_memory_enabled": bool(graph_store is not None),
            }

    async def maintain_storage(
        self,
        db_connection,
        *,
        vacuum: bool = False,
    ) -> dict[str, Any]:
        """执行 FTS optimize、WAL checkpoint 和可选 VACUUM。"""
        try:
            db_path = Path(self.db_path)
            wal_path = Path(f"{self.db_path}-wal")
            before_size = db_path.stat().st_size if db_path.exists() else 0
            before_wal_size = wal_path.stat().st_size if wal_path.exists() else 0

            if db_connection is None:
                return {
                    "success": False,
                    "error": "database connection is not initialized",
                }

            for fts_table in (
                "livingmemory_graph_entries_fts",
                "memory_atoms_fts",
            ):
                try:
                    await db_connection.execute(
                        f"INSERT INTO {fts_table}({fts_table}) VALUES ('optimize')"
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug(
                        f"{tag('memory-stats')} 跳过 FTS optimize: {fts_table}",
                        exc_info=True,
                    )

            await db_connection.commit()
            await db_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if vacuum:
                await db_connection.execute("VACUUM")

            after_size = db_path.stat().st_size if db_path.exists() else 0
            after_wal_size = wal_path.stat().st_size if wal_path.exists() else 0
            return {
                "success": True,
                "vacuum": vacuum,
                "db_size_before": before_size,
                "db_size_after": after_size,
                "wal_size_before": before_wal_size,
                "wal_size_after": after_wal_size,
                "bytes_reclaimed": max(
                    0,
                    before_size
                    + before_wal_size
                    - after_size
                    - after_wal_size,
                ),
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"{tag('memory-stats')} 执行存储维护失败: {exc}",
                exc_info=True,
            )
            return {"success": False, "error": str(exc)}
