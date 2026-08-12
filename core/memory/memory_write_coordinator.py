"""多存储记忆写入编排与可恢复操作日志。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ...log import logger, tag
from ..utils.number_utils import clamp_float


@dataclass(slots=True)
class MemoryWriteContext:
    """一次写入调用所需的动态引擎依赖。"""

    db_connection: Any
    faiss_db: Any
    hybrid_retriever: Any
    atom_store: Any
    graph_memory_manager: Any
    atom_enabled: bool
    get_memory: Callable[[int], Awaitable[dict[str, Any] | None]]
    find_memory_by_idempotency_key: Callable[[str], Awaitable[int | None]]
    add_memory: Callable[..., Awaitable[int | None]]
    delete_memory: Callable[[int], Awaitable[bool]]
    start_write_op: Callable[..., Awaitable[int | None]]
    advance_write_op: Callable[..., Awaitable[None]]
    serialize_atom_for_repair: Callable[[Any], dict[str, Any]]
    delete_graph_and_atoms_for_batch: Callable[[list[int]], Awaitable[None]]
    invalidate_search_cache: Callable[[], None]


class MemoryWriteCoordinator:
    """协调文档、向量、原子和图谱存储的记忆写入。"""

    async def create_write_ops_table(self, context: MemoryWriteContext) -> None:
        """创建可恢复写操作日志。"""
        if context.db_connection is None:
            return
        await context.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_write_ops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                op_type TEXT NOT NULL,
                memory_id INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                step TEXT NOT NULL DEFAULT 'started',
                payload TEXT DEFAULT '{}',
                error TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        await context.db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_write_ops_status
            ON memory_write_ops(status, updated_at)
        """)
        await context.db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_write_ops_memory
            ON memory_write_ops(memory_id, op_type)
        """)

    async def start_write_op(
        self,
        context: MemoryWriteContext,
        op_type: str,
        payload: dict[str, Any] | None = None,
        memory_id: int | None = None,
    ) -> int | None:
        """记录一次多存储写操作的开始状态。"""
        if context.db_connection is None:
            return None
        now = time.time()
        try:
            cursor = await context.db_connection.execute(
                """
                INSERT INTO memory_write_ops(
                    op_type, memory_id, status, step, payload,
                    created_at, updated_at
                ) VALUES (?, ?, 'pending', 'started', ?, ?, ?)
                """,
                (
                    op_type,
                    memory_id,
                    json.dumps(payload or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            await context.db_connection.commit()
            return int(cursor.lastrowid)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(f"{tag('engine')} 写操作日志创建失败", exc_info=True)
            return None

    async def advance_write_op(
        self,
        context: MemoryWriteContext,
        op_id: int | None,
        step: str,
        *,
        status: str = "pending",
        memory_id: int | None = None,
        error: str | None = None,
        payload_patch: dict[str, Any] | None = None,
    ) -> None:
        """推进写操作日志的状态机。"""
        if op_id is None or context.db_connection is None:
            return

        try:
            if status == "completed":
                error = None
            current_payload: dict[str, Any] = {}
            if payload_patch:
                cursor = await context.db_connection.execute(
                    "SELECT payload FROM memory_write_ops WHERE id = ?",
                    (op_id,),
                )
                row = await cursor.fetchone()
                if row and row[0]:
                    try:
                        loaded = json.loads(row[0])
                        current_payload = loaded if isinstance(loaded, dict) else {}
                    except (json.JSONDecodeError, TypeError):
                        current_payload = {}
                current_payload.update(payload_patch)

            fields = ["status = ?", "step = ?", "updated_at = ?"]
            params: list[Any] = [status, step, time.time()]
            if memory_id is not None:
                fields.append("memory_id = ?")
                params.append(memory_id)
            if error is not None:
                fields.append("error = ?")
                params.append(error[:1000])
                if status != "completed":
                    fields.append("retry_count = retry_count + 1")
            elif status == "completed":
                fields.append("error = NULL")
            if payload_patch:
                fields.append("payload = ?")
                params.append(json.dumps(current_payload, ensure_ascii=False))
            params.append(op_id)
            await context.db_connection.execute(
                f"UPDATE memory_write_ops SET {', '.join(fields)} WHERE id = ?",
                params,
            )
            await context.db_connection.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(f"{tag('engine')} 写操作日志更新失败", exc_info=True)

    async def add_memory(
        self,
        context: MemoryWriteContext,
        content: str,
        session_id: str | None = None,
        persona_id: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        atoms: list | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        """写入主文档，并尽力同步原子与图谱索引。"""
        if not content or not content.strip():
            raise ValueError("记忆内容不能为空")

        normalized_idempotency_key = str(idempotency_key or "").strip()
        if normalized_idempotency_key:
            existing_id = await context.find_memory_by_idempotency_key(
                normalized_idempotency_key
            )
            if existing_id is not None:
                logger.info(
                    f"{tag('engine')} 幂等写入命中，复用已有记忆 "
                    f"(memory_id={existing_id})"
                )
                return existing_id

        op_id = await context.start_write_op(
            "add",
            {
                "content_preview": content[:500],
                "session_id": session_id,
                "persona_id": persona_id,
                "importance": importance,
                "metadata": metadata or {},
                "idempotency_key": normalized_idempotency_key or None,
                "atoms": [
                    context.serialize_atom_for_repair(atom) for atom in (atoms or [])
                ],
            },
        )

        current_time = time.time()
        full_metadata = {
            "session_id": session_id,
            "persona_id": persona_id,
            "importance": max(0.0, min(1.0, importance)),
            "create_time": current_time,
            "last_access_time": current_time,
        }
        if metadata:
            full_metadata.update(metadata)
        if normalized_idempotency_key:
            full_metadata["idempotency_key"] = normalized_idempotency_key
        full_metadata["create_time"] = current_time
        full_metadata["last_access_time"] = current_time

        if context.hybrid_retriever is None:
            raise RuntimeError("混合检索器未初始化")
        try:
            doc_id = await context.hybrid_retriever.add_memory(content, full_metadata)
            await context.advance_write_op(
                op_id,
                "document_indexed",
                memory_id=doc_id,
                payload_patch={"memory_id": doc_id},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await context.advance_write_op(
                op_id,
                "document_failed",
                status="failed",
                error=str(exc),
            )
            raise

        atom_write_failed = False
        if atoms and context.atom_store is not None and context.atom_enabled:
            prepared_atoms = []
            for atom in atoms:
                atom.session_id = atom.session_id or session_id
                atom.persona_id = atom.persona_id or persona_id
                atom.parent_memory_id = doc_id
                prepared_atoms.append(atom)
            try:
                await context.atom_store.insert_many(prepared_atoms)
                await context.advance_write_op(
                    op_id,
                    "atoms_indexed",
                    memory_id=doc_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(f"{tag('engine')} 批量写入记忆原子失败", exc_info=True)
                failed_atoms: list[dict[str, Any]] = []
                for atom in prepared_atoms:
                    if getattr(atom, "atom_id", 0):
                        continue
                    try:
                        await context.atom_store.insert(atom)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        failed_atoms.append(context.serialize_atom_for_repair(atom))
                        logger.error(
                            f"{tag('engine')} 写入记忆原子失败 "
                            f"(memory_id={doc_id})",
                            exc_info=True,
                        )
                if failed_atoms:
                    await context.advance_write_op(
                        op_id,
                        "atoms_partial",
                        status="needs_repair",
                        memory_id=doc_id,
                        error="atom insert failed",
                        payload_patch={"failed_atoms": failed_atoms},
                    )
                    atom_write_failed = True
                else:
                    await context.advance_write_op(
                        op_id,
                        "atoms_indexed",
                        memory_id=doc_id,
                    )
        else:
            await context.advance_write_op(
                op_id,
                "atoms_skipped",
                memory_id=doc_id,
            )

        needs_repair = atom_write_failed
        if context.graph_memory_manager is not None:
            try:
                await context.graph_memory_manager.index_memory(
                    doc_id,
                    content,
                    full_metadata,
                    atoms,
                )
                await context.advance_write_op(
                    op_id,
                    "graph_indexed",
                    status="needs_repair" if needs_repair else "pending",
                    memory_id=doc_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await context.advance_write_op(
                    op_id,
                    "graph_failed",
                    status="needs_repair",
                    memory_id=doc_id,
                    error=str(exc),
                )
                needs_repair = True
                logger.error(
                    f"{tag('engine')} 图记忆索引失败，已标记待修复 "
                    f"(memory_id={doc_id})",
                    exc_info=True,
                )
        else:
            await context.advance_write_op(
                op_id,
                "graph_skipped",
                status="needs_repair" if needs_repair else "pending",
                memory_id=doc_id,
            )

        if not needs_repair:
            await context.advance_write_op(
                op_id,
                "completed",
                status="completed",
                memory_id=doc_id,
            )
        context.invalidate_search_cache()
        return doc_id

    async def update_memory(
        self,
        context: MemoryWriteContext,
        memory_id: int,
        updates: dict[str, Any],
    ) -> bool:
        """更新记忆内容或同步元数据。"""
        memory = await context.get_memory(memory_id)
        if not memory:
            logger.error(f"{tag('engine')} [更新] 记忆不存在 (memory_id={memory_id})")
            return False

        current_metadata = memory.get("metadata", {})
        if isinstance(current_metadata, str):
            try:
                current_metadata = json.loads(current_metadata)
            except (json.JSONDecodeError, TypeError):
                current_metadata = {}
        elif not isinstance(current_metadata, dict):
            current_metadata = {}

        if "content" in updates:
            new_content = updates["content"]
            if not new_content or not new_content.strip():
                return False

            try:
                session_id = current_metadata.get("session_id")
                persona_id = current_metadata.get("persona_id")
                importance = clamp_float(
                    current_metadata.get("importance", updates.get("importance", 0.5)),
                    default=0.5,
                )
                new_metadata = current_metadata.copy()
                new_metadata["updated_at"] = time.time()
                new_metadata["previous_id"] = memory_id

                logger.debug(
                    f"{tag('engine')} [更新] 开始内容更新流程 "
                    f"(old_id={memory_id})"
                )
                new_memory_id = await context.add_memory(
                    content=new_content,
                    session_id=session_id,
                    persona_id=persona_id,
                    importance=importance,
                    metadata=new_metadata,
                )
                if new_memory_id is None:
                    logger.error(
                        f"{tag('engine')} [更新] 创建新记忆失败 "
                        f"(old_id={memory_id})"
                    )
                    return False

                logger.debug(
                    f"{tag('engine')} [更新] 新记忆已创建 "
                    f"(new_id={new_memory_id})"
                )
                delete_success = await context.delete_memory(memory_id)
                if not delete_success:
                    logger.warning(
                        f"{tag('engine')} [更新] 删除旧记忆失败，回滚新记忆 "
                        f"(old_id={memory_id}, new_id={new_memory_id})"
                    )
                    await context.delete_memory(new_memory_id)
                    return False

                logger.debug(
                    f"{tag('engine')} [更新] 内容更新完成 "
                    f"(old_id={memory_id} → new_id={new_memory_id})"
                )
                context.invalidate_search_cache()
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    f"{tag('engine')} [更新] 内容更新失败 "
                    f"(memory_id={memory_id}): {exc}",
                    exc_info=True,
                )
                return False

        metadata_updates: dict[str, Any] = {}
        if "importance" in updates:
            metadata_updates["importance"] = clamp_float(
                updates["importance"],
                default=0.5,
            )
        if "metadata" in updates:
            metadata_updates.update(updates["metadata"])

        if metadata_updates:
            if not isinstance(current_metadata, dict):
                try:
                    current_metadata = (
                        json.loads(current_metadata)
                        if isinstance(current_metadata, str)
                        else {}
                    )
                except (json.JSONDecodeError, TypeError):
                    current_metadata = {}

            current_metadata.update(metadata_updates)
            current_metadata["updated_at"] = time.time()
            if context.hybrid_retriever is None:
                logger.error(f"{tag('engine')} 混合检索器未初始化")
                return False
            success = await context.hybrid_retriever.update_metadata(
                memory_id,
                metadata_updates,
            )
            if success:
                logger.debug(
                    f"{tag('engine')} [更新] 元数据更新成功 "
                    f"(memory_id={memory_id})"
                )
                if context.graph_memory_manager is not None:
                    await context.graph_memory_manager.index_memory(
                        memory_id,
                        memory["text"],
                        current_metadata,
                    )
                context.invalidate_search_cache()
            else:
                logger.error(
                    f"{tag('engine')} [更新] 元数据更新失败 "
                    f"(memory_id={memory_id})"
                )
            return success

        return True

    async def delete_memory(
        self,
        context: MemoryWriteContext,
        memory_id: int,
    ) -> bool:
        """从主文档、图谱和原子存储删除一条记忆。"""
        op_id = await context.start_write_op(
            "delete",
            {"memory_id": memory_id},
            memory_id=memory_id,
        )

        if context.hybrid_retriever is None:
            logger.error(f"{tag('engine')} 混合检索器未初始化")
            await context.advance_write_op(
                op_id,
                "document_delete_failed",
                status="failed",
                error="hybrid retriever not initialized",
            )
            return False
        success = await context.hybrid_retriever.delete_memory(memory_id)
        if not success:
            await context.advance_write_op(
                op_id,
                "document_delete_failed",
                status="failed",
                error="document/vector delete failed",
            )
            return False

        await context.advance_write_op(
            op_id,
            "document_deleted",
            memory_id=memory_id,
        )
        needs_repair = False
        try:
            if context.graph_memory_manager is not None:
                await context.graph_memory_manager.delete_memory(memory_id)
            await context.advance_write_op(
                op_id,
                "graph_deleted",
                memory_id=memory_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await context.advance_write_op(
                op_id,
                "graph_delete_failed",
                status="needs_repair",
                memory_id=memory_id,
                error=str(exc),
            )
            needs_repair = True
            logger.error(
                f"{tag('engine')} 图记忆删除失败，已标记待修复 "
                f"(memory_id={memory_id})",
                exc_info=True,
            )

        try:
            if context.atom_store is not None:
                await context.atom_store.delete_by_parent(memory_id)
            await context.advance_write_op(
                op_id,
                "atoms_deleted",
                memory_id=memory_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await context.advance_write_op(
                op_id,
                "atom_delete_failed",
                status="needs_repair",
                memory_id=memory_id,
                error=str(exc),
            )
            needs_repair = True
            logger.error(
                f"{tag('engine')} 记忆原子删除失败，已标记待修复 "
                f"(memory_id={memory_id})",
                exc_info=True,
            )

        if not needs_repair:
            await context.advance_write_op(
                op_id,
                "completed",
                status="completed",
                memory_id=memory_id,
            )
        context.invalidate_search_cache()
        return success

    async def batch_delete_memories(
        self,
        context: MemoryWriteContext,
        memory_ids: list[int],
    ) -> int:
        """以小批次删除多条记忆并记录可恢复进度。"""
        if not memory_ids:
            return 0
        if context.db_connection is None:
            logger.error(f"{tag('engine')} [批量删除] 数据库连接未初始化")
            return 0

        context.invalidate_search_cache()
        total_deleted = 0
        sql_batch_size = 200
        for offset in range(0, len(memory_ids), sql_batch_size):
            batch = memory_ids[offset : offset + sql_batch_size]
            placeholders = ",".join("?" * len(batch))
            op_id = await context.start_write_op(
                "batch_delete",
                {
                    "memory_ids": batch,
                    "batch_offset": offset,
                    "batch_size": len(batch),
                },
            )
            batch_deleted = 0

            try:
                cursor = await context.db_connection.execute(
                    f"SELECT id, doc_id FROM documents WHERE id IN ({placeholders})",
                    batch,
                )
                uuid_rows = await cursor.fetchall()
                found_ids = [int(row["id"]) for row in uuid_rows]
                for row in uuid_rows:
                    uuid_doc_id = row["doc_id"]
                    if not uuid_doc_id:
                        continue
                    try:
                        await context.faiss_db.delete(uuid_doc_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.warning(
                            f"{tag('engine')} [批量删除] FAISS 删除失败 "
                            f"(id={row['id']})",
                            exc_info=True,
                        )
                await context.advance_write_op(
                    op_id,
                    "faiss_deleted",
                    payload_patch={"memory_ids": batch, "found_ids": found_ids},
                )

                cursor = await context.db_connection.execute(
                    f"DELETE FROM documents WHERE id IN ({placeholders})",
                    batch,
                )
                await context.db_connection.commit()
                batch_deleted = int(cursor.rowcount or 0)
                await context.advance_write_op(
                    op_id,
                    "documents_deleted",
                    payload_patch={
                        "memory_ids": batch,
                        "found_ids": found_ids,
                        "deleted_count": batch_deleted,
                    },
                )

                await context.delete_graph_and_atoms_for_batch(batch)
                await context.advance_write_op(
                    op_id,
                    "graph_atoms_deleted",
                    payload_patch={
                        "memory_ids": batch,
                        "deleted_count": batch_deleted,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await context.advance_write_op(
                    op_id,
                    "batch_delete_failed",
                    status="needs_repair",
                    error=str(exc),
                    payload_patch={
                        "memory_ids": batch,
                        "deleted_count": batch_deleted,
                    },
                )
                logger.error(
                    f"{tag('engine')} [批量删除] 批次删除失败 "
                    f"(offset={offset}, size={len(batch)})",
                    exc_info=True,
                )
                raise

            await context.advance_write_op(
                op_id,
                "completed",
                status="completed",
                payload_patch={
                    "memory_ids": batch,
                    "deleted_count": batch_deleted,
                },
            )
            total_deleted += batch_deleted

        if total_deleted:
            logger.info(
                f"{tag('engine')} [批量删除] 共删除 {total_deleted} 条记忆"
            )
        return total_deleted


__all__ = ["MemoryWriteContext", "MemoryWriteCoordinator"]
