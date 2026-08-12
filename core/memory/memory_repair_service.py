"""跨存储写操作日志的恢复与副索引收敛。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ...log import logger, tag
from ..models.memory_atom import AtomStatus, AtomType, DecayType, MemoryAtom


@dataclass(slots=True)
class MemoryRepairContext:
    """一次恢复调用所需的动态依赖。"""

    db_connection: Any
    faiss_db: Any
    atom_store: Any
    graph_memory_manager: Any
    atom_enabled: bool
    max_retries: int
    get_memory: Callable[[int], Awaitable[dict[str, Any] | None]]
    advance_write_op: Callable[..., Awaitable[None]]
    invalidate_search_cache: Callable[[], None]


class MemoryRepairService:
    """重放未完成的 add/delete/batch_delete 写操作。"""

    @staticmethod
    def serialize_atom(atom: Any) -> dict[str, Any]:
        """把 MemoryAtom-like 对象转换为 JSON 安全的恢复载荷。"""
        atom_type = getattr(atom, "atom_type", AtomType.UNKNOWN)
        decay_type = getattr(atom, "decay_type", DecayType.EXPONENTIAL)
        status = getattr(atom, "status", AtomStatus.ACTIVE)
        return {
            "parent_memory_id": int(getattr(atom, "parent_memory_id", 0) or 0),
            "atom_type": getattr(atom_type, "value", str(atom_type)),
            "content": str(getattr(atom, "content", "")),
            "entities": list(getattr(atom, "entities", []) or []),
            "importance": float(getattr(atom, "importance", 0.5) or 0.5),
            "confidence": float(getattr(atom, "confidence", 0.7) or 0.7),
            "created_at": float(
                getattr(atom, "created_at", time.time()) or time.time()
            ),
            "last_accessed_at": float(
                getattr(atom, "last_accessed_at", time.time()) or time.time()
            ),
            "last_reinforced_at": getattr(atom, "last_reinforced_at", None),
            "event_time": getattr(atom, "event_time", None),
            "ttl_days": float(getattr(atom, "ttl_days", 30.0) or 30.0),
            "expires_at": float(getattr(atom, "expires_at", 0.0) or 0.0),
            "status": getattr(status, "value", str(status)),
            "reinforcement_count": int(
                getattr(atom, "reinforcement_count", 0) or 0
            ),
            "decay_type": getattr(decay_type, "value", str(decay_type)),
            "session_id": getattr(atom, "session_id", None),
            "persona_id": getattr(atom, "persona_id", None),
            "metadata": dict(getattr(atom, "metadata", {}) or {}),
        }

    @staticmethod
    def deserialize_atom(
        payload: dict[str, Any],
        parent_memory_id: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> MemoryAtom | None:
        """从恢复载荷重建 MemoryAtom。"""
        content = str(payload.get("content") or "")
        if not content.strip():
            return None

        try:
            atom_type = AtomType(payload.get("atom_type") or AtomType.UNKNOWN.value)
        except ValueError:
            atom_type = AtomType.UNKNOWN
        try:
            decay_type = DecayType(
                payload.get("decay_type") or DecayType.EXPONENTIAL.value
            )
        except ValueError:
            decay_type = DecayType.EXPONENTIAL
        try:
            status = AtomStatus(payload.get("status") or AtomStatus.ACTIVE.value)
        except ValueError:
            status = AtomStatus.ACTIVE

        return MemoryAtom(
            parent_memory_id=parent_memory_id,
            atom_type=atom_type,
            content=content,
            entities=[str(item) for item in payload.get("entities", []) if item],
            importance=float(payload.get("importance", 0.5) or 0.5),
            confidence=float(payload.get("confidence", 0.7) or 0.7),
            created_at=float(payload.get("created_at", time.time()) or time.time()),
            last_accessed_at=float(
                payload.get("last_accessed_at", time.time()) or time.time()
            ),
            last_reinforced_at=payload.get("last_reinforced_at"),
            event_time=payload.get("event_time"),
            ttl_days=float(payload.get("ttl_days", 30.0) or 30.0),
            expires_at=float(payload.get("expires_at", 0.0) or 0.0),
            status=status,
            reinforcement_count=int(payload.get("reinforcement_count", 0) or 0),
            decay_type=decay_type,
            session_id=payload.get("session_id") or session_id,
            persona_id=payload.get("persona_id") or persona_id,
            metadata=dict(payload.get("metadata") or {}),
        )

    async def repair_incomplete_write_ops(
        self,
        context: MemoryRepairContext,
    ) -> int:
        """尽力重放未完成的 add/delete/batch_delete。"""
        if context.db_connection is None:
            return 0

        try:
            cursor = await context.db_connection.execute(
                """
                SELECT id, op_type, memory_id, status, step, payload, retry_count
                FROM memory_write_ops
                WHERE status IN ('pending', 'needs_repair')
                  AND retry_count < ?
                ORDER BY id ASC
                LIMIT 25
                """,
                (context.max_retries,),
            )
            rows = await cursor.fetchall()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                f"{tag('memory-repair')} 读取待修复写操作失败",
                exc_info=True,
            )
            return 0

        repaired = 0
        for row in rows:
            payload = self.safe_json_dict(row["payload"])
            try:
                op_type = row["op_type"]
                memory_id = row["memory_id"]
                if op_type == "add":
                    ok = await self.repair_add_write_op(
                        context,
                        int(row["id"]),
                        int(memory_id) if memory_id is not None else None,
                        payload,
                    )
                elif op_type == "delete":
                    ok = await self.repair_delete_write_op(
                        context,
                        int(row["id"]),
                        int(memory_id) if memory_id is not None else None,
                    )
                elif op_type == "batch_delete":
                    ok = await self.repair_batch_delete_write_op(
                        context,
                        int(row["id"]),
                        payload,
                    )
                else:
                    ok = False
                repaired += 1 if ok else 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    f"{tag('memory-repair')} 修复写操作失败 "
                    f"(op_id={row['id']})",
                    exc_info=True,
                )
                await context.advance_write_op(
                    int(row["id"]),
                    str(row["step"] or "repair_failed"),
                    status="needs_repair",
                    error=str(exc),
                )

        if repaired:
            logger.info(
                f"{tag('memory-repair')} 已修复 {repaired} 个未完成写操作"
            )
            context.invalidate_search_cache()
        return repaired

    async def repair_add_write_op(
        self,
        context: MemoryRepairContext,
        op_id: int,
        memory_id: int | None,
        payload: dict[str, Any],
    ) -> bool:
        if memory_id is None:
            await context.advance_write_op(
                op_id,
                "unrepairable",
                status="failed",
                error="missing memory_id for add repair",
            )
            return False

        memory = await context.get_memory(int(memory_id))
        if memory is None:
            await context.advance_write_op(
                op_id,
                "source_missing",
                status="failed",
                memory_id=int(memory_id),
                error="source document missing",
            )
            return False

        metadata = memory.get("metadata") or payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = self.safe_json_dict(metadata)
        content = str(memory.get("text") or "")
        session_id = metadata.get("session_id") or payload.get("session_id")
        persona_id = metadata.get("persona_id") or payload.get("persona_id")

        atom_payloads = payload.get("failed_atoms") or payload.get("atoms", []) or []
        atoms: list[MemoryAtom] = []
        for atom_payload in atom_payloads:
            if isinstance(atom_payload, dict):
                atom = self.deserialize_atom(
                    atom_payload,
                    int(memory_id),
                    session_id,
                    persona_id,
                )
                if atom is not None:
                    atoms.append(atom)

        if context.atom_store is not None and atoms and context.atom_enabled:
            existing_atoms = await context.atom_store.get_by_parent(int(memory_id))
            if payload.get("failed_atoms"):
                existing_keys = {
                    (
                        atom.content,
                        atom.atom_type.value,
                        atom.session_id,
                        atom.persona_id,
                    )
                    for atom in existing_atoms
                }
                atoms_to_insert = [
                    atom
                    for atom in atoms
                    if (
                        atom.content,
                        atom.atom_type.value,
                        atom.session_id,
                        atom.persona_id,
                    )
                    not in existing_keys
                ]
                if atoms_to_insert:
                    await context.atom_store.insert_many(atoms_to_insert)
            elif not existing_atoms:
                await context.atom_store.insert_many(atoms)
            await context.advance_write_op(
                op_id,
                "atoms_repaired",
                memory_id=memory_id,
            )

        if context.graph_memory_manager is not None and content.strip():
            await context.graph_memory_manager.index_memory(
                int(memory_id),
                content,
                metadata,
                atoms or None,
            )
            await context.advance_write_op(
                op_id,
                "graph_repaired",
                memory_id=memory_id,
            )

        await context.advance_write_op(
            op_id,
            "completed",
            status="completed",
            memory_id=int(memory_id),
        )
        return True

    async def repair_delete_write_op(
        self,
        context: MemoryRepairContext,
        op_id: int,
        memory_id: int | None,
    ) -> bool:
        if memory_id is None:
            await context.advance_write_op(
                op_id,
                "unrepairable",
                status="failed",
                error="missing memory_id for delete repair",
            )
            return False

        if context.graph_memory_manager is not None:
            await context.graph_memory_manager.delete_memory(int(memory_id))
        if context.atom_store is not None:
            await context.atom_store.delete_by_parent(int(memory_id))

        await context.advance_write_op(
            op_id,
            "completed",
            status="completed",
            memory_id=int(memory_id),
        )
        return True

    async def repair_batch_delete_write_op(
        self,
        context: MemoryRepairContext,
        op_id: int,
        payload: dict[str, Any],
    ) -> bool:
        memory_ids_raw = payload.get("memory_ids") or []
        if not isinstance(memory_ids_raw, list):
            await context.advance_write_op(
                op_id,
                "unrepairable",
                status="failed",
                error="missing memory_ids for batch delete repair",
            )
            return False

        memory_ids: list[int] = []
        for raw_id in memory_ids_raw:
            try:
                memory_ids.append(int(raw_id))
            except (TypeError, ValueError):
                continue

        if not memory_ids:
            await context.advance_write_op(
                op_id,
                "unrepairable",
                status="failed",
                error="empty memory_ids for batch delete repair",
            )
            return False

        await self.delete_document_indexes_for_batch(context, memory_ids)
        await self.delete_graph_and_atoms_for_batch(context, memory_ids)
        await context.advance_write_op(
            op_id,
            "completed",
            status="completed",
            payload_patch={"deleted_count": len(memory_ids)},
        )
        return True

    async def delete_document_indexes_for_batch(
        self,
        context: MemoryRepairContext,
        memory_ids: list[int],
    ) -> int:
        if not memory_ids or context.db_connection is None:
            return 0

        placeholders = ",".join("?" * len(memory_ids))
        cursor = await context.db_connection.execute(
            f"SELECT id, doc_id FROM documents WHERE id IN ({placeholders})",
            memory_ids,
        )
        uuid_rows = await cursor.fetchall()
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
                    f"{tag('memory-repair')} [批量删除] FAISS 删除失败 "
                    f"(id={row['id']})",
                    exc_info=True,
                )

        cursor = await context.db_connection.execute(
            f"DELETE FROM documents WHERE id IN ({placeholders})",
            memory_ids,
        )
        await context.db_connection.commit()
        return int(cursor.rowcount or 0)

    @staticmethod
    async def delete_graph_and_atoms_for_batch(
        context: MemoryRepairContext,
        memory_ids: list[int],
    ) -> None:
        if not memory_ids:
            return
        if context.graph_memory_manager is not None:
            await context.graph_memory_manager.batch_delete_memories(memory_ids)
        if context.atom_store is not None:
            await context.atom_store.batch_delete_by_parent(memory_ids)

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
