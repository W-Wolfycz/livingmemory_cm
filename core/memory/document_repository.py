"""主记忆文档的只读访问与批量元数据规范化。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ...log import log_ref, logger, tag
from ..utils.number_utils import safe_float


class DocumentRepository:
    """封装 FaissVecDB DocumentStorage 的只读查询。"""

    def __init__(self, faiss_db) -> None:
        self.faiss_db = faiss_db

    async def find_by_idempotency_key(self, idempotency_key: str) -> int | None:
        """按幂等键查找已存在的主记忆 ID。"""
        if not idempotency_key:
            return None
        try:
            docs = await self.faiss_db.document_storage.get_documents(
                metadata_filters={"idempotency_key": idempotency_key},
                limit=1,
            )
            if not docs:
                return None
            return int(docs[0]["id"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                f"{tag('memory-doc')} 查询幂等键失败，将继续执行写入",
                exc_info=True,
            )
            return None

    async def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        """根据 DocumentStorage 整数 ID 获取主记忆。"""
        try:
            docs = await self.faiss_db.document_storage.get_documents(
                metadata_filters={}, ids=[memory_id], limit=1
            )
            if not docs:
                return None

            doc = docs[0]
            return {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
            }
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(f"{tag('memory-doc')} 获取记忆详情失败", exc_info=True)
            return None

    async def get_session_memories(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """按会话读取并按创建时间倒序返回主记忆。"""
        try:
            total_count = await self.faiss_db.document_storage.count_documents(
                metadata_filters={"session_id": session_id}
            )
            if total_count == 0:
                return []

            if total_count <= limit:
                all_docs = await self.faiss_db.document_storage.get_documents(
                    metadata_filters={"session_id": session_id},
                    limit=limit,
                    offset=0,
                )
                all_docs = await asyncio.to_thread(
                    self.normalize_batch_metadata,
                    all_docs,
                )
                sorted_docs = sorted(
                    all_docs,
                    key=lambda doc: safe_float(
                        doc.get("metadata", {}).get("create_time"), 0.0
                    ),
                    reverse=True,
                )
            else:
                all_docs: list[dict[str, Any]] = []
                batch_size = 500
                offset = 0

                while offset < total_count:
                    batch = await self.faiss_db.document_storage.get_documents(
                        metadata_filters={"session_id": session_id},
                        limit=batch_size,
                        offset=offset,
                    )
                    if not batch:
                        break

                    batch = await asyncio.to_thread(
                        self.normalize_batch_metadata,
                        batch,
                    )
                    all_docs.extend(batch)
                    offset += batch_size

                sorted_docs = sorted(
                    all_docs,
                    key=lambda doc: safe_float(
                        doc.get("metadata", {}).get("create_time"), 0.0
                    ),
                    reverse=True,
                )[:limit]

            return [
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                }
                for doc in sorted_docs
            ]
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                f"{tag('memory-doc')} 获取会话记忆失败 "
                f"(session={log_ref(session_id, 'session')})",
                exc_info=True,
            )
            return []

    @staticmethod
    def normalize_batch_metadata(
        docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """把批量文档中的 JSON metadata 规范为字典。"""
        for doc in docs:
            metadata = doc.get("metadata")
            if isinstance(metadata, str):
                try:
                    doc["metadata"] = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    doc["metadata"] = {}
            elif not isinstance(metadata, dict):
                doc["metadata"] = {}
        return docs
