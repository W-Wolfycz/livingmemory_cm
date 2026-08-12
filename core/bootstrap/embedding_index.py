"""Embedding Provider 指纹与启动前 FAISS 影子重建。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..base.exceptions import InitializationError
from ...log import logger, tag
from .faiss_bootstrap import FaissBootstrapService


@dataclass(frozen=True, slots=True)
class EmbeddingIndexSpec:
    """一个 SQLite 文档库与其 FAISS 索引的绑定。"""

    key: str
    document_path: Path
    index_path: Path
    vector_schema: str = "document-v1"


@dataclass(frozen=True, slots=True)
class EmbeddingIndexResult:
    """单个索引启动检查的结果。"""

    key: str
    action: str
    document_count: int
    reason: str


class EmbeddingFingerprintService:
    """生成不包含凭据明文的 Embedding Provider 语义指纹。"""

    SCHEMA = "livingmemory-cm-vector-v1"

    @staticmethod
    def _provider_config(provider: Any) -> dict[str, Any]:
        value = getattr(provider, "provider_config", None)
        return dict(value) if isinstance(value, Mapping) else {}

    @classmethod
    def payload(cls, provider: Any) -> dict[str, Any]:
        config = cls._provider_config(provider)
        get_model = getattr(provider, "get_model", None)
        try:
            model = get_model() if callable(get_model) else None
        except Exception:
            model = None
        model = (
            model
            or getattr(provider, "model", None)
            or getattr(provider, "model_name", None)
            or config.get("embedding_model")
            or config.get("model")
            or "unknown"
        )
        try:
            dimension = int(provider.get_dim())
        except Exception as exc:
            raise InitializationError("Embedding Provider 返回了无效的向量维度。") from exc
        if dimension <= 0:
            raise InitializationError("Embedding Provider 返回了无效的向量维度。")

        # endpoint 仅参与哈希，不会写入状态文件或日志。
        endpoint = (
            config.get("embedding_api_base")
            or config.get("api_base")
            or config.get("base_url")
            or ""
        )
        return {
            "schema": cls.SCHEMA,
            "provider_class": (
                f"{type(provider).__module__}.{type(provider).__qualname__}"
            ),
            "provider_id": str(config.get("id") or "unknown"),
            "provider_type": str(config.get("type") or "unknown"),
            "model": str(model),
            "dimension": dimension,
            "dimensions_mode": config.get("embedding_dimensions_mode"),
            "input_type": config.get("input_type"),
            "endpoint": str(endpoint).rstrip("/"),
        }

    @classmethod
    def digest(cls, provider: Any) -> str:
        encoded = json.dumps(
            cls.payload(provider),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class EmbeddingIndexBootstrapService:
    """校验 SQLite/FAISS 一致性，并在必要时安全构建影子索引。"""

    STATE_SCHEMA = 1
    DEFAULT_BATCH_SIZE = 16

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        *,
        faiss_bootstrap: FaissBootstrapService | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.state_path = Path(state_path)
        self.faiss_bootstrap = faiss_bootstrap or FaissBootstrapService()
        self.batch_size = max(1, int(batch_size))

    async def prepare(
        self,
        specs: list[EmbeddingIndexSpec],
        provider: Any,
    ) -> list[EmbeddingIndexResult]:
        """依次准备索引；每个索引成功后才提交对应指纹。"""
        provider_fingerprint = EmbeddingFingerprintService.digest(provider)
        dimension = int(provider.get_dim())
        state = self._read_state()
        results: list[EmbeddingIndexResult] = []

        for spec in specs:
            fingerprint = hashlib.sha256(
                f"{provider_fingerprint}:{spec.vector_schema}".encode("ascii")
            ).hexdigest()
            documents = await asyncio.to_thread(
                self._read_documents,
                spec.document_path,
            )
            stored = state.get("indexes", {}).get(spec.key, {}).get("fingerprint")
            consistent, reason = self._validate_existing_index(
                spec.index_path,
                documents,
                dimension,
            )
            fingerprint_changed = bool(stored and stored != fingerprint)

            if consistent and not fingerprint_changed:
                action = "adopted" if stored is None else "kept"
            else:
                rebuild_reason = (
                    "embedding_provider_changed" if fingerprint_changed else reason
                )
                await self._rebuild_shadow_index(
                    spec,
                    documents,
                    provider,
                    dimension,
                )
                action = "rebuilt"
                reason = rebuild_reason

            state.setdefault("indexes", {})[spec.key] = {
                "fingerprint": fingerprint,
                "dimension": dimension,
                "vector_schema": spec.vector_schema,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_state(state)
            results.append(
                EmbeddingIndexResult(
                    key=spec.key,
                    action=action,
                    document_count=len(documents),
                    reason=reason,
                )
            )
            logger.info(
                f"{tag('index')} Embedding 索引准备完成: "
                f"key={spec.key}, action={action}, documents={len(documents)}, "
                f"reason={reason}"
            )

        return results

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema": self.STATE_SCHEMA, "indexes": {}}
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InitializationError(
                "Embedding 索引状态文件损坏；为避免误用未知模型生成的旧向量，"
                "初始化已停止。"
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("schema") != self.STATE_SCHEMA
            or not isinstance(value.get("indexes"), dict)
        ):
            raise InitializationError(
                "Embedding 索引状态文件版本或结构无效，初始化已停止。"
            )
        return value

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_name(f"{self.state_path.name}.tmp")
        try:
            temp_path.write_text(
                json.dumps(state, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_path, self.state_path)
        except OSError as exc:
            raise InitializationError("无法原子写入 Embedding 索引状态文件。") from exc
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _read_documents(path: Path) -> list[tuple[int, str]]:
        if not path.exists():
            return []
        try:
            uri = f"{path.resolve().as_uri()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                table = connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'documents'"
                ).fetchone()
                if table is None:
                    return []
                rows = connection.execute(
                    "SELECT id, text FROM documents ORDER BY id ASC"
                ).fetchall()
        except sqlite3.Error as exc:
            raise InitializationError(
                "无法只读检查向量文档数据库，初始化已停止。"
            ) from exc

        documents: list[tuple[int, str]] = []
        seen_ids: set[int] = set()
        for raw_id, raw_text in rows:
            try:
                document_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise InitializationError("向量文档数据库包含无效的内部 ID。") from exc
            if document_id in seen_ids:
                raise InitializationError("向量文档数据库包含重复的内部 ID。")
            seen_ids.add(document_id)
            documents.append((document_id, str(raw_text or "")))
        return documents

    def _validate_existing_index(
        self,
        index_path: Path,
        documents: list[tuple[int, str]],
        dimension: int,
    ) -> tuple[bool, str]:
        expected_ids = {document_id for document_id, _ in documents}
        if not index_path.exists():
            return (not expected_ids, "missing_index")
        try:
            if index_path.stat().st_size <= 0:
                return False, "empty_index_file"
            index = self.faiss_bootstrap.faiss_read_index_safe(str(index_path))
            if int(getattr(index, "d", 0) or 0) != dimension:
                return False, "dimension_mismatch"
            actual_ids = self._index_ids(index)
            if actual_ids is None:
                return False, "unreadable_index_ids"
            if actual_ids != expected_ids:
                return False, "document_id_mismatch"
            if int(getattr(index, "ntotal", -1)) != len(expected_ids):
                return False, "document_count_mismatch"
        except Exception:
            return False, "unreadable_index"
        return True, "consistent"

    @staticmethod
    def _index_ids(index: Any) -> set[int] | None:
        id_map = getattr(index, "id_map", None)
        if id_map is None:
            return set() if int(getattr(index, "ntotal", 0)) == 0 else None
        try:
            import faiss

            return {int(value) for value in faiss.vector_to_array(id_map)}
        except Exception:
            return None

    async def _rebuild_shadow_index(
        self,
        spec: EmbeddingIndexSpec,
        documents: list[tuple[int, str]],
        provider: Any,
        dimension: int,
    ) -> None:
        try:
            import faiss
            import numpy as np
        except Exception as exc:
            raise InitializationError("缺少可用的 FAISS/Numpy 运行时，无法重建索引。") from exc

        shadow_path = spec.index_path.with_name(f"{spec.index_path.name}.rebuild.tmp")
        writing_path = shadow_path.with_name(f"{shadow_path.name}.writing")
        for stale in (shadow_path, writing_path):
            try:
                stale.unlink(missing_ok=True)
            except OSError as exc:
                raise InitializationError("无法清理旧的向量影子索引。") from exc

        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        expected_ids = {document_id for document_id, _ in documents}
        try:
            for offset in range(0, len(documents), self.batch_size):
                batch = documents[offset : offset + self.batch_size]
                ids = [document_id for document_id, _ in batch]
                texts = [text for _, text in batch]
                vectors = await provider.get_embeddings(texts)
                matrix = np.asarray(vectors, dtype=np.float32)
                if matrix.ndim != 2 or matrix.shape[0] != len(batch):
                    raise InitializationError(
                        "Embedding Provider 返回的向量数量或结构与文档批次不一致。"
                    )
                if matrix.shape[1] != dimension:
                    raise InitializationError(
                        "Embedding Provider 返回的实际维度与配置维度不一致。"
                    )
                index.add_with_ids(matrix, np.asarray(ids, dtype=np.int64))

            if int(getattr(index, "ntotal", -1)) != len(documents):
                raise InitializationError("影子索引向量数量校验失败。")
            if self._index_ids(index) != expected_ids:
                raise InitializationError("影子索引文档 ID 集校验失败。")

            spec.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.faiss_bootstrap.faiss_write_index_safe(index, str(writing_path))
            os.replace(writing_path, shadow_path)
            candidate = self.faiss_bootstrap.faiss_read_index_safe(str(shadow_path))
            if int(getattr(candidate, "d", 0) or 0) != dimension:
                raise InitializationError("落盘后的影子索引维度校验失败。")
            if int(getattr(candidate, "ntotal", -1)) != len(documents):
                raise InitializationError("落盘后的影子索引数量校验失败。")
            if self._index_ids(candidate) != expected_ids:
                raise InitializationError("落盘后的影子索引 ID 集校验失败。")

            os.replace(shadow_path, spec.index_path)
        except asyncio.CancelledError:
            raise
        except InitializationError:
            raise
        except Exception as exc:
            raise InitializationError(
                f"{spec.key} 向量影子索引重建失败；旧索引已保留。"
            ) from exc
        finally:
            for stale in (writing_path, shadow_path):
                try:
                    stale.unlink(missing_ok=True)
                except OSError:
                    pass


__all__ = [
    "EmbeddingFingerprintService",
    "EmbeddingIndexBootstrapService",
    "EmbeddingIndexResult",
    "EmbeddingIndexSpec",
]
