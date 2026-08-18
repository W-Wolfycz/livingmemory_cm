"""长期记忆检索编排与短期结果缓存。"""

from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from ..retrieval.hybrid_retriever import HybridResult
from ..utils.number_utils import clamp_float


class MemorySearchService:
    """协调文档/图双路检索，并维护进程内 TTL 缓存。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._cache_ttl = float(config.get("search_cache_ttl_seconds", 45.0))
        self._cache_max_size = int(config.get("search_cache_max_size", 256))
        self._cache_generation = 0
        self._cache: OrderedDict[
            tuple[Any, ...], tuple[float, list[HybridResult]]
        ] = OrderedDict()

    async def search(
        self,
        *,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
        hybrid_retriever,
        dual_route_retriever,
        schedule_task: Callable[[Awaitable[Any]], None],
        update_access_time: Callable[[int], Awaitable[bool]],
        migrate_session: Callable[[str], Awaitable[None]],
        db_connection=None,
    ) -> list[HybridResult]:
        """执行检索，命中后异步更新访问时间。"""
        if not query or not query.strip():
            return []

        cache_key = self._cache_key(
            query,
            k,
            session_id,
            persona_id,
            dual_route_enabled=dual_route_retriever is not None,
        )
        cached_results = self._get_cached_results(cache_key)
        if cached_results is not None:
            for result in cached_results:
                schedule_task(update_access_time(result.doc_id))
            return cached_results

        if session_id and ":" in session_id:
            schedule_task(migrate_session(session_id))

        if dual_route_retriever is not None:
            results = await dual_route_retriever.search(
                query,
                k,
                session_id,
                persona_id,
            )
        else:
            if hybrid_retriever is None:
                raise RuntimeError("混合检索器未初始化")
            results = await hybrid_retriever.search(
                query,
                k,
                session_id,
                persona_id,
            )

        results = self._filter_by_retrieval_policy(results)
        results = await self._merge_recent_memories(
            results,
            k,
            session_id,
            persona_id,
            db_connection,
        )

        for result in results:
            schedule_task(update_access_time(result.doc_id))

        self._set_cached_results(cache_key, results)
        return results

    def invalidate(self) -> None:
        """写操作后使全部检索缓存失效。"""
        self._cache_generation += 1
        self._cache.clear()

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.casefold().split())

    def _cache_key(
        self,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
        *,
        dual_route_enabled: bool,
    ) -> tuple[Any, ...]:
        return (
            self._cache_generation,
            self._normalize_query(query),
            int(k),
            session_id or "",
            persona_id or "",
            dual_route_enabled,
            round(
                float(
                    self.config.get(
                        "document_route_weight",
                        1.0 - float(self.config.get("graph_route_weight", 0.35)),
                    )
                ),
                4,
            ),
            round(float(self.config.get("graph_route_weight", 0.35)), 4),
            int(self.config.get("graph_expansion_hops", 1)),
            round(
                float(self.config.get("min_importance_for_retrieval", 0.0)),
                4,
            ),
            round(
                float(self.config.get("min_similarity_for_retrieval", 0.0)),
                4,
            ),
            int(self.config.get("recent_memory_count", 0)),
            int(self.config.get("recent_memory_max_age_hours", 72)),
            str(self.config.get("memory_type_filter", "all")),
        )

    def _filter_by_retrieval_policy(
        self,
        results: list[HybridResult],
    ) -> list[HybridResult]:
        """Apply scope-safe, default-compatible final retrieval policies."""
        importance_threshold = clamp_float(
            self.config.get("min_importance_for_retrieval"),
            default=0.0,
        )
        similarity_threshold = clamp_float(
            self.config.get("min_similarity_for_retrieval"),
            default=0.0,
        )
        event_only = self.config.get("memory_type_filter", "all") == "event_only"
        event_atom_types = {"episodic", "planned", "factual"}
        filtered: list[HybridResult] = []

        for result in results:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            if str(metadata.get("status") or "active") != "active":
                continue
            if (
                importance_threshold > 0
                and clamp_float(metadata.get("importance"), default=0.5)
                < importance_threshold
            ):
                continue

            vector_signals: list[float] = []
            if result.vector_score is not None:
                vector_signals.append(
                    clamp_float(result.vector_score, default=0.0)
                )
            breakdown = result.score_breakdown
            if isinstance(breakdown, dict):
                for key in ("document_vector_score", "graph_vector_score"):
                    if key in breakdown:
                        vector_signals.append(
                            clamp_float(breakdown[key], default=0.0)
                        )
            if (
                similarity_threshold > 0
                and vector_signals
                and max(vector_signals) < similarity_threshold
            ):
                continue

            atom_types = metadata.get("atom_types")
            if event_only and isinstance(atom_types, list) and atom_types:
                normalized_types = {str(value).casefold() for value in atom_types}
                if normalized_types.isdisjoint(event_atom_types):
                    continue
            filtered.append(result)
        return filtered

    @staticmethod
    def _safe_json_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    async def _get_recent_memory_results(
        self,
        count: int,
        session_id: str | None,
        persona_id: str | None,
        db_connection,
    ) -> list[HybridResult]:
        if count <= 0 or db_connection is None:
            return []

        conditions = [
            "COALESCE(json_extract(metadata, '$.status'), 'active') = 'active'"
        ]
        params: list[Any] = []
        if session_id is not None:
            conditions.append("json_extract(metadata, '$.session_id') = ?")
            params.append(session_id)
        if persona_id is not None:
            conditions.append("json_extract(metadata, '$.persona_id') = ?")
            params.append(persona_id)
        max_age_hours = max(
            0,
            int(self.config.get("recent_memory_max_age_hours", 72)),
        )
        if max_age_hours > 0:
            conditions.append(
                "CAST(json_extract(metadata, '$.create_time') AS REAL) >= ?"
            )
            params.append(time.time() - max_age_hours * 3600)
        params.append(max(count * 3, count))

        cursor = await db_connection.execute(
            "SELECT id, text, metadata FROM documents WHERE "
            + " AND ".join(conditions)
            + " ORDER BY CAST(json_extract(metadata, '$.create_time') AS REAL) "
            "DESC, id DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        recent = [
            HybridResult(
                doc_id=int(row[0]),
                final_score=1.0,
                vector_score=None,
                content=str(row[1] or ""),
                metadata=self._safe_json_dict(row[2]),
                score_breakdown={"recent_memory": 1.0},
            )
            for row in rows
        ]
        return self._filter_by_retrieval_policy(recent)[:count]

    async def _merge_recent_memories(
        self,
        results: list[HybridResult],
        k: int,
        session_id: str | None,
        persona_id: str | None,
        db_connection,
    ) -> list[HybridResult]:
        recent_count = min(
            max(0, int(self.config.get("recent_memory_count", 0))),
            k,
        )
        if recent_count <= 0:
            return results[:k]

        recent = await self._get_recent_memory_results(
            recent_count,
            session_id,
            persona_id,
            db_connection,
        )
        if not recent:
            return results[:k]

        selected = list(results[: max(0, k - recent_count)])
        selected_ids = {result.doc_id for result in selected}
        for result in recent:
            if result.doc_id not in selected_ids:
                selected.append(result)
                selected_ids.add(result.doc_id)
        for result in results:
            if len(selected) >= k:
                break
            if result.doc_id not in selected_ids:
                selected.append(result)
                selected_ids.add(result.doc_id)
        return selected[:k]

    def _get_cached_results(
        self,
        cache_key: tuple[Any, ...],
    ) -> list[HybridResult] | None:
        if self._cache_ttl <= 0 or self._cache_max_size <= 0:
            return None

        cached = self._cache.get(cache_key)
        if cached is None:
            return None

        cached_at, results = cached
        if time.time() - cached_at > self._cache_ttl:
            self._cache.pop(cache_key, None)
            return None

        self._cache.move_to_end(cache_key)
        return copy.deepcopy(results)

    def _set_cached_results(
        self,
        cache_key: tuple[Any, ...],
        results: list[HybridResult],
    ) -> None:
        if self._cache_ttl <= 0 or self._cache_max_size <= 0:
            return

        self._cache[cache_key] = (time.time(), copy.deepcopy(results))
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)
