"""长期记忆检索编排与短期结果缓存。"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from ...log import logger, tag
from ..retrieval.hybrid_retriever import HybridResult
from ..utils.number_utils import clamp_float

# documents 表 metadata 的规范表达式：必须与 memory_schema_service 里
# idx_doc_status_created / idx_doc_type_created 的索引表达式同形才会被命中。
# json_valid 守卫同时避免坏 JSON 行让整条查询直接报错。
_DOC_STATUS_EXPR = (
    "COALESCE(CASE WHEN json_valid(metadata) "
    "THEN json_extract(metadata, '$.status') END, 'active')"
)
_DOC_CREATED_EXPR = (
    "COALESCE(CASE WHEN json_valid(metadata) "
    "THEN CAST(json_extract(metadata, '$.create_time') AS REAL) END, 0)"
)
_DOC_SESSION_EXPR = (
    "CASE WHEN json_valid(metadata) "
    "THEN json_extract(metadata, '$.session_id') END"
)
_DOC_PERSONA_EXPR = (
    "CASE WHEN json_valid(metadata) "
    "THEN json_extract(metadata, '$.persona_id') END"
)


class MemorySearchService:
    """协调文档/图双路检索，并维护进程内 TTL 缓存。"""

    def __init__(self, config: dict[str, Any], rerank_provider=None) -> None:
        self.config = config
        # Rerank 提供商（可选）：启用后对合并结果重排序，失败自动沿用原排序
        self.rerank_provider = rerank_provider
        self._rerank_enabled = bool(config.get("rerank_enabled", False)) and (
            rerank_provider is not None
        )
        self._rerank_candidates = max(2, int(config.get("rerank_candidates", 20)))
        self._rerank_weight = min(
            1.0, max(0.0, float(config.get("rerank_weight", 0.6)))
        )
        self._rerank_timeout = float(config.get("search_timeout_seconds", 5.0) or 0.0)
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
        self_id: str | None = None,
        exclude_personas: set[str] | None = None,
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

        excluded = {str(item) for item in (exclude_personas or set()) if item}
        cache_key = self._cache_key(
            query,
            k,
            session_id,
            persona_id,
            dual_route_enabled=dual_route_retriever is not None,
            self_id=self_id,
            excluded_personas=excluded,
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
        results = self._filter_legacy_bot_scope(results, self_id)
        results = self._filter_frozen_personas(results, excluded)
        # 启用重排时先保留更大的候选池，重排后再截断到 k——否则候选在重排前
        # 就已经被裁掉，重排只能在同一批里换顺序。
        candidate_k = (
            max(k, self._rerank_candidates) if self._rerank_enabled else k
        )
        results = await self._merge_recent_memories(
            results,
            candidate_k,
            session_id,
            persona_id,
            db_connection,
            excluded_personas=excluded,
        )
        # 近期槽位直接查库、不经过向量过滤，而它的 SQL 只按 session/persona
        # 过滤，因此冻结与旧 Bot 归属两道过滤都要再兜一次，确保两者都不会
        # 从该路径泄漏回注入结果。
        results = self._filter_frozen_personas(results, excluded)
        results = self._filter_legacy_bot_scope(results, self_id)
        results = await self._apply_rerank(query, results)
        results = results[:k]

        for result in results:
            schedule_task(update_access_time(result.doc_id))

        self._set_cached_results(cache_key, results)
        return results

    @staticmethod
    def _filter_frozen_personas(
        results: list[Any],
        excluded: set[str],
    ) -> list[Any]:
        """冷存储隔离：冻结中的 persona 不参与召回。"""
        if not excluded:
            return results
        filtered: list[Any] = []
        for result in results:
            metadata = getattr(result, "metadata", None)
            persona = (
                str(metadata.get("persona_id") or "")
                if isinstance(metadata, dict)
                else ""
            )
            if persona and persona in excluded:
                continue
            filtered.append(result)
        return filtered

    @staticmethod
    def _filter_legacy_bot_scope(
        results: list[Any],
        self_id: str | None,
    ) -> list[Any]:
        """过渡兼容：按旧记忆里记录的 ``metadata.self_id`` 过滤其他 Bot 实例的记忆。

        ``2.5.7-cm.3`` 起写入侧不再记录 ``self_id``（platform_id 会随换号、改适配器名
        而变，不适合当隔离键），但**在过渡期保留读取侧过滤**：只有 cm.3 之前写入、
        确实带 ``self_id`` 的老记录会被过滤，没有该字段的新记录一律保留。因此随着
        旧记忆衰减/清理，这层过滤会自然变成空操作，记忆归属收敛到 session + persona
        两个稳定维度。这是兼容逻辑而非用户开关，不需要配置项。
        """
        if not self_id:
            return results
        current = str(self_id)
        filtered: list[Any] = []
        for result in results:
            metadata = getattr(result, "metadata", None)
            owner = (
                str(metadata.get("self_id") or "")
                if isinstance(metadata, dict)
                else ""
            )
            if owner and owner != current:
                continue
            filtered.append(result)
        dropped = len(results) - len(filtered)
        if dropped:
            logger.debug(
                f"{tag('retrieval')} 旧记忆 self_id 过渡过滤: 候选 {len(results)} 条，"
                f"剔除其他 Bot 实例写入的旧记忆 {dropped} 条"
            )
        return filtered

    async def _apply_rerank(
        self,
        query: str,
        results: list[HybridResult],
    ) -> list[HybridResult]:
        """用 AstrBot 的 Rerank 提供商对候选重排序。

        语义：候选集内把重排分数归一化到 [0,1]，与原有 final_score 按
        ``rerank_weight`` 线性混合后重新排序，明细写入 ``score_breakdown["rerank"]``。
        提供商不可用、调用超时、抛异常或返回空时**原样返回**，召回不受影响；
        超出候选上限的部分保持在尾部（它们没有参与重排）。
        """
        if not self._rerank_enabled or len(results) < 2:
            return results

        head = results[: self._rerank_candidates]
        tail = results[len(head) :]
        documents = [item.content or "" for item in head]

        try:
            rerank_call = self.rerank_provider.rerank(query, documents)
            if self._rerank_timeout > 0:
                scored = await asyncio.wait_for(
                    rerank_call, timeout=self._rerank_timeout
                )
            else:
                scored = await rerank_call
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"{tag('retrieval')} Rerank 调用失败，沿用原排序: {exc}")
            return results

        raw_scores: dict[int, float] = {}
        for item in scored or []:
            try:
                index = int(getattr(item, "index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(head):
                raw_scores[index] = float(
                    getattr(item, "relevance_score", 0.0) or 0.0
                )
        if not raw_scores:
            logger.debug(f"{tag('retrieval')} Rerank 未返回可用分数，沿用原排序")
            return results

        highest = max(raw_scores.values())
        lowest = min(raw_scores.values())
        span = highest - lowest

        weight = self._rerank_weight
        for index, item in enumerate(head):
            if index not in raw_scores:
                continue  # 未被打分的候选保留原分数
            normalized = 1.0 if span <= 0 else (raw_scores[index] - lowest) / span
            blended = weight * normalized + (1.0 - weight) * item.final_score
            breakdown = dict(item.score_breakdown or {})
            breakdown["rerank"] = round(normalized, 4)
            breakdown["final_score"] = round(blended, 4)
            item.score_breakdown = breakdown
            item.final_score = blended

        head.sort(key=lambda item: item.final_score, reverse=True)
        logger.debug(
            f"{tag('retrieval')} Rerank 完成: 候选 {len(head)} 条，"
            f"权重 {weight:.2f}，最高/最低分 {highest:.4f}/{lowest:.4f}"
        )
        return head + tail

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
        self_id: str | None = None,
        excluded_personas: set[str] | None = None,
    ) -> tuple[Any, ...]:
        return (
            self._cache_generation,
            self._normalize_query(query),
            int(k),
            session_id or "",
            persona_id or "",
            dual_route_enabled,
            self_id or "",
            tuple(sorted(str(item) for item in (excluded_personas or set()))),
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
        excluded_personas: set[str] | None = None,
        self_id: str | None = None,
    ) -> list[HybridResult]:
        if count <= 0 or db_connection is None:
            return []

        conditions = [f"{_DOC_STATUS_EXPR} = 'active'"]
        params: list[Any] = []
        if session_id is not None:
            conditions.append(f"{_DOC_SESSION_EXPR} = ?")
            params.append(session_id)
        if persona_id is not None:
            conditions.append(f"{_DOC_PERSONA_EXPR} = ?")
            params.append(persona_id)
        max_age_hours = max(
            0,
            int(self.config.get("recent_memory_max_age_hours", 72)),
        )
        if max_age_hours > 0:
            conditions.append(f"{_DOC_CREATED_EXPR} >= ?")
            params.append(time.time() - max_age_hours * 3600)
        params.append(max(count * 3, count))

        cursor = await db_connection.execute(
            "SELECT id, text, metadata FROM documents WHERE "
            + " AND ".join(conditions)
            + f" ORDER BY {_DOC_CREATED_EXPR} DESC, id DESC LIMIT ?",
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
        # 过滤必须发生在 [:count] 截断之前：截断后再过滤会让槽位空着，
        # 最终返回条数少于 k（该路径直查数据库，不经过向量过滤）。
        recent = self._filter_by_retrieval_policy(recent)
        recent = self._filter_frozen_personas(recent, excluded_personas or set())
        recent = self._filter_legacy_bot_scope(recent, self_id)
        return recent[:count]

    async def _merge_recent_memories(
        self,
        results: list[HybridResult],
        k: int,
        session_id: str | None,
        persona_id: str | None,
        db_connection,
        excluded_personas: set[str] | None = None,
        self_id: str | None = None,
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
            excluded_personas=excluded_personas,
            self_id=self_id,
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
