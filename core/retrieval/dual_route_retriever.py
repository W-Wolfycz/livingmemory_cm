"""Fuse document-route and graph-route retrieval into one result list."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .graph_retriever import GraphRetriever
from .hybrid_retriever import HybridResult, HybridRetriever


def _has_payload(result: HybridResult | None) -> bool:
    """候选自身已带正文与元数据（无需回库补取）时返回 True。"""
    if result is None:
        return False
    return bool(result.content) and isinstance(result.metadata, dict) and bool(
        result.metadata
    )


class DualRouteRetriever:
    """Coordinate document and graph retrieval routes."""

    def __init__(
        self,
        document_retriever: HybridRetriever,
        graph_retriever: GraphRetriever,
        memory_loader: Callable[[int], Awaitable[dict[str, Any] | None]],
        config: dict[str, Any] | None = None,
        memory_batch_loader: (
            Callable[[list[int]], Awaitable[dict[int, dict[str, Any]]]] | None
        ) = None,
    ):
        self.document_retriever = document_retriever
        self.graph_retriever = graph_retriever
        self.memory_loader = memory_loader
        self.memory_batch_loader = memory_batch_loader
        self.config = config or {}
        self.graph_route_weight = float(self.config.get("graph_route_weight", 0.35))
        # 文档路权重：优先使用显式旧键，否则自动 = 1 - 图路权重
        self.document_route_weight = float(
            self.config.get("document_route_weight", 1.0 - self.graph_route_weight)
        )
        self.cross_route_bonus = float(self.config.get("cross_route_bonus", 0.08))
        self.dynamic_route_weighting = bool(
            self.config.get("dynamic_route_weighting", True)
        )

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[HybridResult]:
        """Run both retrieval routes and merge their memory candidates."""
        doc_results, graph_results = await asyncio.gather(
            self.document_retriever.search(
                query, max(k * 2, k), session_id, persona_id
            ),
            self.graph_retriever.search(query, max(k * 2, k), session_id, persona_id),
        )

        if not graph_results:
            return doc_results[:k]
        if not doc_results and not graph_results:
            return []

        document_weight, graph_weight, intent = self._route_weights_for_query(query)

        document_max = (
            max((item.final_score for item in doc_results), default=1.0) or 1.0
        )
        graph_max = (
            max((item.final_score for item in graph_results), default=1.0) or 1.0
        )

        doc_map = {item.doc_id: item for item in doc_results}
        graph_map = {item.doc_id: item for item in graph_results}
        all_doc_ids = set(doc_map) | set(graph_map)

        # 缺内容/元数据的候选先集中补取一次，避免逐个 id 往返（每个 id 一次
        # 查询在候选多时会把召回延迟按 id 数线性拉长）。
        # 判定口径必须与下面合并时的取值口径一致——合并只从 doc_result 取正文
        # 与元数据，因此"图路独有"的候选一律要回库补取，否则会被 continue 丢掉。
        pending_ids = [
            doc_id
            for doc_id in all_doc_ids
            if not _has_payload(doc_map.get(doc_id))
        ]
        loaded = await self._load_memories(pending_ids)

        merged_results: list[HybridResult] = []
        for doc_id in all_doc_ids:
            doc_result = doc_map.get(doc_id)
            graph_result = graph_map.get(doc_id)

            doc_signal = (
                doc_result.final_score / document_max if doc_result is not None else 0.0
            )
            graph_signal = (
                graph_result.final_score / graph_max
                if graph_result is not None
                else 0.0
            )
            route_bonus = (
                self.cross_route_bonus
                if doc_result is not None and graph_result is not None
                else 0.0
            )

            memory_content = doc_result.content if doc_result is not None else ""
            memory_metadata = (
                dict(doc_result.metadata)
                if doc_result is not None and isinstance(doc_result.metadata, dict)
                else {}
            )

            if not memory_content or not memory_metadata:
                memory = loaded.get(doc_id)
                if memory is None and not self.memory_batch_loader:
                    # 无批量通道时退回逐条补取（保持旧行为）
                    memory = await self.memory_loader(doc_id)
                if not memory:
                    continue
                memory_content = str(memory.get("text") or memory_content)
                raw_metadata = memory.get("metadata") or memory_metadata
                memory_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

            final_score = min(
                1.0,
                document_weight * doc_signal
                + graph_weight * graph_signal
                + route_bonus,
            )

            score_breakdown: dict[str, float] = {}
            if doc_result and doc_result.score_breakdown:
                score_breakdown.update(doc_result.score_breakdown)
            if graph_result and graph_result.score_breakdown:
                score_breakdown.update(graph_result.score_breakdown)
            score_breakdown.update(
                {
                    "document_route_score": round(doc_signal, 4),
                    "graph_route_score": round(graph_signal, 4),
                    "document_route_weight": round(document_weight, 4),
                    "graph_route_weight": round(graph_weight, 4),
                    "cross_route_bonus": round(route_bonus, 4),
                    "dual_route_final_score": round(final_score, 4),
                }
            )
            if intent:
                score_breakdown["query_intent"] = intent
            if doc_result is not None:
                score_breakdown["document_vector_score"] = round(
                    float(doc_result.vector_score or 0.0),
                    4,
                )
            if graph_result is not None:
                score_breakdown["graph_keyword_score"] = round(
                    float(graph_result.keyword_score or 0.0),
                    4,
                )
                score_breakdown["graph_vector_score"] = round(
                    float(graph_result.vector_score or 0.0),
                    4,
                )

            merged_results.append(
                HybridResult(
                    doc_id=doc_id,
                    final_score=final_score,
                    vector_score=(
                        doc_result.vector_score if doc_result is not None else None
                    ),
                    content=memory_content,
                    metadata=memory_metadata,
                    score_breakdown=score_breakdown,
                )
            )

        merged_results.sort(key=lambda item: item.final_score, reverse=True)
        return merged_results[:k]

    async def _load_memories(
        self, doc_ids: list[int]
    ) -> dict[int, dict[str, Any]]:
        """批量补齐候选的正文/元数据；无批量通道或批量失败时退回逐条。"""
        if not doc_ids:
            return {}
        if self.memory_batch_loader is not None:
            try:
                loaded = await self.memory_batch_loader(doc_ids)
                if loaded:
                    return loaded
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    f"{tag('retrieval')} 批量补取候选记忆失败，退回逐条补取",
                    exc_info=True,
                )
        fallback: dict[int, dict[str, Any]] = {}
        for doc_id in doc_ids:
            try:
                memory = await self.memory_loader(doc_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    f"{tag('retrieval')} 补取候选记忆失败: {doc_id}",
                    exc_info=True,
                )
                continue
            if memory:
                fallback[doc_id] = memory
        return fallback

    def _route_weights_for_query(self, query: str) -> tuple[float, float, str]:
        """Adjust document/graph weights with lightweight query intent rules."""
        base_document = self.document_route_weight
        base_graph = self.graph_route_weight
        if not self.dynamic_route_weighting:
            return base_document, base_graph, "fixed"

        normalized = query.casefold()
        relation_terms = (
            "谁",
            "和谁",
            "关系",
            "认识",
            "朋友",
            "同事",
            "家人",
            "父母",
            "妈妈",
            "爸爸",
            "老师",
            "同学",
            "partner",
            "friend",
            "relationship",
            "with whom",
        )
        temporal_terms = (
            "上次",
            "昨天",
            "前天",
            "刚才",
            "之前",
            "什么时候",
            "哪天",
            "最近",
            "last time",
            "yesterday",
            "recently",
            "when",
        )
        factual_terms = (
            "是什么",
            "什么是",
            "解释",
            "定义",
            "怎么",
            "如何",
            "why",
            "what is",
            "explain",
            "define",
            "how to",
        )

        relation_hit = any(term in normalized for term in relation_terms)
        temporal_hit = any(term in normalized for term in temporal_terms)
        factual_hit = any(term in normalized for term in factual_terms)

        document_weight = base_document
        graph_weight = base_graph
        intent = "default"

        if relation_hit:
            graph_weight += 0.2
            document_weight -= 0.2
            intent = "relationship"
        if temporal_hit:
            graph_weight += 0.1
            document_weight -= 0.1
            intent = "temporal" if intent == "default" else f"{intent}+temporal"
        if factual_hit and not relation_hit:
            document_weight += 0.15
            graph_weight -= 0.15
            intent = "factual" if intent == "default" else f"{intent}+factual"

        document_weight = max(0.15, min(0.9, document_weight))
        graph_weight = max(0.1, min(0.85, graph_weight))
        total = document_weight + graph_weight
        if total <= 0:
            return base_document, base_graph, "fixed"
        return document_weight / total, graph_weight / total, intent


__all__ = ["DualRouteRetriever"]
