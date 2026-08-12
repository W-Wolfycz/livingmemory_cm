"""
文档路检索器 - 基于向量检索 + 加权 + MMR 去重

历史包袱：原名为"混合检索器"（BM25 + 向量 + RRF 融合），
CM-only 分支已硬编码关闭 BM25，文档路退化为纯向量 + 加权 + MMR。
RRF/BM25 仍被 graph_retriever 内部复用，那里才是真正的双路融合。
"""

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any

from ...log import logger, tag

from ..utils.number_utils import clamp_float, safe_float
from .vector_retriever import VectorResult, VectorRetriever


@dataclass
class HybridResult:
    """文档路检索结果（类名沿用 HybridResult 以减少下游引用改动）"""

    doc_id: int
    final_score: float  # 加权后的最终分数
    vector_score: float | None  # 向量原始分数
    content: str
    metadata: dict[str, Any]
    score_breakdown: dict[str, float] | None = None  # 各维度分数明细


class HybridRetriever:
    """
    文档路检索器（纯向量）

    CM-only 分支下文档路只走向量检索 + 加权 + MMR 多样性去重，
    不再涉及 BM25 / RRF 融合（图路内部仍保留双路）。
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        config: dict[str, Any] | None = None,
    ):
        """
        Args:
            vector_retriever: 向量检索器实例
            config: 配置字典,支持以下参数:
                - decay_rate: 时间衰减率,默认0.01
        """
        self.vector_retriever = vector_retriever
        self.config = config or {}

        self.decay_rate = self.config.get("decay_rate", 0.01)

        # 加权求和各维度权重（可通过配置覆盖）
        self.score_alpha = self.config.get("score_alpha", 0.5)  # 检索相关性
        self.score_beta = self.config.get("score_beta", 0.25)  # 重要性
        self.score_gamma = self.config.get("score_gamma", 0.25)  # 时间新鲜度

        # MMR 多样性参数
        self.mmr_lambda = self.config.get("mmr_lambda", 0.7)  # 相关性 vs 多样性权衡

    async def add_memory(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> int:
        """添加记忆到向量索引"""
        metadata = metadata or {}

        if "importance" not in metadata:
            metadata["importance"] = 0.5
        if "create_time" not in metadata:
            metadata["create_time"] = time.time()
        if "last_access_time" not in metadata:
            metadata["last_access_time"] = time.time()
        if "session_id" not in metadata:
            metadata["session_id"] = None
        if "persona_id" not in metadata:
            metadata["persona_id"] = None

        doc_id = await self.vector_retriever.add_document(content, metadata)
        return doc_id

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[HybridResult]:
        """执行向量检索 + 加权 + MMR 去重"""
        if not query or not query.strip():
            return []

        vector_results, vector_error = await self._search_route(
            "向量", self.vector_retriever.search(query, k, session_id, persona_id)
        )
        if vector_error or not vector_results:
            return []

        current_time = time.time()
        weighted_results = await asyncio.to_thread(
            self._apply_weighting, vector_results, current_time
        )

        if len(weighted_results) > 1:
            weighted_results = await asyncio.to_thread(
                self._apply_mmr, weighted_results, k
            )

        return weighted_results

    async def _search_route(
        self, route_name: str, search_coro
    ) -> tuple[list, Exception | None]:
        """Run one retrieval route and convert ordinary failures into route errors."""
        try:
            return await search_coro, None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"{tag('retrieval')} {route_name}检索异常: {e}", exc_info=True)
            return [], e

    def _apply_weighting(
        self, vector_results: list[VectorResult], current_time: float
    ) -> list[HybridResult]:
        """
        应用重要性和时间衰减加权

        使用加权求和（而非乘法）避免任何单一维度低分导致整体清零。
        时间衰减基于 max(create_time, last_access_time)，高频访问记忆衰减更慢。
        """
        if not vector_results:
            return []

        # 先归一化向量分数到 [0, 1]
        max_vector = max(r.score for r in vector_results)
        if max_vector <= 0:
            max_vector = 1.0

        hybrid_results = []

        for result in vector_results:
            # 安全解析 metadata，确保它是字典类型
            metadata = result.metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                    logger.debug(
                        f"{tag('retrieval')} [hybrid_retriever] 将字符串metadata转换为字典: doc_id={result.doc_id}"
                    )
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"{tag('retrieval')} [hybrid_retriever] 解析metadata JSON失败: {e}, doc_id={result.doc_id}, "
                        f"metadata类型={type(metadata)}, 使用空字典"
                    )
                    metadata = {}
            elif metadata is None:
                logger.debug(
                    f"{tag('retrieval')} [hybrid_retriever] metadata为None, doc_id={result.doc_id}, 使用空字典"
                )
                metadata = {}
            elif not isinstance(metadata, dict):
                logger.warning(
                    f"{tag('retrieval')} [hybrid_retriever] metadata类型不支持: {type(metadata)}, doc_id={result.doc_id}, "
                    f"使用空字典"
                )
                metadata = {}

            importance = clamp_float(metadata.get("importance"), default=0.5)

            # 时间衰减：取 create_time 与 last_access_time 的较大值
            # 高频访问的记忆衰减更慢，符合"记忆强化"认知规律
            create_time = safe_float(metadata.get("create_time"), current_time)
            last_access_time = safe_float(metadata.get("last_access_time"), 0.0)
            reference_time = max(create_time, last_access_time)
            days_old = max(0.0, (current_time - reference_time) / 86400)
            recency_weight = math.exp(-self.decay_rate * days_old)

            vector_normalized = result.score / max_vector

            # 加权求和：各维度互补而非互斥
            final_score = (
                self.score_alpha * vector_normalized
                + self.score_beta * importance
                + self.score_gamma * recency_weight
            )

            score_breakdown = {
                "vector_normalized": round(vector_normalized, 4),
                "importance": round(importance, 4),
                "recency_weight": round(recency_weight, 4),
                "days_old": round(days_old, 2),
                "final_score": round(final_score, 4),
            }

            hybrid_results.append(
                HybridResult(
                    doc_id=result.doc_id,
                    final_score=final_score,
                    vector_score=result.score,
                    content=result.content,
                    metadata=metadata,
                    score_breakdown=score_breakdown,
                )
            )

        hybrid_results.sort(key=lambda x: x.final_score, reverse=True)

        return hybrid_results

    def _apply_mmr(self, results: list[HybridResult], k: int) -> list[HybridResult]:
        """
        最大边际相关性（MMR）去重，避免多条语义重复的记忆占据 Top-K。

        使用内容词袋相似度作为轻量代理（无需额外向量计算）。
        mmr_lambda 越高越偏向相关性，越低越偏向多样性。
        """
        if len(results) <= k:
            return results

        def _token_set(text: str) -> set[str]:
            tokens = set(text.lower().split())
            return tokens if tokens else {"<empty>"}

        selected: list[HybridResult] = []
        candidates = list(results)

        while candidates and len(selected) < k:
            if not selected:
                # 第一条直接选最高分
                selected.append(candidates.pop(0))
                continue

            best_idx = -1
            best_mmr = -1.0
            selected_tokens = [_token_set(s.content) for s in selected]

            for i, cand in enumerate(candidates):
                cand_tokens = _token_set(cand.content)
                # 与已选结果的最大 Jaccard 相似度
                max_sim = max(
                    len(cand_tokens & st) / max(len(cand_tokens | st), 1)
                    for st in selected_tokens
                )
                mmr_score = (
                    self.mmr_lambda * cand.final_score - (1 - self.mmr_lambda) * max_sim
                )
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(candidates.pop(best_idx))
            else:
                break

        return selected

    async def update_metadata(self, doc_id: int, metadata: dict[str, Any]) -> bool:
        """同步更新 FAISS 向量库中的元数据"""
        try:
            vector_success = await self.vector_retriever.update_metadata(
                doc_id, metadata
            )

            if not vector_success:
                logger.error(f"{tag('retrieval')} [同步更新] FAISS更新失败 (doc_id={doc_id})")
                return False

            logger.debug(f"{tag('retrieval')} [同步更新] 元数据更新成功 (doc_id={doc_id})")
            return True

        except Exception as e:
            logger.error(f"{tag('retrieval')} [同步更新] 失败 (doc_id={doc_id}): {e}", exc_info=True)
            return False

    async def delete_memory(self, doc_id: int) -> bool:
        """从向量库删除记忆"""
        try:
            vector_deleted = await self.vector_retriever.delete_document(doc_id)
            if not vector_deleted:
                logger.error(f"{tag('retrieval')} [删除] 向量库删除失败 (doc_id={doc_id})")
                return False
            logger.debug(f"{tag('retrieval')} [删除] 向量库已删除 (doc_id={doc_id})")
            logger.debug(f"{tag('retrieval')} [删除] 记忆删除成功 (doc_id={doc_id})")
            return True
        except Exception as e:
            logger.error(f"{tag('retrieval')} [删除] 删除记忆失败 (doc_id={doc_id}): {e}", exc_info=True)
            return False
