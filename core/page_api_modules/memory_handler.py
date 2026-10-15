"""
记忆管理处理模块
"""

import json
import time
from typing import TYPE_CHECKING, Any

import aiosqlite
from quart import request

from ...log import logger, tag

if TYPE_CHECKING:
    from .utils import PageApiUtils


class MemoryHandler:
    """记忆管理处理器"""

    # 导出：默认 2 万条、上限 10 万条（limit<=0 视为取上限），避免一次导出
    # 把整库读进内存；触到上限时文件名带 _partial，页面据此提示用户。
    EXPORT_DEFAULT_LIMIT = 20000
    EXPORT_MAX_LIMIT = 100000
    EXPORT_FORMATS = ("json", "csv")
    # 以这些字符开头的单元格会被表格软件当公式执行，导出时加前导单引号切断
    _CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

    def __init__(self, utils: "PageApiUtils"):
        """
        初始化记忆管理处理器

        Args:
            utils: PageApiUtils 工具实例
        """
        self.utils = utils

    @staticmethod
    def _normalize_importance_update(value: Any, value_scale: str = "auto") -> float:
        """Normalize WebUI/API importance input into the stored 0-1 scale."""
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("重要性必须是数字") from exc

        scale = str(value_scale or "auto").strip().lower()
        if scale in {"display", "0-10", "ten"}:
            if not 0.0 <= parsed <= 10.0:
                raise ValueError("重要性必须在 0-10 范围内")
            return parsed / 10.0

        if scale in {"stored", "normalized", "0-1"}:
            if not 0.0 <= parsed <= 1.0:
                raise ValueError("重要性必须在 0-1 范围内")
            return parsed

        if 0.0 <= parsed <= 1.0:
            return parsed
        if 0.0 <= parsed <= 10.0:
            return parsed / 10.0

        raise ValueError("重要性必须在 0-1 或 0-10 范围内")

    async def list_memories(self, memory_engine) -> dict[str, Any]:
        """
        获取记忆列表（带分页和过滤）

        查询参数:
            - session_id: 会话ID过滤
            - keyword: 关键词搜索（支持ID或文本）
            - status: 状态过滤（all/active/archived）
            - type: 记忆类型过滤（all/GENERAL/FACT/PREFERENCE/...）
            - sort: 排序方式（created_desc/created_asc/updated_desc/...）
            - page: 页码（默认1）
            - page_size: 每页数量（默认20，最大500）

        Returns:
            包含记忆列表和分页信息的字典
        """
        query = request.args
        spec, error = self._build_query_spec(query)
        if error:
            return self.utils.error(error)

        try:
            page = max(1, int(query.get("page", 1)))
            page_size = min(500, max(1, int(query.get("page_size", 20))))
        except (TypeError, ValueError):
            return self.utils.error("分页参数无效")

        db_path = getattr(memory_engine, "db_path", None)
        if not db_path:
            return self.utils.error("MemoryEngine db_path unavailable")

        where_clause = spec["where_clause"]
        params = list(spec["params"])
        sort_expr = spec["sort_expr"]
        sort_key = spec["sort_key"]
        offset = (page - 1) * page_size

        try:
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row

                count_cursor = await db.execute(
                    f"SELECT COUNT(*) AS total FROM documents {where_clause}",
                    params,
                )
                count_row = await count_cursor.fetchone()
                total = int(count_row["total"]) if count_row else 0

                cursor = await db.execute(
                    f"""
                    SELECT id, doc_id, text, metadata, created_at, updated_at
                    FROM documents
                    {where_clause}
                    ORDER BY {sort_expr}
                    LIMIT ? OFFSET ?
                    """,
                    (*params, page_size, offset),
                )
                rows = await cursor.fetchall()
        except Exception as exc:
            logger.error(f"{tag('page')} 获取记忆列表失败: {exc}", exc_info=True)
            return self.utils.error(str(exc))

        items: list[dict[str, Any]] = []
        for row in rows:
            items.append(
                {
                    "id": row["id"],
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "metadata": self.utils.normalize_metadata(row["metadata"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )

        return self.utils.ok(
            {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "has_more": (offset + page_size) < total,
                "filters": spec["filters"],
                "sort": sort_key,
            }
        )

    def _build_query_spec(self, query) -> tuple[dict[str, Any] | None, str | None]:
        """把列表与导出共用的过滤/排序参数编译成 SQL 片段。

        两处必须共用这一份：否则「导出」的集合会与页面当前所见不一致，
        而这种错位在导出文件里看不出来（静默错位）。
        """
        session_id = self.utils.optional_text(query.get("session_id"))
        persona_id = self.utils.optional_text(query.get("persona_id"))
        keyword = str(query.get("keyword", "")).strip()
        status_filter = str(query.get("status", "all")).strip().lower() or "all"
        type_filter = self.utils.optional_text(query.get("type"))
        if type_filter and type_filter.lower() == "all":
            type_filter = None
        sort_key = str(query.get("sort", "created_desc")).strip().lower()

        where_clauses: list[str] = []
        params: list[Any] = []
        type_expr = (
            "UPPER(COALESCE("
            "CASE WHEN json_valid(metadata) "
            "THEN json_extract(metadata, '$.memory_type') END,"
            "'GENERAL'"
            "))"
        )

        if session_id:
            where_clauses.append(
                "CASE WHEN json_valid(metadata) "
                "THEN json_extract(metadata, '$.session_id') END = ?"
            )
            params.append(session_id)

        if persona_id:
            where_clauses.append(
                "CASE WHEN json_valid(metadata) "
                "THEN json_extract(metadata, '$.persona_id') END = ?"
            )
            params.append(persona_id)

        if status_filter != "all":
            where_clauses.append(
                "COALESCE("
                "CASE WHEN json_valid(metadata) "
                "THEN json_extract(metadata, '$.status') END,"
                "'active'"
                ") = ?"
            )
            params.append(status_filter)

        if type_filter:
            where_clauses.append(f"{type_expr} = ?")
            params.append(type_filter.upper())

        if keyword:
            keyword_like = f"%{keyword}%"
            if keyword.isdigit():
                where_clauses.append(
                    "(CAST(id AS TEXT) = ? OR text LIKE ? COLLATE NOCASE)"
                )
                params.extend([keyword, keyword_like])
            else:
                where_clauses.append(
                    "("
                    "text LIKE ? COLLATE NOCASE "
                    "OR COALESCE("
                    "CASE WHEN json_valid(metadata) "
                    "THEN json_extract(metadata, '$.memory_type') END,"
                    "''"
                    ") LIKE ? COLLATE NOCASE"
                    ")"
                )
                params.extend([keyword_like, keyword_like])

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        created_expr = (
            "COALESCE("
            "CASE WHEN json_valid(metadata) "
            "THEN CAST(json_extract(metadata, '$.create_time') AS REAL) END,"
            "0)"
        )
        updated_expr = (
            "COALESCE("
            "CASE WHEN json_valid(metadata) "
            "THEN CAST(json_extract(metadata, '$.updated_at') AS REAL) END,"
            "CASE WHEN json_valid(metadata) "
            "THEN CAST(json_extract(metadata, '$.create_time') AS REAL) END,"
            "0)"
        )
        importance_raw_expr = (
            "COALESCE("
            "CASE WHEN json_valid(metadata) "
            "THEN CAST(json_extract(metadata, '$.importance') AS REAL) END,"
            "0.5)"
        )
        importance_expr = (
            f"CASE WHEN {importance_raw_expr} <= 1.0 "
            f"THEN {importance_raw_expr} * 10.0 ELSE {importance_raw_expr} END"
        )
        sort_options = {
            "created_desc": f"{created_expr} DESC, id DESC",
            "created_asc": f"{created_expr} ASC, id ASC",
            "updated_desc": f"{updated_expr} DESC, id DESC",
            "updated_asc": f"{updated_expr} ASC, id ASC",
            "importance_desc": f"{importance_expr} DESC, id DESC",
            "importance_asc": f"{importance_expr} ASC, id ASC",
            "type_asc": f"{type_expr} ASC, id DESC",
            "type_desc": f"{type_expr} DESC, id DESC",
            "id_desc": "id DESC",
            "id_asc": "id ASC",
        }
        sort_expr = sort_options.get(sort_key)
        if sort_expr is None:
            sort_key = "created_desc"
            sort_expr = sort_options[sort_key]

        return {
            "where_clause": where_clause,
            "params": params,
            "sort_key": sort_key,
            "sort_expr": sort_expr,
            "persona_id": persona_id,
            "filters": {
                "session_id": session_id,
                "keyword": keyword,
                "status": status_filter,
                "type": type_filter,
            },
        }, None

    async def export_memories(self, memory_engine):
        """按当前筛选（或指定 ID）导出记忆，返回可直接存盘的文件响应。

        返回 ``(content, status_code, headers)`` 元组：官方插件页桥的
        ``files:download`` 会把响应体原样存成文件，因此这里不能返回
        ``{"status": "ok"}`` 包装，否则用户存下的是包装 JSON。

        查询参数: 与列表相同，另加
            - format: json（默认）| csv
            - ids: 逗号分隔的记忆 ID；给出时只导出这些 ID（页面勾选场景）
            - limit: 条数上限（默认 20000、上限 100000、0 表示取上限）
        """
        query = request.args
        export_format = str(query.get("format", "json")).strip().lower() or "json"
        if export_format not in self.EXPORT_FORMATS:
            return self._export_error("导出格式仅支持 json / csv", 400)

        spec, error = self._build_query_spec(query)
        if error:
            return self._export_error(error, 400)

        ids, id_error = self._parse_export_ids(query.get("ids"))
        if id_error:
            return self._export_error(id_error, 400)

        try:
            limit = int(query.get("limit", self.EXPORT_DEFAULT_LIMIT))
        except (TypeError, ValueError):
            return self._export_error("limit 参数无效", 400)
        if limit <= 0:
            limit = self.EXPORT_MAX_LIMIT
        else:
            limit = min(limit, self.EXPORT_MAX_LIMIT)

        db_path = getattr(memory_engine, "db_path", None)
        if not db_path:
            return self._export_error("MemoryEngine db_path unavailable", 500)

        where_clause = spec["where_clause"]
        params = list(spec["params"])
        if ids:
            placeholders = ", ".join("?" for _ in ids)
            ids_clause = f"id IN ({placeholders})"
            where_clause = (
                f"{where_clause} AND {ids_clause}" if where_clause else f"WHERE {ids_clause}"
            )
            params.extend(ids)

        try:
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                count_cursor = await db.execute(
                    f"SELECT COUNT(*) AS total FROM documents {where_clause}",
                    params,
                )
                count_row = await count_cursor.fetchone()
                total = int(count_row["total"]) if count_row else 0

                cursor = await db.execute(
                    f"""
                    SELECT id, doc_id, text, metadata, created_at, updated_at
                    FROM documents
                    {where_clause}
                    ORDER BY {spec["sort_expr"]}
                    LIMIT ?
                    """,
                    (*params, limit),
                )
                rows = await cursor.fetchall()
        except Exception as exc:
            logger.error(f"{tag('page')} 导出记忆失败: {exc}", exc_info=True)
            return self._export_error("导出失败", 500)

        items = [self._export_item(row) for row in rows]
        truncated = total > len(items)
        filename = self._export_filename(export_format, len(items), truncated)
        if export_format == "csv":
            content = self._to_csv(items)
            content_type = "text/csv; charset=utf-8"
        else:
            content = json.dumps(
                {
                    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "format_version": 1,
                    "format": "json",
                    "filtered_by": {
                        **spec["filters"],
                        "persona_id": spec["persona_id"],
                        "ids": ids,
                        "sort": spec["sort_key"],
                    },
                    "count": len(items),
                    "total_matched": total,
                    "truncated": truncated,
                    "memories": items,
                },
                ensure_ascii=False,
                indent=2,
            )
            content_type = "application/json; charset=utf-8"

        logger.info(
            f"{tag('page')} 导出记忆: {len(items)}/{total} 条，格式 {export_format}"
            + ("（触达上限，文件名含 _partial）" if truncated else "")
        )
        return (
            content,
            200,
            {
                "Content-Type": content_type,
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    def _parse_export_ids(self, raw: Any) -> tuple[list[int], str | None]:
        """解析逗号分隔的 ID 列表；未提供时返回空列表表示"按筛选导出"。"""
        text = str(raw or "").strip()
        if not text:
            return [], None
        ids: list[int] = []
        for part in text.split(","):
            piece = part.strip()
            if not piece:
                continue
            try:
                ids.append(int(piece))
            except (TypeError, ValueError):
                return [], f"ids 含非法记忆 ID：{piece}"
        return ids, None

    def _export_item(self, row) -> dict[str, Any]:
        """把一行文档展开成导出字段（CSV/JSON 共用）。"""
        metadata = self.utils.normalize_metadata(row["metadata"])
        return {
            "id": row["id"],
            "doc_id": row["doc_id"],
            "text": row["text"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "memory_type": metadata.get("memory_type", "GENERAL"),
            "status": metadata.get("status", "active"),
            "importance": metadata.get("importance"),
            "persona_id": metadata.get("persona_id"),
            "session_id": metadata.get("session_id"),
            "create_time": metadata.get("create_time"),
            "event_time": metadata.get("event_time"),
            "participants": metadata.get("participants"),
            "topics": metadata.get("topics"),
            "key_facts": metadata.get("key_facts"),
            "summary": metadata.get("neutral_summary") or metadata.get("canonical_summary"),
        }

    def _export_filename(self, export_format: str, count: int, truncated: bool) -> str:
        """文件名自带条数与截断标记，页面无需额外协议即可提示用户。"""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_partial" if truncated else ""
        return f"livingmemory_cm_memories_{stamp}_{count}{suffix}.{export_format}"

    @staticmethod
    def _export_error(message: str, status_code: int):
        """下载路径不会解析 {"status": "error"}，改用非 2xx 让页面提示错误。"""
        return (
            json.dumps({"status": "error", "message": message}, ensure_ascii=False),
            status_code,
            {"Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store"},
        )

    @classmethod
    def _to_csv(cls, items: list[dict[str, Any]]) -> str:
        """导出 CSV：UTF-8 BOM 便于 Excel 直接打开中文，单元格按 RFC 4180 转义。"""
        columns = [
            ("id", "ID"),
            ("doc_id", "doc_id"),
            ("memory_type", "类型"),
            ("status", "状态"),
            ("importance", "重要性"),
            ("persona_id", "人格"),
            ("session_id", "会话"),
            ("create_time", "创建时间戳"),
            ("event_time", "事件时间"),
            ("participants", "参与者"),
            ("topics", "主题"),
            ("key_facts", "事实"),
            ("summary", "摘要"),
            ("text", "内容"),
            ("created_at", "入库时间"),
            ("updated_at", "更新时间"),
        ]
        lines = [",".join(name for _, name in columns)]
        for item in items:
            lines.append(
                ",".join(cls._csv_cell(item.get(key)) for key, _ in columns)
            )
        return "\ufeff" + "\r\n".join(lines) + "\r\n"

    @classmethod
    def _csv_cell(cls, value: Any) -> str:
        """列表字段用 、 连接；含分隔符/换行/引号时按标准加引号并转义。"""
        if value is None:
            text = ""
        elif isinstance(value, list | tuple):
            text = "、".join(str(part) for part in value)
        elif isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)

        if text.startswith(cls._CSV_FORMULA_PREFIXES):
            # 记忆正文可能来自群成员，前导公式字符会被表格软件执行
            text = "'" + text
        if any(char in text for char in (",", '"', "\n", "\r")):
            text = '"' + text.replace('"', '""') + '"'
        return text

    async def get_memory_detail(self, memory_engine) -> dict[str, Any]:
        """
        获取单个记忆的完整详情

        查询参数:
            - memory_id: 记忆ID（必需）

        Returns:
            包含记忆详情和相关图谱上下文的字典
        """
        from ..utils.number_utils import clamp_float

        query = request.args
        try:
            memory_id = int(query.get("memory_id", ""))
        except (TypeError, ValueError):
            return self.utils.error("memory_id 必须是整数")

        memory = await self._get_memory_record(memory_id, memory_engine)
        if not memory:
            return self.utils.error("记忆不存在")

        metadata = self.utils.normalize_metadata(memory.get("metadata"))

        # 构建完整的详情数据
        detail = {
            "memory_id": memory.get("id"),
            "doc_id": memory.get("doc_id"),
            "text": memory.get("text"),
            "summary": metadata.get("canonical_summary") or memory.get("text", ""),
            "created_at": memory.get("created_at"),
            "updated_at": memory.get("updated_at"),
            "metadata": metadata,
            "memory_type": metadata.get("memory_type", "GENERAL"),
            "importance": clamp_float(metadata.get("importance"), default=0.5),
            "status": metadata.get("status", "active"),
            "session_id": metadata.get("session_id"),
            "persona_id": metadata.get("persona_id"),
            "key_facts": metadata.get("key_facts", []),
            "topics": metadata.get("topics", []),
            "create_time": metadata.get("create_time"),
            "last_access_time": metadata.get("last_access_time"),
            "update_history": metadata.get("update_history", []),
        }

        # 附加相关的图谱子图
        graph_store = self.utils.get_graph_store(memory_engine)
        if graph_store is not None:
            try:
                subgraph = await graph_store.get_subgraph_for_memories(
                    [memory_id],
                    limit_entries=20,
                    limit_nodes=20,
                    limit_edges=30,
                )
                detail["graph_context"] = {
                    "nodes": subgraph.get("nodes", []),
                    "edges": subgraph.get("edges", []),
                    "entries": subgraph.get("entries", []),
                }
            except Exception:
                detail["graph_context"] = None
        else:
            detail["graph_context"] = None

        return self.utils.ok(detail)

    async def update_memory(self, memory_engine) -> dict[str, Any]:
        """
        更新单个记忆的字段

        支持的字段:
            - content: 记忆内容（会创建新记忆并删除旧记忆）
            - importance: 重要性（0-1 或 0-10）
            - status: 状态（active/archived/deleted）
            - type: 类型

        Payload:
            - memory_id: 记忆ID（必需）
            - field: 要更新的字段（必需）
            - value: 新值（必需）
            - reason: 更新原因（可选）

        Returns:
            包含更新结果的字典
        """
        from ..utils.number_utils import clamp_float

        payload = await request.get_json(silent=True) or {}
        try:
            memory_id = int(payload.get("memory_id"))
        except (TypeError, ValueError):
            return self.utils.error("memory_id 必须是整数")

        field = str(payload.get("field", "")).strip()
        value = payload.get("value")
        value_scale = str(payload.get("value_scale", "auto")).strip().lower()
        reason = str(payload.get("reason", "")).strip()

        if not field or value is None:
            return self.utils.error("需要指定 field 和 value")

        memory = await self._get_memory_record(memory_id, memory_engine)
        if not memory:
            return self.utils.error("记忆不存在")

        current_metadata = self.utils.normalize_metadata(memory.get("metadata"))

        # 特殊处理：content 更新需要重新创建记忆
        if field == "content":
            new_content = str(value).strip()
            if not new_content:
                return self.utils.error("记忆内容不能为空")

            session_id = current_metadata.get("session_id")
            persona_id = current_metadata.get("persona_id")
            importance = clamp_float(current_metadata.get("importance"), default=0.5)
            updated_at = time.time()
            update_history = self.utils.append_update_history(
                current_metadata,
                field="content",
                old_value=memory.get("text", ""),
                new_value=new_content,
                reason=reason,
                timestamp=updated_at,
            )

            if reason:
                current_metadata["update_reason"] = reason
            current_metadata["updated_at"] = updated_at
            current_metadata["previous_content"] = str(memory.get("text", ""))[:100]
            current_metadata["update_history"] = update_history

            new_memory_id = None
            try:
                new_memory_id = await memory_engine.add_memory(
                    content=new_content,
                    session_id=session_id,
                    persona_id=persona_id,
                    importance=importance,
                    metadata=current_metadata,
                )
                delete_success = await memory_engine.delete_memory(memory_id)
                if not delete_success:
                    await memory_engine.delete_memory(new_memory_id)
                    return self.utils.error("旧记忆删除失败，已回滚本次内容更新")
            except Exception as exc:
                if new_memory_id is not None:
                    try:
                        await memory_engine.delete_memory(new_memory_id)
                    except Exception:
                        logger.error(
                            f"{tag('page')} 回滚新记忆失败 (new_memory_id={new_memory_id})",
                            exc_info=True,
                        )
                logger.error(f"{tag('page')} 更新记忆内容失败: {exc}", exc_info=True)
                return self.utils.error(str(exc))

            return self.utils.ok(
                {
                    "message": f"记忆内容已更新（ID: {memory_id} → {new_memory_id}）",
                    "old_memory_id": memory_id,
                    "new_memory_id": new_memory_id,
                    "field": field,
                }
            )

        # 其他字段更新
        updates: dict[str, Any] = {}
        old_value_for_history: Any
        new_value_for_history: Any
        if field == "importance":
            try:
                normalized = self._normalize_importance_update(value, value_scale)
            except ValueError as exc:
                return self.utils.error(str(exc))
            updates["importance"] = normalized
            old_value_for_history = self.utils.importance_to_display(
                current_metadata.get("importance", 0.5)
            )
            new_value_for_history = round(normalized * 10.0, 2)
        elif field == "status":
            status_value = str(value).strip()
            if status_value not in {"active", "archived", "deleted"}:
                return self.utils.error("状态必须是 active、archived 或 deleted")
            updates["metadata"] = {"status": status_value}
            old_value_for_history = current_metadata.get("status", "active")
            new_value_for_history = status_value
        elif field == "type":
            type_value = str(value).strip()
            if not type_value:
                return self.utils.error("类型不能为空")
            updates["metadata"] = {"memory_type": type_value}
            old_value_for_history = current_metadata.get("memory_type", "GENERAL")
            new_value_for_history = type_value
        else:
            return self.utils.error(f"不支持编辑字段: {field}")

        updated_at = time.time()
        updates.setdefault("metadata", {})
        updates["metadata"]["update_history"] = self.utils.append_update_history(
            current_metadata,
            field=field,
            old_value=old_value_for_history,
            new_value=new_value_for_history,
            reason=reason,
            timestamp=updated_at,
        )
        updates["metadata"]["updated_at"] = updated_at

        if reason:
            updates["metadata"]["update_reason"] = reason

        try:
            success = await memory_engine.update_memory(memory_id, updates)
        except Exception as exc:
            logger.error(f"{tag('page')} 更新记忆失败: {exc}", exc_info=True)
            return self.utils.error(str(exc))

        if not success:
            return self.utils.error("更新失败")

        return self.utils.ok(
            {
                "message": f"记忆 {memory_id} 的 {field} 已更新",
                "memory_id": memory_id,
                "field": field,
            }
        )

    async def batch_delete_memories(self, memory_engine) -> dict[str, Any]:
        """
        批量删除记忆

        Payload:
            - memory_ids: 记忆ID列表（必需）

        Returns:
            包含删除统计的字典
        """
        payload = await request.get_json(silent=True) or {}
        memory_ids = payload.get("memory_ids", [])
        if not isinstance(memory_ids, list) or not memory_ids:
            return self.utils.error("需要提供记忆 ID 列表")

        deleted_count = 0
        failed_count = 0
        failed_ids: list[Any] = []

        valid_ids: list[int] = []
        for raw_id in memory_ids:
            try:
                valid_ids.append(int(raw_id))
            except Exception:
                failed_count += 1
                failed_ids.append(raw_id)

        if valid_ids:
            deleted_count = await memory_engine.batch_delete_memories(valid_ids)

        return self.utils.ok(
            {
                "deleted_count": deleted_count,
                "failed_count": failed_count,
                "total": len(memory_ids),
                "failed_ids": failed_ids,
            }
        )

    async def batch_update_memories(self, memory_engine) -> dict[str, Any]:
        """
        批量更新记忆字段

        支持的字段:
            - status: 状态
            - importance: 重要性
            - type: 类型

        Payload:
            - memory_ids: 记忆ID列表（必需）
            - field: 要更新的字段（必需）
            - value: 新值（必需）

        Returns:
            包含更新统计的字典
        """
        payload = await request.get_json(silent=True) or {}
        memory_ids = payload.get("memory_ids", [])
        field = str(payload.get("field", "")).strip()
        value = payload.get("value")
        value_scale = str(payload.get("value_scale", "auto")).strip().lower()

        if not isinstance(memory_ids, list) or not memory_ids:
            return self.utils.error("需要提供记忆 ID 列表")
        if not field or value is None:
            return self.utils.error("需要指定 field 和 value")

        if field not in ("status", "importance", "type"):
            return self.utils.error(f"批量更新不支持字段: {field}")

        updated_count = 0
        failed_ids: list[Any] = []

        for raw_id in memory_ids:
            try:
                memory_id = int(raw_id)
            except (TypeError, ValueError):
                failed_ids.append(raw_id)
                continue

            try:
                updates: dict[str, Any] = {}
                if field == "status":
                    status_value = str(value).strip()
                    if status_value not in {"active", "archived", "deleted"}:
                        failed_ids.append(raw_id)
                        continue
                    updates["metadata"] = {"status": status_value}
                elif field == "importance":
                    try:
                        updates["importance"] = self._normalize_importance_update(
                            value, value_scale
                        )
                    except ValueError:
                        failed_ids.append(raw_id)
                        continue
                elif field == "type":
                    type_value = str(value).strip()
                    if not type_value:
                        failed_ids.append(raw_id)
                        continue
                    updates["metadata"] = {"memory_type": type_value}

                success = await memory_engine.update_memory(memory_id, updates)
                if success:
                    updated_count += 1
                else:
                    failed_ids.append(raw_id)
            except Exception:
                failed_ids.append(raw_id)

        return self.utils.ok(
            {
                "updated_count": updated_count,
                "failed_count": len(failed_ids),
                "total": len(memory_ids),
                "failed_ids": failed_ids,
            }
        )

    async def _get_memory_record(
        self, memory_id: int, memory_engine
    ) -> dict[str, Any] | None:
        """
        获取单个记忆的原始记录

        Args:
            memory_id: 记忆ID
            memory_engine: 记忆引擎实例

        Returns:
            记忆记录字典，如果不存在则返回 None
        """
        db_path = getattr(memory_engine, "db_path", None)
        if not db_path:
            return None

        try:
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT id, doc_id, text, metadata, created_at, updated_at
                    FROM documents
                    WHERE id = ?
                    """,
                    (memory_id,),
                )
                row = await cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "metadata": row["metadata"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
        except Exception as exc:
            logger.error(f"{tag('page')} 获取记忆记录失败: {exc}", exc_info=True)
            return None
