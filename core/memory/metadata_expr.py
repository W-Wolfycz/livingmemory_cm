"""documents.metadata 的规范 SQL 表达式（单一来源）。

这些表达式同时用于两处，**必须保持一致**：
1. `memory_schema_service` 建立的表达式索引（`idx_doc_status_created`、
   `idx_doc_type_created`）——索引表达式与查询表达式同形时 SQLite 才会命中索引；
2. 检索与页面 API 的查询条件 / 排序。

因此它们只在这里定义一次，两边都从这里取；一旦某处手写副本漂移，索引会静默失效
（查询照常返回结果，只是退化为全表扫描）。`EXPLAIN QUERY PLAN` 断言负责守住这一点。

所有表达式都带 `json_valid(metadata)` 守卫：坏 JSON 行不会让整条查询报错。
"""

from __future__ import annotations

# status：缺字段视为 active（与索引、清理/召回口径一致）
STATUS_EXPR = (
    "COALESCE(CASE WHEN json_valid(metadata) "
    "THEN json_extract(metadata, '$.status') END, 'active')"
)

# memory_type：缺字段视为 GENERAL；类型比较统一大写
TYPE_EXPR = (
    "UPPER(COALESCE("
    "CASE WHEN json_valid(metadata) "
    "THEN json_extract(metadata, '$.memory_type') END,"
    "'GENERAL'"
    "))"
)

# create_time / updated_at / importance：CAST 成 REAL 后参与比较与排序
CREATED_EXPR = (
    "COALESCE("
    "CASE WHEN json_valid(metadata) "
    "THEN CAST(json_extract(metadata, '$.create_time') AS REAL) END,"
    "0)"
)
UPDATED_EXPR = (
    "COALESCE("
    "CASE WHEN json_valid(metadata) "
    "THEN CAST(json_extract(metadata, '$.updated_at') AS REAL) END,"
    "CASE WHEN json_valid(metadata) "
    "THEN CAST(json_extract(metadata, '$.create_time') AS REAL) END,"
    "0)"
)
IMPORTANCE_RAW_EXPR = (
    "COALESCE("
    "CASE WHEN json_valid(metadata) "
    "THEN CAST(json_extract(metadata, '$.importance') AS REAL) END,"
    "0.5)"
)
# 页面按 0-10 展示重要性，排序口径与展示一致
IMPORTANCE_EXPR = (
    f"CASE WHEN {IMPORTANCE_RAW_EXPR} <= 1.0 "
    f"THEN {IMPORTANCE_RAW_EXPR} * 10.0 ELSE {IMPORTANCE_RAW_EXPR} END"
)

# 分区字段：精确匹配，不 COALESCE（缺字段的行不应被当成某个具体分区）
SESSION_EXPR = (
    "CASE WHEN json_valid(metadata) "
    "THEN json_extract(metadata, '$.session_id') END"
)
PERSONA_EXPR = (
    "CASE WHEN json_valid(metadata) "
    "THEN json_extract(metadata, '$.persona_id') END"
)

__all__ = [
    "CREATED_EXPR",
    "IMPORTANCE_EXPR",
    "IMPORTANCE_RAW_EXPR",
    "PERSONA_EXPR",
    "SESSION_EXPR",
    "STATUS_EXPR",
    "TYPE_EXPR",
    "UPDATED_EXPR",
]
