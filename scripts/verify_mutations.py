r"""变异验证：确认测试真的能拦住它要守的那类回归。

AGENTS.md 把"删完/补完做变异验证"列为准入条件（不是可选）——不失败的用例等于没写。
用法：
    & 'C:\Users\Administrator\AppData\Local\AstrBot\backend\python\python.exe' ^
        D:\AI\dsh\plugin\livingmemory_cm\scripts\verify_mutations.py

维护：每批新增/修改测试后，把该批最关键的 3-5 条断言按下面 MUTATIONS 的格式登记——
`(说明, 文件, 原文, 改坏后的文本, 目标用例)`；改坏后必须让目标用例失败。
注意 ① 变异要真的改变行为（等价改写会让用例假通过）；② 目标用例必须真的被收集执行。
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable

MUTATIONS = [
    (
        "纯引用消息不再回退引用文本（召回被静默跳过）",
        "core/event_handler_modules/message_utils.py",
        "        return self._quoted_message_text(event)",
        '        return ""  # 变异：不回退引用文本',
        "tests/test_utils.py::test_recall_query_falls_back_to_quoted_message_text",
    ),
    (
        "近期槽位直查库路径不再过滤冻结 persona（隔离泄漏）",
        "core/memory/memory_search_service.py",
        "        recent = self._filter_frozen_personas(recent, excluded_personas or set())",
        "        pass  # 变异：近期槽位不再过滤冻结人格",
        "tests/test_freeze_store.py::test_recent_slot_does_not_leak_frozen_persona",
    ),
    (
        "分区键少一维（persona 分区互相覆盖）",
        "core/reflection/cursor_service.py",
        'f"{conversation_id}\\x1f{persona_id}\\x1f{scope}\\x1f{mode_signature}"',
        'f"{conversation_id}\\x1f{scope}\\x1f{mode_signature}"',
        "tests/test_reflection_cm_path.py::"
        "test_cursor_is_partitioned_by_conversation_persona_and_scope",
    ),
    (
        "首次运行按 epoch 计天数（全库猛衰减）",
        "core/schedulers/decay_scheduler.py",
        "        if not last_date_str:\n            return 0",
        "        if not last_date_str:\n            return 99999",
        "tests/test_decay_scheduler.py::"
        "TestDecaySchedulerMissedDaysCalculation::"
        "test_calculate_missed_days_no_previous_run",
    ),
    (
        "清理阈值 0 被当成开启（静默删旧记忆）",
        "core/schedulers/decay_scheduler.py",
        "            if cleanup_days > 0:",
        "            if cleanup_days >= 0:",
        "tests/test_decay_scheduler.py::"
        "TestDecaySchedulerExecution::test_execute_decay_with_auto_cleanup_disabled",
    ),
    (
        "备份失败仍先写版本标记（此后永不备份）",
        "core/managers/backup_manager.py",
        "        try:\n            copied_count = 0",
        "        self.write_current_version()\n        try:\n            copied_count = 0",
        "tests/test_backup_manager.py::"
        "test_backup_failure_does_not_mark_version_complete",
    ),
    (
        "备份通配漏掉图数据文件（备份悄悄不完整）",
        "core/managers/backup_manager.py",
        '    "livingmemory_graph_documents.db",\n',
        "",
        "tests/test_backup_manager.py::test_backup_covers_all_data_file_patterns",
    ),
    (
        "坏 backup_info.json 直接抛出（列表空白）",
        "core/managers/backup_manager.py",
        "                except (json.JSONDecodeError, OSError):\n                    pass",
        "                except (json.JSONDecodeError, OSError):\n                    raise",
        "tests/test_backup_manager.py::test_list_backups_corrupt_info_json",
    ),
    (
        "importance 展示刻度被当成存储值（1.0 写回成 10）",
        "core/page_api_modules/memory_handler.py",
        '        if scale in {"display", "0-10", "ten"}:',
        "        if False:",
        "tests/test_page_api.py::"
        "TestUpdateMemory::test_importance_display_scale_preserves_one_point_zero",
    ),
    (
        "CSV 公式前缀守卫被移除（表格软件执行记忆正文）",
        "core/page_api_modules/memory_handler.py",
        "        if text.startswith(cls._CSV_FORMULA_PREFIXES):",
        "        if False:",
        "tests/test_page_api.py::"
        "TestExportMemories::test_export_csv_escapes_cells_and_neutralizes_formulas",
    ),
    (
        "命中的既有会话不再刷新活跃时间（活跃时间永不前进）",
        "core/managers/conversation_manager.py",
        "            await self.store.update_session_activity(session_id)",
        "            pass  # 变异：活跃时间不再刷新",
        "tests/test_conversation_manager.py::"
        "test_create_or_get_session_returns_existing_and_updates_activity",
    ),
    (
        "查询侧表达式漂移（status 常量被改歪）",
        "core/memory/memory_search_service.py",
        "from .metadata_expr import STATUS_EXPR as _DOC_STATUS_EXPR",
        (
            "_DOC_STATUS_EXPR = (\n"
            "    \"COALESCE(CASE WHEN json_valid(metadata) \"\n"
            "    \"THEN json_extract(metadata, '$.status') END, 'enabled')\"\n"
            ")"
        ),
        "tests/test_retrieval_components.py::"
        "test_metadata_expression_indexes_back_every_hot_query",
    ),
    (
        "索引侧表达式漂移（DDL 包一层 UPPER）",
        "core/memory/memory_schema_service.py",
        "                {STATUS_EXPR},",
        "                UPPER({STATUS_EXPR}),",
        "tests/test_retrieval_components.py::"
        "test_metadata_expression_indexes_back_every_hot_query",
    ),
    (
        "超长内容又变回截断入库（落库写的是截断文本）",
        "core/retrieval/vector_retriever.py",
        "        int_id = await document_store.insert_document(doc_uuid, content, metadata)",
        "        int_id = await document_store.insert_document(\n"
        "            doc_uuid, embedding_text, metadata\n"
        "        )",
        "tests/test_retrieval_components.py::"
        "test_long_content_stores_full_text_and_truncates_only_embedding",
    ),
    (
        "旧注入模式不再降级告警",
        "core/utils/injection_adapter.py",
        "        if mode in _DEPRECATED_MODES:",
        "        if False:",
        "tests/test_utils.py::"
        "test_injection_adapter_downgrades_legacy_modes_with_reason",
    ),
    (
        "面板详情刻度标记被忽略",
        "pages/dashboard/modules/utils.js",
        'if (!detail || detail.importance_scale !== "display") {',
        "if (!detail) {",
        "tests/test_dashboard_frontend.py::"
        "test_dashboard_escape_and_retry_policy",
    ),
]


def run_test(nodeid: str) -> tuple[str, str]:
    """跑目标用例，返回 (判定, 输出尾部)。

    判定取值：`pass` / `fail` / `invalid`。`invalid` 用于"用例压根没跑起来"的情形
    —— 收集错误、导入失败、没收集到用例都会让退出码非 0，误判成"被拦住"；
    AGENTS.md 明确要求判"被拦住"必须确认测试真的跑起来了。
    """
    proc = subprocess.run(
        # node id 以插件目录名开头（cwd 是父目录，包按 livingmemory_cm.* 导入）
        [PY, "-m", "pytest", f"{ROOT.name}/{nodeid}", "-q", "--no-header",
         "-p", "no:cacheprovider"],
        # 注意：必须用**父目录**作 cwd，测试模块是按 `livingmemory_cm.core...` 导入的
        cwd=str(ROOT.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = out[-400:]
    if "ModuleNotFoundError" in out or "ImportError while importing" in out:
        return "invalid", tail
    if re.search(r"no tests ran|errors? during collection|error during collection", out):
        return "invalid", tail
    passed = re.search(r"(\d+) passed", out)
    failed = re.search(r"(\d+) failed", out)
    if failed and int(failed.group(1)) > 0:
        return "fail", tail
    if passed and int(passed.group(1)) > 0:
        return "pass", tail
    return "invalid", tail


def apply_edits(original: str, edits) -> tuple[str, bool]:
    """应用一处或多处替换。

    `edits` 可以是 `(old, new)` 或 `[(old, new), ...]`。多行变异的用途：有些守卫是
    **互为冗余的多层**（例如冻结 persona 过滤在向量路 / 合并后兜底 / 近期槽位直查库
    三处），只改一层会被另一层兜住，跑不出差别——那种情况下要整组一起改才能验证
    "边界真的被覆盖"。任一条 old 没命中就整体失败（不静默跳过）。
    """
    pairs = edits if isinstance(edits, list) else [edits]
    text = original
    for old, new in pairs:
        if old not in text:
            return original, False
        text = text.replace(old, new, 1 if not old.endswith("\n") else 1)
    return text, True


def main() -> int:
    problems: list[str] = []
    for entry in MUTATIONS:
        # 两种登记形式：`(标签, 文件, old, new, 用例)`；多行变异 `(标签, 文件, [(old, new), ...], 用例)`
        if len(entry) == 5:
            label, rel, old, new, nodeid = entry
            edits = (old, new)
        else:
            label, rel, edits, nodeid = entry
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        mutated, ok = apply_edits(original, edits)
        if not ok:
            problems.append(f"{label}: 变异目标未找到（{rel}）")
            print(f"[SKIP] {label}：目标串未找到")
            continue
        # 基线：未变异时必须通过，否则这次变异的结果没有意义
        baseline, baseline_out = run_test(nodeid)
        if baseline != "pass":
            print(f"[INVALID] {label}：基线未通过（{baseline}）")
            print("   ", baseline_out.strip().splitlines()[-1] if baseline_out.strip() else "")
            problems.append(f"{label}: 基线未通过（{baseline}）")
            continue
        path.write_text(mutated, encoding="utf-8", newline="")
        try:
            verdict, output = run_test(nodeid)
        finally:
            path.write_text(original, encoding="utf-8", newline="")

        if verdict == "fail":
            print(f"[caught] {label} -> {nodeid}")
        elif verdict == "invalid":
            print(f"[INVALID] {label}：变异后用例没跑起来（不是被拦住）")
            print("   ", output.strip().splitlines()[-1] if output.strip() else "")
            problems.append(f"{label}: 变异后用例没跑起来")
        else:
            print(f"[NOT-CAUGHT] {label} -> {nodeid}")
            problems.append(f"{label}: 用例没失败（{nodeid}）")

    print("\n=== 变异验证结论 ===")
    if problems:
        print("有未拦住的变异：")
        for item in problems:
            print(" -", item)
        return 1
    print(f"全部 {len(MUTATIONS)} 条变异都被对应用例拦住")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
