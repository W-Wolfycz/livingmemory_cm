"""
注入策略适配层

只支持 add-only 注入（`extra_user_content`）：记忆块追加到 `extra_user_content_parts`
并标记 `mark_as_temp()`，**不改写宿主对话历史**，因此不存在"注入内容被持久化、
下一轮再清理"的往返。

历史上还有三档会改写历史的模式（`user_message_before` / `user_message_after` /
`fake_tool_call`），已在 2.7.0-beta-cm 移除：它们都要依赖"下一轮开头把注入内容从
历史里抹掉"，一旦清理漏掉（会话切换、插件卸载、流式路径），注入块或伪造的工具调用
就会留在真实历史里。此处保留对旧配置值的降级说明，避免升级后静默改变行为。
"""

from __future__ import annotations

# 唯一受支持的注入方式
SUPPORTED_MODE = "extra_user_content"

# 已移除的历史模式 → 降级说明（旧配置命中时给出可见告警）
_DEPRECATED_MODES: dict[str, str] = {
    "user_message_before": "该模式会改写宿主对话历史，已在 2.7.0-beta-cm 移除",
    "user_message_after": "该模式会改写宿主对话历史，已在 2.7.0-beta-cm 移除",
    "fake_tool_call": "伪造工具调用会留在历史里，已在 2.7.0-beta-cm 移除",
}


class InjectionAdapter:
    """把配置里的注入方式解析成实际使用的（唯一）模式。"""

    def resolve(self, configured_mode: str) -> tuple[str, str | None]:
        """解析注入模式。

        Returns:
            (resolved_mode, fallback_reason)
            - resolved_mode: 实际使用的注入方式（恒为 `extra_user_content`）
            - fallback_reason: 旧配置降级说明；未降级时为 None
        """
        mode = str(configured_mode or "").strip()
        if mode == SUPPORTED_MODE:
            return SUPPORTED_MODE, None
        if mode in _DEPRECATED_MODES:
            return (
                SUPPORTED_MODE,
                f"{mode} → {SUPPORTED_MODE}：{_DEPRECATED_MODES[mode]}，"
                "该配置项已无实际开关作用",
            )
        if mode:
            return SUPPORTED_MODE, f"未知注入方式 {mode} → {SUPPORTED_MODE}"
        return SUPPORTED_MODE, None
