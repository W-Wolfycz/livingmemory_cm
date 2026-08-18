# LivingMemoryCM

LivingMemoryCM 是面向 AstrBot 的长期记忆插件，基于
[astrbot_plugin_livingmemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory)
2.5.7 修改，并针对
[chat_memory](https://github.com/W-Wolfycz/chat_memory) 的上下文接管模式维护。
本项目是独立的 CM-only fork，不是上游官方发行版。

- 当前版本：`2.5.7-cm.1`
- 源码仓库：<https://github.com/W-Wolfycz/livingmemory_cm>
- 许可证：GNU Affero General Public License v3.0（AGPL-3.0）

> [!IMPORTANT]
> 必须安装并启用 `chat_memory >= 1.1.1`，开启 `context_takeover`，并设置
> `ct_limit_rounds > 0`；不满足条件时插件会拒绝启动。

## 项目特点

- ChatMemory 负责短期上下文和原始消息归档，LivingMemoryCM 只维护长期记忆。
- 文档路使用向量检索，图路使用关键词、图向量和 RRF 融合；不恢复上游 BM25 文档路。
- 反思读取 ChatMemory 会话，使用严格复合游标、稳定批次 ID 和幂等写入，避免重试时重复落库。
- 召回和反思按 session、persona 与用户身份隔离，支持近期对话消歧和可配置注入预算。
- 记忆包含中性摘要、关键事实和独立记忆原子，支持 TTL、衰减、归档和图关系维护。
- SQLite/FAISS 使用代际与 Embedding 指纹校验；索引重建采用影子存储，失败时保留旧索引。
- Dashboard 采用中文单语言，保留 CM persona 筛选、批量编辑/删除和 2D 知识图谱。
- 可选提供 `recall_long_term_memory` 与 `memorize_long_term_memory` 两个 Agent 工具。

相较上游，本 fork 不提供 PromptManager、JSON/CSV 导入导出、原始来源编辑和上游自管
conversation history。

## 安装与升级

运行要求：

- AstrBot `>= 4.24.2`
- `chat_memory >= 1.1.1`
- 可用的 Embedding Provider 与 LLM Provider

将 ZIP 中的 `livingmemory_cm` 文件夹放入 AstrBot 的 `data/plugins/`，然后在 AstrBot 中
reload 或重启插件。AstrBot 会按 `requirements.txt` 安装插件依赖。

Release 包不包含 `data/`、数据库、FAISS 索引或用户配置。升级时应保留部署端原有数据
目录；插件检测到版本变化后会先执行版本备份。重要数据仍建议在升级前额外做一次离线备份。

## 配置

最常用的配置项：

- `provider_settings.embedding_provider_id`：Embedding Provider，留空使用 AstrBot 默认值。
- `provider_settings.llm_provider_id`：LLM Provider，留空使用 AstrBot 默认值。
- `recall_engine.injection_method`：长期记忆注入方式，默认 `extra_user_content`。
- `recall_engine.query_context_rounds`：用于指代消歧的最近 CM 历史，单位跟随 CM `llm_status_filter`（仅 `llm_success` 按轮、含其他状态按条），默认 `2`。
- `reflection_engine.trigger_count`：反思触发数量；`0` 跟随 ChatMemory 配置。
- `graph_memory.graph_route_weight`：图路融合权重，默认 `0.35`；文档路权重自动为 `1 - graph_route_weight`。
- `log_with_bot_id`（全局配置项，不属于配置段）：多 Bot 共存时在关键事件日志前缀附加 self_id 原文（如 `[livingmemory_cm:bot-10000]`），便于定位；会话/用户引用仍脱敏。
- `maintenance.cleanup_days_threshold`、`backup_keep_days`：`0` 表示关闭对应维护任务。

全部 7 个配置段 + 1 个全局配置项（`log_with_bot_id`）、共 44 个配置项的作用和建议见
AstrBot 配置面板中各配置项的说明文案。

## 命令

| 命令 | 说明 |
| --- | --- |
| `/lmem status` | 查看记忆库、索引与图向量状态 |
| `/lmem search <关键词> [数量]` | 搜索长期记忆 |
| `/lmem forget <ID>` | 删除指定记忆 |
| `/lmem rebuild-graph` | 使用影子存储安全重建图索引 |
| `/lmem webui` | 查看 Dashboard 入口 |
| `/lmem reset` | 重置当前会话的记忆上下文 |
| `/lmem cleanup [preview\|exec]` | 预演或清理历史消息中的旧注入片段 |
| `/lmem help` | 显示帮助和当前源码入口 |

## 开发与验收边界

```powershell
python -m pip install -r requirements.txt -r requirements-test.txt
python tests\run_tests.py
```

本地测试只验证领域算法、状态机、存储协议和边界条件；`astrbot.api` 与 `astrbot.core`
由 `tests/conftest.py` 提供最小 fake 类型树，不导入、也不安装真实 AstrBot
core/backend，因此不再需要 `--astrbot-source` / `--astrbot-backend`。当前代码级
回归为 `508 passed`；AstrBot reload、真实 core 兼容性、Provider、平台发送、
ChatMemory Hook 顺序和 Dashboard 浏览器交互必须在部署端验收。

## 更新记录

本 fork 的公开变更位于 [CHANGELOG.md](CHANGELOG.md) 顶部；其后保留上游
LivingMemory 的原始版本记录。

## 许可证与源码

本项目继承上游 AGPL-3.0，完整条款见 [LICENSE](LICENSE)，fork 来源、主要修改和分发
要求见 [NOTICE.md](NOTICE.md)。复制、修改、部署或分发时应保留这些文件，并继续按
AGPL-3.0 提供与实际运行版本对应的完整源码；通过网络与修改版交互的用户也应能显著、
免费地取得对应源码。

Dashboard 随附的 Lucide 材料采用其自有许可证，见
[pages/dashboard/vendor/LUCIDE_LICENSE](pages/dashboard/vendor/LUCIDE_LICENSE)。
ChatMemory 是未包含在本仓库中的独立 MIT 依赖；如将其一并分发，还需保留 ChatMemory
自己的许可证和版权声明。
