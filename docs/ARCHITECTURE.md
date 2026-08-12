# LivingMemoryCM 整体流程与架构说明

> 适用版本：`2.5.7-cm`
> 本文以当前源码和 AstrBot 4.26.7 参考源码为准。README、CHANGELOG 或上游设计与源码冲突时，以源码为准。

## 1. 一句话定位

LivingMemoryCM 不负责保存或接管短期对话历史。

`chat_memory` 是原始消息和 LLM 短期上下文的唯一数据源；LivingMemoryCM 只负责：

1. 从 CM 历史中分批萃取长期记忆。
2. 将记忆写入文档、原子和图谱存储。
3. 在新的 LLM 请求到来时检索相关记忆。
4. 将精简后的历史记忆临时附加到当前用户请求。

```text
平台消息
   │
   ├── chat_memory：原始消息归档 + contexts 接管
   │
   └── livingmemory_cm
          ├── 读取 CM 历史→LLM 萃取→长期记忆写入
          └── 当前问题→长期记忆检索→临时注入
```

## 2. 责任边界

### 2.1 ChatMemory 负责

- 保存 user / assistant / 主动消息。
- 保存 `user_id`、`sender_nickname`、`self_id`、`group_id`、`persona_id`、`turn_id`、`pair_id` 等归因字段。
- 根据 `context_takeover` 配置重建 `ProviderRequest.contexts`。
- 处理 `cross_session`、`full_group`、人格过滤和内容类型过滤。

### 2.2 LivingMemoryCM 负责

- 不再自己累积 conversation messages。
- `ConversationManager` 只保存反思游标等插件元数据。
- 召回时强制按完整 `session_id + persona_id` 过滤长期记忆。
- 注入内容是临时背景，不能反向写回 CM 历史。

## 3. 核心组件

| 组件 | 职责 |
|---|---|
| `LivingMemoryCMPlugin` | AstrBot 生命周期、Hook、命令、Agent 工具和 Page API 入口 |
| `PluginInitializer` | Provider、FaissVecDB、数据库、迁移、引擎和调度器初始化 |
| `EventHandler` | 组合召回、反思和会话重置逻辑 |
| `MemoryRecall` | 构建查询、检索记忆、去重并注入 LLM 请求 |
| `MemoryReflection` | 从 CM 获取待萃取历史，调用 LLM 生成长期记忆 |
| `MemoryProcessor` | 发言者格式化、中性多主题萃取、LLM JSON 解析、记忆标准化 |
| `MemoryEngine` | 兼容 Facade、组件装配和初始化；对外方法签名保持稳定 |
| `DocumentRepository` | 文档读取、幂等键查询、会话记忆分页与 metadata 规范化 |
| `MemorySearchService` | 文档/图双路检索编排、TTL 缓存和访问时间异步更新 |
| `MemoryWriteCoordinator` | `memory_write_ops` 状态机及 add/update/delete/batch-delete 跨存储协调 |
| `MemoryRepairService` | 启动时重放未完成写操作，使文档、原子和图谱副索引收敛 |
| `MemorySchemaService` | 主 SQLite 建表、旧字段补齐、表达式索引、版本初始化和遗留 trigger 清理 |
| `MemoryLifecycleService` | 图索引重建、访问计数、重要性衰减、旧记忆清理和 session 迁移 |
| `MemoryStatisticsService` | 统计聚合、WAL checkpoint、FTS optimize、VACUUM 和文件大小诊断 |
| `FaissBootstrapService` | FAISS 运行时检查、`FaissVecDB` 加载、非 ASCII 路径桥接和安全读写 |
| `EmbeddingIndexBootstrapService` | Provider 指纹、SQLite/FAISS ID 校验、分批影子重建和原子切换 |
| `HybridRetriever` | 文档路：纯向量 + 重要性/时间加权 + MMR |
| `GraphRetriever` | 图路：关键词 + 图向量 + RRF + 时间权重 |
| `DualRouteRetriever` | 文档路和图路最终融合 |
| `AtomLifecycleManager` | 记忆原子过期、遗忘、清理和强化 |
| `DecayScheduler` | 每日衰减、旧记忆清理、备份和存储维护 |

## 4. 启动流程

### 4.1 AstrBot 生命周期入口

AstrBot 完成 Handler 注册后，调用并等待：

```python
await LivingMemoryCMPlugin.initialize()
```

`__init__` 只创建轻量对象，不再 fire-and-forget 启动整个插件。

### 4.2 CM 前置校验

LivingMemoryCM 最多等待 5 次，确认：

```text
chat_memory.ct_enable == true
chat_memory.ct_limit_rounds > 0
```

如果不满足，`initialize()` 直接抛出异常，AstrBot 将插件记为加载失败，不会留下半初始化实例。

### 4.3 版本备份

任何数据库操作之前，`BackupManager` 检查 `.plugin_version`。版本发生变化时，先备份数据文件，再写入新版本号。

### 4.4 Provider 和存储初始化

`PluginInitializer` 依次执行：

1. 在 Provider 等待前对已有 `livingmemory.db` 执行版本 preflight；不支持的旧库先备份再阻断。
2. 解析 Embedding Provider 和 LLM Provider。
3. Provider 就绪后再次检查数据库版本，覆盖等待期间数据库被替换的边界。
4. 加载 FaissVecDB（同时检查 FAISS 运行时及当前 CPU 兼容性）。
5. 在打开正式索引前，只读校验主/图 SQLite 与 FAISS 的 ID 集、数量、维度和 Provider 指纹；必要时分批写入影子索引，完整验证后原子替换。
6. 初始化主文档 FaissVecDB；图记忆开启时再初始化图向量 FaissVecDB。
7. 初始化 `MemoryEngine`、`ConversationManager`、`MemoryProcessor`。
8. 加载停用词。
9. 按配置启动衰减/清理/备份调度器。
10. 核心就绪后注册 Agent 记忆工具和 Plugin Page API。

Provider 在首次 5 秒短等待内未就绪时，初始化器会在后台按退避策略最多重试 60 次。此期间插件已加载但核心未就绪；最终超时会标记初始化失败，修正 Provider 后需要重载插件重新初始化。

## 5. 普通对话时序

```text
CM on_llm_request(priority=-100)
    └── 重写 req.contexts

LM on_llm_request(priority=-200)
    ├── 当前发言 + CM 当前用户最近完整问答构建召回查询
    ├── 检索长期记忆
    └── 追加临时 extra_user_content

Agent / LLM / Tool loop
    └── 形成 Bot 结果

CM on_decorating_result(priority=10)
    └── 写入 assistant prepared 记录

LM on_decorating_result(priority=0)
    └── 检查是否达到长期记忆萃取阈值

Respond / send attempt

LM after_message_sent
    └── 如果是 /reset 或 /new，重置 LM 反思元数据
```

Hook priority 数值越大越早执行。

## 6. 自动召回与注入流程

### 6.1 入口

```python
@filter.on_llm_request(priority=-200)
```

该 Hook 晚于 CM takeover，但召回查询不读取 `req.contexts`。这样 CM 的
`full_group` / `cross_session` 窗口不会改变长期记忆 embedding 查询的主体。

### 6.2 召回查询构建

1. 归一化历史中的纯文本 content parts。
2. 删除旧版 LM 遗留的注入块或伪工具调用。
3. 从 event 提取当前用户原始文本。
4. 如果 `top_k <= 0`，只清理旧注入，不执行召回。
5. 按 AstrBot 正常人格优先级解析 `persona_id`。
6. 调用 CM `query_rounds()`，严格限制当前 UMO、当前 CID、当前 user、当前 persona。
7. 读取该用户最近 `query_context_rounds` 个完整 user/assistant 轮次（默认 2，另受 800 字符预算限制）。
8. 当前发言置于查询首尾，历史问答只作为“那个、继续、上次”等指代消歧信息。

`query_context_rounds=0` 或 `query_context_max_chars=0` 时，只用当前发言检索。
即使 CM 开启 `full_group` 或 `cross_session`，这里也不会混入其他用户或其他会话。

检索范围始终为：

```text
session_id = event.unified_msg_origin
persona_id = 当前 resolved persona
```

本 fork 不允许关闭会话或人格隔离。

### 6.3 召回缓存

缓存 key 包含：

```text
规范化 query + k + session_id + persona_id + 检索路由配置 + cache generation
```

默认 TTL 为 45 秒，最多 256 条。新增、更新或删除记忆后，缓存 generation 变化，旧结果自动失效。

### 6.4 召回后精简

`top_k` 控制检索器返回的候选数量。候选进入注入前还会：

1. 按中文文本词元重合和序列相似度去重。
2. 按原检索排名保留候选。
3. 默认最多注入 3 条。
4. 默认整个记忆块不超过 3200 字符。

### 6.5 默认注入格式

默认 `injection_method=extra_user_content`：

```python
req.extra_user_content_parts.append(
    TextPart(text=memory_text).mark_as_temp()
)
```

它会被追加在当前用户内容后，但 `mark_as_temp()` 使其不进入持久对话历史。

```text
<RAG-Faiss-Memory>
历史信息安全规则

<Memory id="1">
事件时间：...
参与者：...
摘要：...（仅 v3 中性摘要；旧记录有结构化事实时不再优先展示 persona_summary）
事实：
- ...
</Memory>

优先回答当前用户消息的提醒
</RAG-Faiss-Memory>
```

历史记忆中的 XML/HTML 标记会转义，不能提前闭合记忆块。

## 7. 反思萃取流程

反思路径按职责拆分：

```text
event_handler_modules/memory_reflection.py  AstrBot Hook 与配置入口
core/reflection/reflection_service.py       CID/persona/scope、分页和任务编排
core/reflection/cm_history_reader.py        CM 严格 keyset 查询与过滤
core/reflection/cursor_service.py           v2/v3 游标迁移、比较和持久化
core/reflection/extraction_service.py       消息身份转换、LLM 萃取和原子分类
core/reflection/batch_writer.py             幂等主记忆写入与游标提交
```

### 7.1 入口和时机

```python
@filter.on_decorating_result(priority=0)
```

CM 的 `capture_bot(priority=10)` 先把 assistant 写入数据库，LM 再检查反思阈值。因此当前实现以“Bot 结果已组装，CM 已记录 prepared assistant”为触发点。

它不表示平台已确认送达。

### 7.2 阈值

```text
CM llm_status_filter == {"llm_success"}
    → 配对模式，单位为完整 user/assistant 轮

CM 包含 no_llm / proactive / orphan / pending 等其他状态
    → 混合模式，单位为过滤后的消息条数

reflection_engine.trigger_count == 0
    → 数值跟随 CM ct_limit_rounds

reflection_engine.trigger_count > 0
    → 覆盖数值，但不改变上述单位
```

### 7.3 游标分区

反思进度保存在 `conversations.db` 的 session metadata：

```json
{
  "reflection_cursors_v3": {
    "<hashed partition key>": {
      "created_at": "2026-07-19T08:00:00.000000Z",
      "record_id": 12345
    }
  }
}
```

`reflection_cursors_v2` 的纯时间戳值仍会同步保留，便于源码回退；首次读取旧 v2
游标时，会向 CM 查询该时间戳对应的最大记录 ID 并写入 v3。

分区 key 由以下内容组成后做 SHA-256 截断：

```text
conversation_id + persona_id + scope + CM 模式/状态/内容类型签名
```

`scope` 为：

- `full_group`：CM 在群聊中开启 full-group。
- `user:<current_user_id>`：其他情况。

这避免新 conversation、新 persona 或不同用户共用反思进度。

### 7.4 CM 查询

配对模式调用：

```python
chat_memory.query_rounds(
    umo=...,
    conversation_id=...,
    user_id=None if full_group else current_user_id,
    persona_id=current_persona,
    since=cursor,
    after_id=cursor_record_id,
    from_oldest=True,
    content_kind_all_match=cm_all_match,
)
```

通过 ChatMemory 1.1.1 的 `from_oldest=True + since + after_id` 从游标后的最旧完整轮次开始查询，只收集当前触发批次。下界是严格 `(created_at, id)`，即使大量记录时间戳完全相同也不会跨页漏失。

混合模式调用 `chat_memory.query_history()` 使用同一复合游标。ANY/ALL 内容白名单均由 CM 在 SQLite 查询中完成，LM 只做防御性复核；2000 轮 / 4000 条只保留为异常实现的安全保护，正常积压不会先全量扫描再触发保护。

新分区第一次出现时只查询最新一条消息/一轮，以其最新时间建立基线，不扫描也不重新萃取已存在的全部旧历史。

### 7.5 身份格式化

CM 记录转为 LM `Message` 时：

- user 使用 `user_id + sender_nickname`。
- assistant 使用 `self_id`，展示名默认为 `Bot`。
- 保留 `group_id`、`platform`、`turn_id`。
- 根据触发反思的用户标记 `current_user` / `other_user` / `bot`。

群聊 Prompt 看到的文本类似：

```text
[当前发言者: Alice | ID: 10001 | 2026-07-19 20:00:00] ...
[其他发言者: Bob | ID: 10002 | 2026-07-19 20:00:05] ...
[Bot: Bot | ID: 10000 | 2026-07-19 20:00:10] ...
```

### 7.6 LLM 萃取产物

`MemoryProcessor` 不读取 Persona Prompt。`persona_id` 仅用于存储/召回分区。
私聊和群聊 Prompt 都要求 LLM 返回：

```json
{
  "memories": [
    {
      "summary": "中性主题摘要",
      "topics": ["主题"],
      "key_facts": ["具体人物对应的可持久事实"],
      "participants": ["群聊参与者"],
      "event_time": "事件发生时间或时间范围",
      "sentiment": "positive | neutral | negative",
      "importance": 0.7
    }
  ]
}
```

`memories` 可为空，表示这批消息没有持久价值；最多 5 条。主要约束：

- 每条记忆包含 1 至 5 条 `key_facts`。
- 同一连续事件放一条，不同主题或不同时间拆开。
- 使用中性事实表达，不模仿 Persona，不按消息顺序文学化重述。
- 重点提取偏好、身份、关系、约定、计划、决策和后续有用的事实。
- 事实必须归属到具体昵称或 ID。
- `event_time` 是事件时间，不是记忆写入时间。

## 8. 一次萃取会生成什么

一次成功反思产生：

```text
0 至 5 条主题主记忆
  + 每条主记忆各自的 N 条记忆原子（N 约等于有效 key_facts 数）
  + 每条主记忆/原子对应的完整图节点、边和 entries
  + 每条来源主记忆 1 个聚合图向量
```

`memories=[]` 时不写长期记忆，但仍推进该批次游标，避免反复萃取同一段噪声。

### 8.1 主记忆

`content` 使用适合检索的 v3 canonical summary：

```text
事件时间 | 参与者 | 主题 | 事实 | 中性摘要
```

metadata 保存：

```text
session_id
persona_id
importance
create_time / last_access_time
topics
key_facts
participants
event_time
interaction_type
canonical_summary
persona_summary
neutral_summary
summary_quality
source_window
```

新记录不再生成 `persona_summary`。它只为旧数据库和旧 UI 兼容保留；v3 使用
`neutral_summary`，`canonical_summary` 用于向量检索和稳定归档。正常回复时 AstrBot
仍会携带当前 Persona Prompt，因此无需在长期记忆中再保存一份人格化改写。

### 8.2 记忆原子

每个 `key_fact` 通过规则分类为：

- `EPISODIC`：事件型。
- `PLANNED`：计划/截止日期型。
- `FACTUAL`：稳定事实型。
- `RELATIONAL`：关系型。
- `PREFERENCE`：偏好型。

原子拥有独立的：

```text
TTL
衰减类型
重要性
置信度
事件时间
强化次数
session/persona 范围
```

### 8.3 图谱

开启图记忆时，`GraphExtractor` 优先从原子生成图数据，否则从主记忆 metadata 生成。

图数据包括：

- topic / participant / fact 节点。
- 记忆与节点的关联。
- 跨记忆语义边。
- 边置信度 EMA 和证据权重。
- 每条来源主记忆一个聚合向量文档；代表 entry 保存 `vector_doc_id`，其余 entries 不重复生成向量。
- `/lmem status` 会识别旧的“每 entry 一个向量”格式；迁移由 `/lmem rebuild-graph` 显式触发。

### 8.4 多存储写入一致性

`MemoryEngine` 的兼容入口委托 `MemoryWriteCoordinator`。协调器写入前创建
`memory_write_ops` 操作日志，分步标记：

```text
started
→ document_indexed
→ atoms_indexed / atoms_partial / atoms_skipped
→ graph_indexed / graph_failed / graph_skipped
→ completed / needs_repair / failed
```

插件下次启动时由 `MemoryRepairService` 尝试修复未完成的多存储写入。

## 9. 检索架构

### 9.1 文档路

文档路是：

```text
查询向量
→ Faiss 向量召回
→ session/persona 过滤
→ 相关性 + importance + recency 加权
→ MMR 多样性去重
```

默认加权：

```text
0.50 * 向量相关性
+ 0.25 * 重要性
+ 0.25 * 时间新鲜度
```

文档路已删除 BM25，`HybridRetriever` 是为减少下游改动而保留的历史类名。

### 9.2 图路

```text
图关键词检索 ─┐
                 ├→ RRF 融合→importance/recency/confidence/原子时间衰减
图向量检索   ─┘
```

图关键词路会从命中节点向相邻节点扩展，支持一跳或二跳关系。

### 9.3 文档路 + 图路

`DualRouteRetriever` 并行执行两路，默认权重：

```text
文档路 0.65
图路   0.35
双路同时命中同一记忆：+0.08 bonus
```

开启 `dynamic_route_weighting` 时：

- 关系查询提高图路权重。
- 时间查询适度提高图路权重。
- 定义/解释查询提高文档路权重。

### 9.4 原子检索的当前状态

`AtomRetriever` 和原子 FTS 已初始化，用于原子级数据管理和时间评分。

但当前自动 `MemoryEngine.search_memories()` 的主路径是“文档路 + 图路”，没有把 `AtomRetriever.search()` 作为第三路直接融入最终召回。

原子目前主要通过以下方式影响召回：

- 作为图谱抽取的细粒度输入。
- 为图路提供 TTL、衰减和置信度信号。
- 通过生命周期管理影响图记忆的长期存活。

## 10. Agent 主动工具

### 10.1 `recall_long_term_memory`

默认开启。Agent 传入简短查询词和 `k`，工具会：

1. 获取当前 event。
2. 解析当前 session/persona。
3. 调用与自动召回相同的 `MemoryEngine.search_memories()`。
4. 返回 JSON 记忆列表给 Agent tool loop。

该工具的返回值不直接发给用户，而是作为 `role=tool` 内容进入下一轮 Agent 推理。

### 10.2 `memorize_long_term_memory`

默认关闭。开启后，Agent 可以提交：

```text
memory
topics
key_facts
sentiment
importance
reason
```

工具会复用 `MemoryProcessor.build_memory_from_structured_data()` 和标准 `MemoryEngine.add_memory()` 写入流程。

## 11. 命令和 WebUI

### 11.1 命令

所有 `/lmem` 子命令当前都要求 Bot 管理员权限：

| 命令 | 作用 |
|---|---|
| `/lmem status` | 引擎、数据和索引状态 |
| `/lmem search <query> [k]` | 手动检索记忆 |
| `/lmem forget <id>` | 删除主记忆及关联原子/图数据 |
| `/lmem rebuild-graph` | 影子重建图谱并迁移为每来源记忆一个图向量 |
| `/lmem webui` | 输出 Plugin Page 入口说明 |
| `/lmem reset` | 重置当前 LM 会话元数据 |
| `/lmem cleanup` | 清理历史中的旧版记忆注入块 |

### 11.2 Plugin Page API

路由通过 `Context.register_web_api()` 注册，Dashboard 实际访问路径由 AstrBot Bridge 转换为 `/api/plug/livingmemory_cm/...`。

主要能力：

- 统计概览。
- 记忆列表、详情、更新、批量更新和删除。
- 召回测试。
- 图谱概览和图查询。
- 备份列表。
- persona 列表。

Plugin Page 前端通过 `window.AstrBotPluginPage.apiGet/apiPost` 使用 AstrBot 登录态，插件不自行拼接鉴权 token。

## 12. 每日维护

默认执行时间为每日 `00:05`。启动时会先检查是否错过了前几天的衰减。

流程：

```text
重要性衰减
→ 按年龄 + 重要性阈值清理主记忆
→ 数据库备份
→ 清理超过保留天数的备份
→ checkpoint / VACUUM 等存储维护
```

另外，`AtomLifecycleManager` 按独立间隔运行：

```text
active 过期
→ expired
→ 延迟遗忘并移出检索索引
→ forgotten
→ 延迟物理删除
```

## 13. 数据文件

| 文件 | 内容 |
|---|---|
| `livingmemory.db` | 主记忆辅助表、写操作日志、图表、原子表等 |
| `livingmemory.index` | 主记忆 FAISS 向量索引 |
| `livingmemory_graph_documents.db` | 图向量文档存储 |
| `livingmemory_graph.index` | 图向量 FAISS 索引 |
| `conversations.db` | LM 会话元数据、canonical `reflection_cursors_v3` 和 v2 回退兼容游标 |
| `decay_state.json` | 上次每日衰减日期 |
| `.plugin_version` | 已运行的插件版本，用于触发升级备份 |
| `backups/` | 版本备份和每日数据库备份 |

这些文件属于运行数据，不应同步回插件源码仓库。

## 14. 关闭与重载

`LivingMemoryCMPlugin.terminate()` 直接定义在最终 Star 类上。关闭顺序：

1. 设置 `_terminating`，拒绝创建新运行期组件。
2. 取消插件层后台任务。
3. 停止 Provider 初始化重试。
4. 等待已派发的记忆萃取任务结束。
5. 停止每日衰减调度器和原子生命周期任务。
6. 关闭 ConversationStore、MemoryEngine、FaissVecDB 和图向量库。

## 15. 失败语义

### 15.1 召回失败

- 记录异常并跳过本轮长期记忆注入。
- 不阻断正常 LLM 对话。

### 15.2 萃取 LLM 失败

- 不写入主记忆。
- 不推进反思游标。
- 下次触发可重试同一批消息。
- LLM 合法返回零条记忆时视为成功批次并推进游标，避免噪声窗口永久重试。

### 15.3 主记忆已写入，游标更新失败

- 主记忆保留。
- 反思窗口使用稳定 `batch_id`，每条记忆使用独立幂等键。
- 下次重试会复用已写入的记忆 ID，不重复落库；全部条目确认后再推进游标。

### 15.4 原子或图谱写入失败

- 主记忆不回滚删除。
- `memory_write_ops` 标记 `needs_repair`。
- 下次启动尝试修复未完成的副存储。

## 16. 当前边界与已确认语义

### 16.1 `full_group` 的反思单位

当前实现跟随 CM 自身的配对/混合模式，而不是仅由 `full_group` 开关决定：

```text
llm_status_filter == {"llm_success"}
    → query_rounds()，按完整 user/assistant 轮数

llm_status_filter 包含其他状态
    → query_history()，按过滤后的消息条数
```

`full_group=true` 时，混合模式的查询范围为当前 UMO + CID 下整群消息（`user_id=None`）；
非 full-group 时仍限制当前用户。`trigger_count` 只覆盖数量，不改变单位。

### 16.2 Bot 昵称

CM 保存 Bot `self_id`，但当前反思转换时没有稳定的 Bot 昵称来源，因此展示名使用 `Bot`。身份 ID 是准确的。

### 16.3 真实平台送达

LM 反思触发点位于实际发送之前。AstrBot `after_message_sent` 也只能表示发送流程已尝试，不是平台 delivery acknowledgement。

### 16.4 日志隐私

默认日志只记录数量、长度、状态、游标和稳定短引用，不记录用户查询、记忆正文、完整对话、System Prompt 或 LLM 原始响应。session、UMO、persona、sender 和 Bot platform ID 均先转换为不可直接识别的短引用。

## 17. 建议的调试入口

### 召回问题

1. 确认 CM takeover 已先于 LM Hook 运行。
2. 检查本轮 `session_id` 和 resolved `persona_id`。
3. 查看文档路/图路 score breakdown。
4. 检查召回缓存是否命中。
5. 确认候选是否被相似度去重或 3200 字符预算过滤。

### 萃取问题

1. 确认 CM 当前 CID 存在。
2. 确认 CM 当前 persona 与 LM 游标分区一致。
3. 优先检查 `reflection_cursors_v3`，再核对 `reflection_cursors_v2` 兼容值。
4. 检查 CM `llm_status_filter` 和内容类型排除。
5. 检查 LLM 输出是否包含有效 JSON、具体发言者和 `event_time`。
6. 检查 `memory_write_ops` 是否存在 `needs_repair`。

### 数据验证

- 读取真实 SQLite 数据时优先使用 SQLite backup API 创建临时快照。
- 不把真实账号、昵称、persona ID 或消息内容写入测试、文档或日志摘要。
- 本地单元测试不代表平台发送、Provider 响应或 Plugin Page Bridge 已完成真实集成验证。

## 18. 部署流程

```text
源码工作树
    │  完整本地代码验收
    │
    └── Release ZIP 或 sync.sh（排除 .git / 缓存 / db / data）
          ↓
AstrBot data/plugins/livingmemory_cm
    │
    ├── AstrBot 插件加载/重载
    ├── 真实 Provider / CM / FAISS / Plugin Page 验证
    └── 运行数据位于 plugin_data/livingmemory_cm
```

同步源码时禁止复制 `.git`，也不应用源码目录覆盖 `plugin_data` 中的真实运行数据。

## 19. 当前验证边界

本地自动化只覆盖函数、模块、存储协议、反思复合游标与大积压、批次幂等、迁移与备份、
MemoryEngine 分层服务、命令/Page API 处理器、图谱、原子、召回和注入数据协议。数据库测试
使用临时目录，不读取或修改部署端 `plugin_data` 中的运行数据库。

AstrBot 生命周期、Hook 调度、真实 Provider、CM 在线消息触发和平台发送由远端集成环境验证；
既有 FAISS 数据、Plugin Page Bridge 与实际使用体验由部署端验收。
