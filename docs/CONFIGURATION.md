# LivingMemoryCM 配置说明

> 适用版本：`2.5.7-cm`。当前共有 8 个配置段、45 个配置项。
> CM 自身的 `context_takeover` 配置不在本文件重复展开；两者的耦合关系见
> [CM_INTEGRATION.md](CM_INTEGRATION.md)。

## 1. 配置加载规则

插件会先把已有用户配置与当前默认值递归合并，再进行类型和范围校验。因此旧版本
配置缺少新字段时，会自动补默认值，不需要手工迁移。

有两个需要特别注意的行为：

1. 任一配置值类型错误或越界时，当前实现会把**整份生效配置**降级为默认配置，
   不是只修正错误字段。升级或改配置后应检查日志中是否出现“配置验证失败”。
2. 大部分配置在插件初始化时读取。修改后应 reload 插件；不要假定已经运行的检索器、
   图谱管理器或定时任务会热更新。

## 2. 模型提供商 `provider_settings`

| 配置 | 默认值 | 作用与建议 |
|---|---:|---|
| `embedding_provider_id` | 空 | 生成主记忆和图记忆向量。留空时使用 AstrBot 返回的第一个可用 Embedding Provider。更换 Provider、模型、端点或向量维度会触发索引代际检查；必要时安全重建。 |
| `llm_provider_id` | 空 | 用于中性事实萃取、主题拆分和重要性评估。留空时使用 AstrBot 当前默认聊天 Provider。建议选择结构化输出稳定、中文理解较好的模型。 |

两个 Provider 都是插件核心依赖。初始化重试结束后仍无法取得其中任意一个，插件核心
组件会初始化失败。Provider ID 必须填 AstrBot 后台中已有的 ID，而不是模型显示名称。

## 3. 记忆召回 `recall_engine`

### 3.1 返回数量与注入方式

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `top_k` | `5`；0～50 | 自动召回最多返回多少条候选。`0` 会跳过自动检索和注入，但仍会清理请求中残留的旧注入片段。通常 3～8 足够。 |
| `max_k` | `10`；1～50 | Agent 调用 `recall_long_term_memory` 时允许请求的最大数量，只限制主动工具，不改变普通自动召回的 `top_k`。 |
| `injection_method` | `extra_user_content` | 决定长期记忆放进 Provider 请求的哪个位置，见下表。 |
| `injection_max_memories` | `3`；1～10 | 候选去重后，单次真正写入 Prompt 的最大记忆数。它可以小于 `top_k`，用来“多召回、少注入”。 |
| `injection_max_chars` | `3200`；500～12000 | 单次注入的总字符预算，包含规则、参与者、摘要和关键事实。低排名记忆会在超预算时被舍弃。 |

`injection_method` 的可选值：

| 值 | 行为 | 建议 |
|---|---|---|
| `extra_user_content` | 作为临时附加内容追加到本轮用户消息末尾；不进入持久历史，也尽量不破坏 Provider 前缀缓存 | 默认推荐 |
| `user_message_before` | 直接加到 `req.prompt` 的用户正文前 | 记忆优先级更显眼，但会改变用户消息正文 |
| `user_message_after` | 直接加到 `req.prompt` 的用户正文后 | 语义直观，但同样会改变用户消息正文 |
| `fake_tool_call` | 在 `req.contexts` 中伪造一次检索工具调用及返回 | 部分模型理解自然；Gemini/Google GenAI 会自动降级为 `extra_user_content` |

`top_k` 控制“检索多少”，`injection_max_memories` 和 `injection_max_chars` 控制“最终
给模型看多少”。例如 `top_k=8`、`injection_max_memories=3` 会先从更多候选中排序，
最后最多注入 3 条。

### 3.2 检索缓存

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `search_cache_ttl_seconds` | `45.0`；0～600 | 相同会话、persona、查询和检索策略在 TTL 内复用结果。`0` 关闭缓存。写入、更新或删除记忆时会主动失效缓存。 |
| `search_cache_max_size` | `256`；0～10000 | 进程内最多保留的检索结果条目数，超出后淘汰最久未使用项。`0` 也等价于关闭缓存。 |

缓存只减少短时间重复查询的 SQLite/FAISS 开销，不会持久化到磁盘。调试排序变化时可
暂时把 TTL 设为 0；正常使用保留默认值即可。

### 3.3 CM 问答消歧

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `query_context_rounds` | `2`；0～10 | 从 CM 读取当前用户、当前 UMO/CID/persona 下最近 N 个完整 `llm_success` 问答轮次，用于理解“那个”“继续”。`0` 只用当前发言。 |
| `query_context_max_chars` | `800`；0～4000 | 上述历史问答可占的字符上限。`0` 表示完全不加入历史，即使 `query_context_rounds > 0`。 |
| `query_context_max_age_seconds` | `0`；0～31,536,000 | 只读取该时间窗口内的 CM 问答。`0` 不限制年龄；例如 `7200` 表示最近 2 小时。 |

这些历史只参与构造 Embedding 检索查询，不会作为长期记忆保存，也不会跟随 CM 的
`cross_session` 或 `full_group` 扩大范围。当前发言始终放在查询首尾，保持本轮意图为
主体。

### 3.4 召回过滤与近期槽位

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `min_importance_for_retrieval` | `0.0`；0～1 | 排除重要性低于阈值的候选。缺少重要性元数据时按 0.5 处理；`0` 不过滤。 |
| `min_similarity_for_retrieval` | `0.0`；0～1 | 当候选存在文档/图向量信号时，若所有可用向量信号都低于阈值则排除。纯关键词和无向量分数的近期候选不受影响；`0` 不过滤。 |
| `recent_memory_count` | `0`；0～20 | 在最终 top-k 内，为当前 session/persona 的近期 `active` 记忆保留最多 N 个槽位。`0` 关闭。 |
| `recent_memory_max_age_hours` | `72`；0～8760 | 近期槽位只读取该小时窗口内的记忆。`0` 不限时间；`recent_memory_count=0` 时本项不生效。 |
| `memory_type_filter` | `all` | `all` 不按类型过滤；`event_only` 只保留带有 `episodic`、`planned` 或 `factual` 原子类型的明确事件类记忆。 |

`event_only` 为兼容旧数据采取保守规则：旧记忆没有 `atom_types` 时仍可召回；只有
已经存在非空类型列表、且与三种事件类型完全不相交时才排除。

近期槽位不会增加最终条数。例如 `top_k=5`、`recent_memory_count=2` 仍最多返回 5 条，
只是优先确保最多 2 条符合范围的近期记忆进入结果。

## 4. 重要性衰减 `importance_decay`

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `decay_rate` | `0.01`；0～1 | 每日基础重要性衰减率。也用于文档路和图路排序中的时间新鲜度权重 `exp(-rate × 天数)`；`0` 同时关闭每日重要性衰减并取消这部分按年龄降权。 |
| `access_decay_window_days` | `30.0`；1～3650 | 判断一次访问是否属于“近期访问”的窗口。窗口内访问对衰减的保护更强，窗口外仍保留一半的访问保护因子。 |
| `access_decay_max_count` | `10`；1～10000 | 访问多少次达到最大衰减保护。计数超过该值不会继续增强保护。 |
| `access_count_decay_multiplier` | `0.5`；0～1 | 每次执行每日衰减后，访问次数乘以该比例并取整数，避免旧热点永久获得最大保护。`0` 每次清零，`1` 永不回落。 |
| `protected_importance_threshold` | `1.0`；0～1 | 重要性大于等于该阈值的记忆完全跳过每日重要性衰减。默认只保护满分记忆。 |

访问强化后的实际衰减率会低于 `decay_rate`，但访问不会让重要性反向增长。
`protected_importance_threshold` 只影响每日衰减，不阻止管理员删除，也不绕过
`maintenance` 的自动清理条件。

兼容性提示：旧 `2.3.6-cm.12` 会衰减重要性恰好为 1.0 的记忆；当前默认阈值会
保护它们。由于允许范围最大为 1.0，当前配置不能表达“仅让 1.0 继续衰减”；把阈值
调低只会保护更多记忆。

## 5. 记忆生成 `reflection_engine`

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `trigger_count` | `0`；0～2000 | 多少个 CM 单位后触发一次长期记忆萃取。`0` 跟随 CM `context_takeover.limit_rounds` 的数值；正数覆盖数值，但不覆盖单位。 |

单位由 CM 的状态过滤决定：

- `llm_status_filter` 恰好只有 `llm_success`：按完整 user/assistant **轮数**；
- 还包含其他状态：按过滤后的**消息条数**。

数值越小，萃取更及时但 LLM 调用更频繁、上下文更碎；数值越大，主题上下文更完整，
但长期记忆生成延迟和单次 Prompt 都会增加。修改 CM scope、状态或内容白名单会建立新
反思游标分区，并以最新记录为基线跳过旧历史，详见 CM 耦合文档。

## 6. Agent 主动工具 `agent_tools`

| 配置 | 默认值 | 作用与建议 |
|---|---:|---|
| `enable_recall_tool` | `true` | 注册 `recall_long_term_memory`。Agent 可在自动召回之外主动搜索；仍受当前 session/persona 隔离、召回过滤和 `max_k` 限制。 |
| `enable_memorize_tool` | `false` | 注册 `memorize_long_term_memory`，允许 Agent 主动写入长期记忆。写入能力影响数据，应只对可信模型和可控工具权限开启。 |

关闭工具不会关闭自动召回或自动反思，只是不再向 Agent 暴露对应函数。

## 7. 图记忆 `graph_memory`

### 7.1 总开关和双路融合

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `enabled` | `true` | 初始化图 SQLite、图向量、图关键词检索和双路融合。关闭后只走主文档向量路；已有图数据不会被删除。 |
| `document_route_weight` | `0.65`；0～1 | 文档向量路在最终双路融合中的基础权重。 |
| `graph_route_weight` | `0.35`；0～1 | 图路在最终双路融合中的基础权重。 |
| `cross_route_bonus` | `0.08`；0～0.5 | 同一记忆同时被文档路和图路命中时追加的得分。值过大可能让“双命中”压过更高质量的单路结果。 |
| `dynamic_route_weighting` | `true` | 根据查询中的关系、时间、定义/操作类词语动态偏向图路或文档路；关闭后始终使用固定基础权重。 |

文档权重与图权重会在验证阶段归一化，使两者之和为 1；若两者都为 0，会自动恢复
为 `0.65 / 0.35`。动态权重会在此基础上调整：关系问题更偏图路，时间问题略偏图路，
定义和“如何做”类问题更偏文档路。

关闭 `graph_memory.enabled` 后，下方图扩展、图节点上限和原子生命周期配置不会参与
运行，但主长期记忆和主向量索引仍正常工作。

### 7.2 图检索扩展

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `expansion_limit` | `24`；1～200 | 从图关键词命中的节点向邻居扩展时，控制候选规模。越大越可能找到间接关系，也增加 SQLite 查询和排序开销。 |
| `expansion_hops` | `1`；1～2 | 图查询向外扩展几跳。`2` 能发现间接关联，但噪声和开销更高。 |
| `second_hop_weight` | `0.4`；0～1 | 二跳候选相对一跳候选的权重。只有 `expansion_hops=2` 时生效。 |

一般先保持一跳；只有确实需要“某人与某主题再关联到另一事实”一类间接关系时，才
考虑启用二跳。

### 7.3 每条记忆的图结构上限

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `max_topics_per_memory` | `6`；1～20 | 每条来源记忆最多写入多少个主题节点。 |
| `max_participants_per_memory` | `8`；1～30 | 每条来源记忆最多写入多少个参与者节点。 |
| `max_facts_per_memory` | `8`；1～30 | 每条来源记忆最多写入多少个事实节点/entry；没有关键事实时会以摘要兼容回退。 |

这些限制用于防止一次萃取把图谱膨胀得过快。修改后主要影响新写入或重新构建的图；
若希望现有记忆按新上限重建，执行 `/lmem rebuild-graph`。重建会产生 Embedding 调用，
但采用影子图，失败不会替换正式图。

### 7.4 记忆原子生命周期

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `atom_enabled` | `true` | 把每条 `key_fact` 分类为独立记忆原子，并启用原子检索、强化、TTL 和生命周期维护。关闭后保留粗粒度主记忆/图路径；已有原子数据不会立即删除。 |
| `atom_maintenance_interval_hours` | `24.0`；1～168 | 原子生命周期任务运行间隔。每轮依次执行过期、软遗忘和物理清理。 |
| `atom_forget_delay_days` | `7.0`；1～90 | 原子过期后等待多少天再软遗忘；软遗忘会让其退出检索，但保留元数据。 |
| `atom_purge_delay_days` | `30.0`；1～365 | 已软遗忘原子再等待多久后从数据库物理清理。通常应大于遗忘延迟。 |

配置校验目前没有强制 `atom_purge_delay_days >= atom_forget_delay_days`，但实际使用
建议保持该关系，形成“先退出检索、后物理删除”的恢复窗口。

## 8. 维护任务 `maintenance`

| 配置 | 默认值 / 范围 | 作用与建议 |
|---|---|---|
| `cleanup_days_threshold` | `30`；0～3650 | 创建时间超过该天数、且重要性低于下项阈值的记忆会被自动删除。`0` 关闭自动清理。 |
| `cleanup_importance_threshold` | `0.3`；0～1 | 自动清理的重要性严格上限；只有 `importance < threshold` 才删除。`cleanup_days_threshold=0` 时不生效。 |
| `backup_keep_days` | `7`；0～365 | 启用每日 `livingmemory.db` SQLite 备份，并清除超过该天数的同类每日备份。`0` 关闭每日备份。 |

每日顺序是：重要性衰减 → 旧记忆清理 → 主 SQLite 备份 → 存储维护。

当前实现有一个组合限制：调度器只在 `decay_rate > 0` 或
`cleanup_days_threshold > 0` 时启动。因此如果这两项都设为 0，即使
`backup_keep_days > 0`，每日备份也不会执行。若需要“只备份、不衰减、不清理”，
当前需要代码调整，不能只靠配置实现。

这里的每日备份与版本变更备份是两套机制：

- `backup_keep_days` 管理 `backups/livingmemory_backup_*.db`，只备份主 SQLite；
- 每次检测到插件版本变化时，`BackupManager` 会创建 `backups/v<旧版本>/`，保存主库、
  会话库、FAISS、图索引和衰减状态等完整数据；这些版本目录不受
  `backup_keep_days` 清理。

Dashboard“系统概览”中看到的多个 `backup` 通常是后者，即不同插件版本留下的完整
升级备份，不是重复的记忆记录。

## 9. 日志 `log`

| 配置 | 默认值 | 作用与建议 |
|---|---:|---|
| `debug_to_info` | `false` | 把本插件的 debug 日志以 info 级别输出，无需修改 AstrBot 全局日志级别。排障时临时开启，长期开启会明显增加日志量。 |
| `log_with_bot_id` | `false` | 消息事件日志前缀附加脱敏后的 bot/platform 稳定短标识，方便多 Bot 环境区分实例。不会输出完整平台 ID。 |

插件日志已避免记录用户完整查询、记忆正文、LLM 原始输出和 Provider 密钥；开启
debug 仍建议按敏感运行数据管理日志文件。

## 10. 常用调整方案

### 保守升级，尽量保持旧 cm.12 召回

保留默认的新增策略：重要性/相似度阈值为 0、近期槽位为 0、类型为 `all`、CM 历史
年龄为 0。唯一明确变化是满分重要性记忆受 `protected_importance_threshold=1.0`
保护，详见上文兼容性提示。

### 减少 Token 和检索成本

优先依次降低：

1. `injection_max_chars`；
2. `injection_max_memories`；
3. `query_context_max_chars` / `query_context_rounds`；
4. `top_k`。

不要一开始就提高相似度阈值，因为不同 Embedding Provider 的分数分布不完全一致。

### 更重视最近发生的事

可以先设置 `recent_memory_count=1` 或 `2`，保留 `recent_memory_max_age_hours=72`。
这比大幅提高 `decay_rate` 更可控，因为后者还会改变每日重要性和所有候选的时间排序。

### 图谱太大或检索偏慢

优先保持 `expansion_hops=1`，再降低 `expansion_limit`；若新记忆本身产生节点过多，
再降低 topics/participants/facts 三个上限并择机重建图谱。
