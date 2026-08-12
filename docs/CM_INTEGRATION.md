# LivingMemoryCM 与 ChatMemory 集成说明

> 适用版本：LivingMemoryCM `2.5.7-cm`、ChatMemory `>= 1.1.1`。
> 本文只说明 CM 集成契约；配置项的默认值和调优建议见
> [CONFIGURATION.md](CONFIGURATION.md)，完整运行流程见
> [ARCHITECTURE.md](ARCHITECTURE.md)。

## 1. 集成结论

ChatMemory（下文简称 CM）是启动级硬依赖，不是可选增强：

- 插件注册名必须为 `chat_memory`；
- `context_takeover.enable` 必须开启；
- `context_takeover.limit_rounds` 必须大于 `0`；
- CM 必须提供兼容的 `query_rounds()`、`query_history()` 和严格复合游标参数。

LivingMemoryCM 不提供无 CM 回退模式。启动门禁失败时会停止初始化，避免两个插件同时
维护短期上下文或长期记忆读取不完整。

## 2. 责任与数据边界

```text
平台消息
  ├─ ChatMemory
  │    ├─ 归档原始 user/assistant 消息
  │    └─ 接管 Provider 短期 contexts
  └─ LivingMemoryCM
       ├─ 从 CM 查询近期问答，辅助长期记忆召回
       ├─ 从 CM 按游标读取已完成历史，萃取长期记忆
       └─ 写入自己的 SQLite、FAISS、图数据库和记忆原子
```

CM 负责原始消息、短期历史、LLM 状态、内容类型、配对轮次和 CM 自己的人格标记。
LivingMemoryCM 负责长期记忆正文、主向量、图谱、记忆原子、反思游标、衰减和备份。

LivingMemoryCM 只读调用 CM 查询 API，不写 `chat_memory.db`。自身
`conversations.db` 中保留的旧 `messages` 表仅用于旧数据兼容和清理命令；CM-only
正常路径不再读写该表。

## 3. 启动门禁与 Hook 顺序

`main.py::_initialize_plugin()` 会有限次数等待 CM，然后通过
`core/utils/cm_bridge.py` 解析插件实例并检查 `ct_enable`、`ct_limit_rounds`。
为兼容 AstrBot 的插件包装差异，桥接会从常见实例包装字段中寻找真实对象。

关键 Hook 顺序为：

| 阶段 | ChatMemory | LivingMemoryCM | 目的 |
| --- | ---: | ---: | --- |
| `on_llm_request` | 先接管短期 contexts | `priority=-200` 召回长期记忆 | 避免长期记忆覆盖 CM 短期历史 |
| `on_decorating_result` | 先归档本轮 assistant | `priority=0` 执行反思 | 反思能读取完整问答 |

本地单元测试不模拟 AstrBot 调度器。每次升级 AstrBot 或 CM 后，都应在部署端确认实际
Hook 顺序、Provider 调用和消息归档结果。

## 4. 运行时接口契约

LivingMemoryCM 直接使用以下 CM 成员：

| 成员 | 用途 |
| --- | --- |
| `ct_enable`、`ct_limit_rounds` | 启动门禁；反思默认阈值 |
| `ct_llm_status_filter` | 决定按完整轮次还是按消息萃取 |
| `ct_full_group` | 群聊反思使用当前用户或整群范围 |
| `ct_include_kinds`、`ct_include_all_match` | 内容类型过滤及 ALL/ANY 语义 |
| `_is_group_umo()` | 判断群聊；缺失时使用受限字符串兜底 |
| `query_rounds()` | 召回消歧和完整轮次反思 |
| `query_history()` | 消息模式反思 |

`_is_group_umo()` 是唯一使用到的 CM 私有辅助方法，升级 CM 时应重点检查。CM 的
`cross_session`、`max_context_chars`、`clear_native_history`、
`fallback_to_native_on_empty` 和 `filter_by_persona` 会影响 CM 自己构造的短期上下文，
但不会扩大 LivingMemoryCM 的长期记忆查询范围。

CM 查询结果需要保留这些语义字段：

- `record_id`、`created_at_utc` 或 `created_at`：严格排序和游标；
- `role`、`content`、`turn_id`：问答还原；
- `user_id`/`sender_id`、昵称字段、`self_id`、`group_id`、平台字段：稳定身份；
- `content_kind`：内容类型过滤。

## 5. 召回和反思

自动召回只查询当前 UMO、conversation、user 和 persona 下最近的完整问答，用于解决
“那个”“继续”等指代；当前发言始终是检索主体。运行过程中 CM 临时不可用时，召回
可退回仅使用当前发言，但这只是容错，不代表插件支持脱离 CM 启动。

反思不保留自己的原始消息归档，必须从 CM 读取：

- 状态过滤恰好只有 `llm_success` 时按完整 user/assistant 轮次萃取；
- 包含其他状态时按过滤后的消息数量萃取；
- 群聊 `full_group` 决定当前用户或整群范围；
- 查询始终显式传入当前 conversation 和 persona，不跟随 `cross_session` 扩展。

反思使用严格 `(created_at, record_id)` keyset 游标，依赖 `since`、`until`、
`from_oldest`、`after_id`、`content_kind_all_match` 和 `persona_id` 等查询参数。
长期记忆的 `metadata.source_window` 会记录模式、时间窗口、record ID、消息数和稳定
`batch_id`；主记忆已写入但游标更新失败时，重试不会重复落库。

## 6. 配置变化与游标

反思 v3 游标分区包含 conversation、persona、当前用户/整群 scope、按轮/按消息模式、
状态白名单、内容类型白名单和 ALL/ANY 模式。

修改 CM 的 `full_group`、`llm_status_filter`、`include_content_kinds` 或
`include_all_match` 后会形成新分区。新分区首次只以当前最新记录建立基线，不会把旧
历史从头萃取，从而避免配置切换后产生重复长期记忆。需要重放旧历史时必须设计显式
迁移，不能只修改配置。

LivingMemoryCM 的 8 个配置段中，只有以下部分直接关联 CM：

- `recall_engine.query_context_*`：从 CM 读取近期问答用于消歧；
- `reflection_engine.trigger_count`：`0` 时数值和有效单位跟随 CM。

其他召回、图谱、衰减、维护、Agent 工具和日志配置均控制 LivingMemoryCM 自身。

## 7. 升级与数据安全

- Release 不包含 `data/`；升级时保留部署端两边的数据目录，并先做离线备份。
- LivingMemoryCM 版本变化会先创建自身数据备份，不会修改 CM 数据库。
- 旧配置会与当前默认值递归合并；新增字段缺失时采用默认值。
- 配置类型或范围错误会导致整份生效配置回退默认值，升级后应检查验证警告。
- 主长期记忆数据库 schema 保持兼容；索引代际或 Embedding 指纹变化通过影子索引
  重建，验证成功后才切换。
- 旧 entry 级图向量不会静默删除；`/lmem status` 会提示并由
  `/lmem rebuild-graph` 显式迁移为记忆级图向量。

## 8. CM 升级回归清单

1. 注册名仍为 `chat_memory`，实例包装仍可被桥接解析。
2. 上述 `ct_*` 属性和 `_is_group_umo()` 仍兼容。
3. `query_rounds()`、`query_history()` 的参数与返回字段保持兼容。
4. `record_id` 仍稳定递增，时间戳可按 UTC 解析。
5. CM 先归档本轮 assistant，LivingMemoryCM 再执行反思。
6. 召回仍限定当前 UMO、conversation、user 和 persona。
7. 切换 CM 过滤规则后只建立新游标基线，不重复萃取旧历史。
8. 完成 AstrBot reload、真实 Provider、平台消息和 Dashboard 远端验收。

## 9. 主要源码位置

- `main.py`：启动门禁与 Hook；
- `core/utils/cm_bridge.py`：CM 实例解析和状态读取；
- `core/event_handler_modules/memory_recall.py`：近期问答辅助召回；
- `core/event_handler_modules/memory_reflection.py`：反思入口和运行期门禁；
- `core/reflection/`：模式、分页、复合游标和幂等批次；
- `storage/conversation_store.py`：CM-only 会话边界；
- `core/base/config_validator.py`：配置合并与校验。
