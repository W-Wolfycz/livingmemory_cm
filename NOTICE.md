# LivingMemoryCM 来源、修改与分发声明

LivingMemoryCM 是
[astrbot_plugin_livingmemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory)
的修改版，不是上游官方发行版。

- 原项目：LivingMemory / `astrbot_plugin_livingmemory`
- 原项目作者：`lxfight` 及上游贡献者
- 当前 fork 维护者：`Wolfycz`
- 当前 fork 源码：<https://github.com/W-Wolfycz/livingmemory_cm>
- 修改版权：Copyright (C) 2026 Wolfycz
- 修改期间：2026 年 3 月至 2026 年 8 月（后续修改以 CHANGELOG 为准）
- 许可证：GNU Affero General Public License v3.0（AGPL-3.0）

本项目根目录的 [`LICENSE`](LICENSE) 是适用于整个派生作品的完整许可证正文。本文件
用于履行修改版的来源、署名和显著修改说明，不替代许可证正文。

## 主要修改

相对上游，本 fork 的核心差异包括：

- 改为 CM-only 单路径，运行时要求 `chat_memory` 开启 `context_takeover`；
- 将原始消息归档和短期上下文交给 ChatMemory，LivingMemory 只维护长期记忆；
- 删除文档路 BM25，保留主文档向量路和图关键词/图向量融合；
- 使用 ChatMemory 严格复合游标完成反思分页、幂等写入和 persona/session 隔离；
- 追踪上游 2.5.7 的可移植数据安全、图向量、召回策略和 Dashboard 能力；
- Dashboard 改为中文单语言并保留 CM persona 适配；
- 使用独立插件名 `LivingMemoryCM` 和 `-cm` 版本标识，避免与上游官方版本混淆。

详细修改和日期见 [`CHANGELOG.md`](CHANGELOG.md)，架构与 CM 接口边界见
[`docs/CM_INTEGRATION.md`](docs/CM_INTEGRATION.md)。

## 分发与网络使用

AGPL-3.0 允许使用、修改、复制和收费或免费分发本项目。分发修改版时至少应：

1. 保留 `LICENSE`、本 `NOTICE.md`、上游来源和已有版权/许可证声明；
2. 显著标明修改者、修改日期和修改版身份，不暗示上游为本 fork 提供官方支持；
3. 将整个派生作品继续按 AGPL-3.0 提供，不增加限制下游复制、修改或再分发的条款；
4. 提供与所分发版本对应的完整、可修改源码及必要的构建/安装脚本，而不只是差异补丁；
5. 当用户通过网络与修改版交互时，向这些用户显著提供免费取得当前运行版本对应源码
   的入口。本 fork 在 `/lmem help` 和 Dashboard 中提供上述源码仓库链接。

源码入口必须与实际分发或部署的版本保持同步。如果公开仓库尚未包含正在运行的修改，
应先推送对应源码，或改为提供另一个可直接取得完整对应源码的地址。

本段是工程合规摘要，不构成法律意见。正式商业分发、特殊授权或复杂组合软件场景应
咨询合格法律专业人士。

## 第三方材料

Dashboard 随附 Lucide 图标代码及图标数据，使用其自有许可。复制或分发 Dashboard
时必须同时保留：

- [`pages/dashboard/vendor/LUCIDE_LICENSE`](pages/dashboard/vendor/LUCIDE_LICENSE)

ChatMemory 是独立的运行时依赖，其源码不包含在本仓库中。如果把 ChatMemory 与本
插件放进同一个发行包，还需要同时保留 ChatMemory 自己的 MIT 许可证和版权声明。

## 官方参考

- [GNU AGPL-3.0 正文](https://www.gnu.org/licenses/agpl-3.0.html)
- [GNU 许可证常见问题](https://www.gnu.org/licenses/gpl-faq.html)
- [LivingMemory 上游仓库](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory)
