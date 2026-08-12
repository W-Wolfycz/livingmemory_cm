(() => {
  const MSG = {
    /* ---- Common ---- */
    "common.close":       { zh: "关闭" },
    "common.cancel":      { zh: "取消" },
    "common.clear":       { zh: "清空" },
    "common.save":        { zh: "保存" },
    "common.refresh":     { zh: "刷新" },
    "common.search":      { zh: "搜索" },
    "common.confirm":     { zh: "确定" },
    "common.loading":     { zh: "加载中..." },
    "common.noData":      { zh: "暂无数据" },
    "common.unavailable": { zh: "暂不可用" },
    "common.page":        { zh: "第 {0} / {1} 页 · 共 {2} 条" },
    "common.perPage":     { zh: "每页" },
    "common.perPage20":   { zh: "20 条/页" },
    "common.perPage50":   { zh: "50 条/页" },
    "common.perPage100":  { zh: "100 条/页" },

    /* ---- Title / Header ---- */
    "page.title":         { zh: "LivingMemory 控制台" },
    "header.title":       { zh: "LivingMemory 管理面板" },
    "header.subtitle":    { zh: "长期记忆与会话管理 · 基于混合检索的智能记忆系统" },
    "header.theme":       { zh: "切换主题" },
    "language.current.zh": { zh: "中文" },
    "language.current.en": { zh: "英文" },
    "language.current.ru": { zh: "俄文" },
    "persona.label":      { zh: "人格" },
    "persona.all":        { zh: "全部人格" },
    "license.source":     { zh: "源代码 · AGPL-3.0" },
    "license.sourceAria": { zh: "查看 LivingMemoryCM 源代码（AGPL-3.0）" },

    /* ---- Navigation ---- */
    "nav.memory":         { zh: "记忆管理" },
    "nav.graph":          { zh: "知识图谱" },
    "nav.recallTest":     { zh: "召回测试" },
    "nav.system":         { zh: "系统概览" },
    "nav.recall":         { zh: "召回测试" },

    /* ---- Nuke ---- */
    "nuke.cancel":        { zh: "取消核爆" },
    "nuke.button":        { zh: "核爆清除" },
    "nuke.startToast":    { zh: "核爆倒计时启动！" },
    "nuke.cancelledToast":{ zh: " 核爆已取消！记忆保留" },
    "nuke.cancelFail":    { zh: "取消失败，请稍后重试" },
    "nuke.countdown":     { zh: "所有记忆将在 {0} 秒后被抹除。立即取消以中止核爆！" },
    "nuke.erasing":       { zh: "正在抹除所有记忆... 请保持窗口打开。" },
    "nuke.doneTable":     { zh: " 核爆完成！所有记忆已被抹除。点击「刷新」重新加载。" },
    "nuke.doneToast":     { zh: " 核爆完成！所有记忆已从界面移除（仅视觉效果）" },
    "nuke.cantStart":     { zh: "无法启动核爆模式" },

    /* ---- Stats ---- */
    "stats.total":        { zh: "总记忆" },
    "stats.active":       { zh: "活跃" },
    "stats.archived":     { zh: "已归档" },
    "stats.deleted":      { zh: "已删除" },
    "stats.sessions":     { zh: "活跃会话" },
    "stats.graphNodes":   { zh: "图谱节点" },
    "stats.atoms":        { zh: "原子记忆" },

    /* ---- Filter ---- */
    "filter.keyword":     { zh: "关键字（支持 memory_id / 内容搜索）" },
    "filter.sessionId":   { zh: "会话 ID（可选）" },
    "filter.statusAll":   { zh: "全部状态" },
    "filter.statusActive":{ zh: "活跃" },
    "filter.statusArchived":{ zh: "已归档" },
    "filter.statusDeleted":{ zh: "已删除" },
    "filter.typeAll":     { zh: "全部类型" },
    "filter.apply":       { zh: "筛选" },

    /* ---- Sort ---- */
    "sort.createdDesc":   { zh: "最新创建" },
    "sort.createdAsc":    { zh: "最早创建" },
    "sort.updatedDesc":   { zh: "最近更新" },
    "sort.importanceDesc":{ zh: "重要性高到低" },
    "sort.importanceAsc": { zh: "重要性低到高" },
    "sort.typeAsc":       { zh: "类型 A-Z" },

    /* ---- Table ---- */
    "table.id":           { zh: "记忆 ID" },
    "table.summary":      { zh: "摘要" },
    "table.type":         { zh: "类型" },
    "table.importance":   { zh: "重要性" },
    "table.status":       { zh: "状态" },
    "table.created":      { zh: "创建时间" },
    "table.lastAccess":   { zh: "最后访问" },
    "table.actions":      { zh: "操作" },
    "table.detail":       { zh: "详情" },
    "table.noSummary":    { zh: "（无摘要）" },
    "table.noContent":    { zh: "（无内容）" },
    "table.noData":       { zh: "暂无数据" },
    "table.na":           { zh: "--" },
    "table.updated":      { zh: "更新于 {0}" },

    /* ---- Pagination ---- */
    "pagination.prev":    { zh: "上一页" },
    "pagination.next":    { zh: "下一页" },
    "pagination.allLoaded":{ zh: "共 {0} 条记录（已加载全部）" },
    "pagination.filtering":{ zh: "筛选中:" },
    "pagination.byKeyword":{ zh: "关键词=\"{0}\"" },
    "pagination.byStatus":{ zh: "状态=\"{0}\"" },
    "pagination.bySession":{ zh: "会话=\"{0}\"" },

    /* ---- Search / Results Toast ---- */
    "search.resultToast": { zh: "搜索结果：找到 {0} 条记忆，当前显示第 {1} 条" },

    /* ---- Delete ---- */
    "delete.confirmTitle":{ zh: "确认删除？" },
    "delete.confirmMsg":  { zh: "即将删除 {0} 条记忆。\n此操作无法撤销！\n\n点击\"确定\"继续删除，点击\"取消\"保留。" },
    "delete.selected":    { zh: "删除所选 ({0})" },
    "delete.selectedTitle":{ zh: "删除当前页中选中的记忆" },
    "delete.selectAll":   { zh: "选择当前页全部记忆" },
    "delete.selectOne":   { zh: "选择记忆 #{0}" },
    "delete.cancelled":   { zh: "已取消删除操作" },
    "delete.deleting":    { zh: "删除中..." },
    "delete.allFailed":   { zh: " 删除失败：全部 {0} 条记忆无法删除\n失败ID: {1}\n请检查日志了解详情" },
    "delete.partialFailed":{ zh: "部分删除失败：成功 {0} 条，失败 {1} 条\n失败ID: {2}" },
    "delete.success":     { zh: " 已成功删除 {0} 条记忆" },
    "delete.successOne":  { zh: "已删除记忆 #{0}" },
    "delete.none":        { zh: "没有删除任何记忆" },
    "delete.error":       { zh: "删除失败，请稍后重试" },

    "batchEdit.button":   { zh: "批量编辑 ({0})" },
    "batchEdit.title":    { zh: "批量编辑 {0} 条记忆" },
    "batchEdit.field":    { zh: "编辑字段" },
    "batchEdit.importance":{ zh: "重要性" },
    "batchEdit.status":   { zh: "状态" },
    "batchEdit.type":     { zh: "类型" },
    "batchEdit.value":    { zh: "新值" },
    "batchEdit.typePlaceholder":{ zh: "如 FACT / EVENT / PREFERENCE" },
    "batchEdit.apply":    { zh: "应用" },
    "batchEdit.success":  { zh: "已成功更新 {0} 条记忆" },
    "batchEdit.partialFailed":{ zh: "部分更新失败：成功 {0} 条，失败 {1} 条" },
    "batchEdit.error":    { zh: "批量编辑失败，请稍后重试" },
    "batchEdit.valueRequired":{ zh: "请输入新值" },
    "batchEdit.importanceRange":{ zh: "重要性必须在 0-10 之间" },

    /* ---- Archive ---- */
    "archive.success":    { zh: "已归档 {0} 条记忆" },
    "archive.fail":       { zh: "归档失败" },
    "archive.error":      { zh: "归档失败" },

    /* ---- Detail Drawer ---- */
    "detail.title":       { zh: "记忆详情" },
    "detail.edit":        { zh: "编辑记忆" },
    "detail.close":       { zh: "关闭详情" },
    "detail.memoryId":    { zh: "记忆 ID" },
    "detail.source":      { zh: "来源" },
    "detail.sourceCustom":{ zh: "自定义存储" },
    "detail.sourceVector":{ zh: "向量存储" },
    "detail.status":      { zh: "状态" },
    "detail.importance":  { zh: "重要性" },
    "detail.type":        { zh: "类型" },
    "detail.created":     { zh: "创建时间" },
    "detail.lastAccess":  { zh: "最后访问" },
    "detail.notFound":    { zh: "未找到对应的记录" },

    /* ---- Edit Modal ---- */
    "edit.title":         { zh: "编辑记忆" },
    "edit.field":         { zh: "编辑字段" },
    "edit.fieldContent":  { zh: "内容" },
    "edit.fieldImportance":{ zh: "重要性" },
    "edit.fieldType":     { zh: "类型" },
    "edit.fieldStatus":   { zh: "状态" },
    "edit.newContent":    { zh: "新内容" },
    "edit.newContentPh":  { zh: "输入新的记忆内容" },
    "edit.newImportance": { zh: "新重要性 (0-10)" },
    "edit.importanceHint":{ zh: "重要性越高，记忆被召回的优先级越高" },
    "edit.newType":       { zh: "新类型" },
    "edit.typePh":        { zh: "如: FACT, EVENT, PREFERENCE" },
    "edit.typeHint":      { zh: "记忆类型用于分类管理" },
    "edit.newStatus":     { zh: "新状态" },
    "edit.statusPh":      { zh: "活跃" },
    "edit.statusArchived":{ zh: "已归档" },
    "edit.statusDeleted": { zh: "已删除" },
    "edit.statusHint":    { zh: "已删除的记忆不会被召回" },
    "edit.reason":        { zh: "更新原因 (可选)" },
    "edit.reasonPh":      { zh: "说明本次更新的原因" },
    "edit.noItem":        { zh: "未找到当前记忆信息" },
    "edit.enterValue":    { zh: "请输入新值" },
    "edit.updateFailed":  { zh: "更新失败" },
    "edit.success":       { zh: "更新成功" },

    /* ---- Status pills ---- */
    "status.active":      { zh: "活跃" },
    "status.archived":    { zh: "已归档" },
    "status.deleted":     { zh: "已删除" },

    /* ---- Type labels ---- */
    "type.general":       { zh: "通用" },
    "type.fact":          { zh: "事实" },
    "type.factual":       { zh: "事实" },
    "type.preference":    { zh: "偏好" },
    "type.event":         { zh: "事件" },
    "type.episodic":      { zh: "事件" },
    "type.relational":    { zh: "关系" },
    "type.planned":       { zh: "计划" },
    "type.opinion":       { zh: "观点" },

    /* ---- Graph Hero ---- */
    "graph.kicker":       { zh: "Graph Memory Explorer" },
    "graph.title":        { zh: "知识图谱视图" },
    "graph.subtitle":     { zh: "从双路四模式召回结果中观察人物、主题、事实与记忆之间的连接。" },

    /* ---- Graph Toolbar ---- */
    "graph.queryLabel":   { zh: "图谱查询" },
    "graph.queryPh":      { zh: "输入人物、主题、事实或整句，查看召回到的图谱子图" },
    "graph.sessionLabel": { zh: "会话过滤" },
    "graph.sessionPh":    { zh: "可选：限定 session_id" },
    "graph.personaLabel": { zh: "人格过滤" },
    "graph.personaPh":    { zh: "可选：限定 persona_id" },
    "graph.memoryIdLabel":{ zh: "记忆 ID" },
    "graph.memoryIdPh":   { zh: "输入记忆 ID 定位局部子图" },
    "graph.searchBtn":    { zh: "检索图谱" },
    "graph.focusBtn":     { zh: "定位记忆" },
    "graph.overviewBtn":  { zh: "扩展图谱" },

    /* ---- Graph Stats ---- */
    "graph.visibleNodes": { zh: "可视节点" },
    "graph.nodes":        { zh: "节点" },
    "graph.edges":        { zh: "关系" },
    "graph.visibleEdges": { zh: "关系边" },
    "graph.visibleEntries":{ zh: "图谱条目" },
    "graph.routeLabel":   { zh: "检索视角" },
    "graph.visibleMemories":{ zh: "关联记忆" },

    /* ---- Graph Panels ---- */
    "graph.canvasTitle":  { zh: "图谱画布" },
    "graph.canvasSubtitle":{ zh: "点击节点、记忆卡片或召回结果即可切换焦点。" },
    "graph.focusDetail":  { zh: "焦点详情" },
    "graph.topNodes":     { zh: "核心节点" },
    "graph.relatedMemories":{ zh: "相关记忆" },
    "graph.retrievalPath":{ zh: "召回路径" },

    /* ---- Graph Status / Modes ---- */
    "graph.modeOverview": { zh: "最近概览" },
    "graph.modeQuery":    { zh: "检索视图" },
    "graph.modeFocus":    { zh: "记忆聚焦" },
    "graph.modeUnknown":  { zh: "图谱视图" },
    "graph.routeDual":    { zh: "文档 + 图 · 关键词 + 向量" },
    "graph.routeBrowse":  { zh: "图谱浏览" },
    "graph.statusDefault":{ zh: "展示图记忆中的核心连接。" },
    "graph.statusQuery":  { zh: "当前展示 \"{0}\" 的双路四模式召回对应子图。" },
    "graph.statusFocus":  { zh: "当前聚焦记忆 #{0} 的关系子图。" },
    "graph.filterSession":{ zh: "会话 {0}" },
    "graph.filterPersona":{ zh: "人格 {0}" },
    "graph.filterPrefix": { zh: " 过滤条件：{0}" },

    /* ---- Graph Node Types ---- */
    "graph.nodeTopic":    { zh: "主题" },
    "graph.nodePerson":   { zh: "人物" },
    "graph.nodeFact":     { zh: "事实" },
    "graph.nodeSummary":  { zh: "摘要" },
    "graph.nodeUnknown":  { zh: "节点" },

    /* ---- Graph Score Labels ---- */
    "graph.scoreDocKW":   { zh: "文档关键词" },
    "graph.scoreDocVec":  { zh: "文档向量" },
    "graph.scoreGraphKW": { zh: "图关键词" },
    "graph.scoreGraphVec":{ zh: "图向量" },

    /* ---- Graph Disabled ---- */
    "graph.disabledBadge":{ zh: "图记忆未启用" },
    "graph.disabledMsg":  { zh: "当前实例未启用图记忆功能，请先开启图记忆并完成索引。" },
    "graph.disabledRoute":{ zh: "未启用" },
    "graph.disabledLegend":{ zh: "暂无图数据" },
    "graph.disabledMemories":{ zh: "暂无可展示的图记忆" },
    "graph.disabledRetrieval":{ zh: "点击\"扩展图谱\"加载更大的关系窗口，或直接输入检索词。" },
    "graph.disabledInspector":{ zh: "请选择节点或记忆查看详细信息。" },
    "graph.disabledCanvas":{ zh: "当前实例尚未启用图记忆。" },

    /* ---- Graph Error ---- */
    "graph.errorBadge":   { zh: "图谱加载失败" },
    "graph.errorLegend":  { zh: "请求失败" },
    "graph.errorFetch":   { zh: "无法加载图谱概览" },

    /* ---- Graph Canvas Messages ---- */
    "graph.canvasDefault":{ zh: "点击\"扩展图谱\"加载更大的关系窗口，或直接输入检索词。" },
    "graph.canvasNo3D":   { zh: "3D 图谱组件未加载，请刷新页面并检查静态资源。" },
    "graph.canvasEmpty":  { zh: "当前范围内暂无可视化图数据。" },
    "graph.canvasNoScene":{ zh: "当前页面未能加载 3D 图谱组件，请刷新页面后重试。" },

    /* ---- Graph Loading ---- */
    "graph.loadingOverview":{ zh: "正在加载扩展节点与关系..." },
    "graph.loadingQuery": { zh: "正在检索\"{0}\"相关图谱..." },
    "graph.loadingFocus": { zh: "正在聚焦记忆 #{0} 的关系图..." },
    "graph.loadingGeneric":{ zh: "图谱载入中..." },

    /* ---- Graph Errors (actions) ---- */
    "graph.queryFail":    { zh: "图谱检索失败" },
    "graph.focusEmpty":   { zh: "请输入要定位的记忆 ID。" },
    "graph.focusNotInt":  { zh: "记忆 ID 必须是整数。" },
    "graph.focusFail":    { zh: "定位记忆失败" },
    "graph.statsFailed":  { zh: "获取图谱统计失败" },

    /* ---- Graph Legend ---- */
    "graph.legendEmpty":  { zh: "暂无图谱连接" },

    /* ---- Graph Panels Content ---- */
    "graph.noTopNodes":   { zh: "暂无核心节点" },
    "graph.noRelatedMemories":{ zh: "暂无关联记忆" },
    "graph.noRetrieval":  { zh: "执行检索后，这里会展示文档 / 图 × 关键词 / 向量的召回细节。" },
    "graph.noInspector":  { zh: "点击节点、记忆卡片或召回结果查看详细信息。" },
    "graph.unnamedNode":  { zh: "未命名节点" },
    "graph.noSummary":    { zh: "无摘要" },
    "graph.focusThisMemory":{ zh: "聚焦此记忆" },
    "graph.noSession":    { zh: "未设置会话" },

    /* ---- Graph Inspector ---- */
    "graph.inspectorMemoryCount":{ zh: "关联记忆" },
    "graph.inspectorDegree":{ zh: "连接度" },
    "graph.inspectorEntryCount":{ zh: "命中条目" },
    "graph.inspectorWeight":{ zh: "权重" },
    "graph.inspectorRelatedMemories":{ zh: "相关记忆" },
    "graph.inspectorNoRelatedMemories":{ zh: "暂无相关记忆" },
    "graph.inspectorRelatedEntries":{ zh: "相关条目" },
    "graph.inspectorNoRelatedEntries":{ zh: "暂无相关条目" },
    "graph.inspectorNodeDist":{ zh: "节点分布" },
    "graph.inspectorNoNodes":{ zh: "暂无节点" },
    "graph.inspectorGraphEntries":{ zh: "图谱条目" },
    "graph.inspectorNoGraphEntries":{ zh: "暂无图谱条目" },
    "graph.inspectorNodeCount":{ zh: "节点" },
    "graph.inspectorEntryCount2":{ zh: "条目" },
    "graph.inspectorRelationCount":{ zh: "关系" },
    "graph.inspectorImportance":{ zh: "重要性" },
    "graph.inspectorMemory":{ zh: "记忆" },

    /* ---- Graph Tooltip ---- */
    "graph.tooltipMemory": { zh: "记忆 {0} · 关系 {1} · 条目 {2}" },

    /* ---- Graph Bridge Error ---- */
    "graph.bridgeError":  { zh: "当前页面必须运行在 AstrBot 官方插件 Page 内。" },

    /* ---- Recall Test ---- */
    "recall.clearBtn":    { zh: "清空结果" },
    "recall.title":       { zh: "记忆召回功能测试" },
    "recall.subtitle":    { zh: "输入查询语句，测试混合检索引擎的召回能力" },
    "recall.queryLabel":  { zh: "查询内容" },
    "recall.queryPh":     { zh: "输入你的查询语句，系统将使用混合检索（BM25+向量相似度）进行召回" },
    "recall.countLabel":  { zh: "返回数量" },
    "recall.kLabel":      { zh: "结果数 (k)" },
    "recall.countPh":     { zh: "返回的记忆数量" },
    "recall.sessionLabel":{ zh: "会话 ID (可选)" },
    "recall.sessionPh":   { zh: "输入会话 ID 以过滤特定会话的记忆（支持多种格式）" },
    "recall.searchBtn":   { zh: "执行召回" },
    "recall.resultTitle": { zh: "召回结果" },
    "recall.resultCount": { zh: "召回数量" },
    "recall.resultsCount":{ zh: "{0} 条结果" },
    "recall.time":        { zh: "查询耗时" },
    "recall.empty":       { zh: "暂无召回结果 · 请输入查询内容并执行召回" },
    "recall.noMatch":     { zh: "未找到匹配的记忆" },
    "recall.noResults":   { zh: "未找到匹配的记忆" },
    "recall.enterQuery":  { zh: "请输入查询内容" },
    "recall.queryRequired":{ zh: "请输入查询内容" },
    "recall.searching":   { zh: "执行中..." },
    "recall.successToast":{ zh: "成功召回 {0} 条记忆" },
    "recall.fail":        { zh: "召回失败" },
    "recall.testFailed":  { zh: "召回测试失败" },
    "recall.timeElapsed": { zh: "耗时 {0} 秒" },

    /* ---- Recall Results Metadata ---- */
    "recall.resultId":    { zh: "记忆 ID:" },
    "recall.resultScore": { zh: "相似度得分:" },
    "recall.resultSession":{ zh: "会话 UUID:" },
    "recall.resultImportance":{ zh: "重要性:" },
    "recall.resultType":  { zh: "类型:" },
    "recall.resultStatus":{ zh: "状态:" },

    /* ---- Theme ---- */
    "theme.darkToast":    { zh: "已切换到深色模式" },
    "theme.lightToast":   { zh: "已切换到浅色模式" },

    /* ---- Bridge Error ---- */
    "bridge.error":       { zh: "当前页面必须运行在 AstrBot 官方插件 Page 内" },

    /* ---- Misc ---- */
    "misc.requestFailed": { zh: "请求失败" },
    "misc.initFail":      { zh: "初始化加载失败" },
    "misc.statsFail":     { zh: "获取统计信息失败" },
    "misc.statsUnavailable":{ zh: "无法获取统计信息" },
    "misc.fetchMemoriesFail":{ zh: "获取记忆失败" },
    "misc.loadFail":      { zh: "加载失败" },
    "misc.systemFail":    { zh: "系统概览加载失败" },

    /* ---- System ---- */
    "system.importanceDistribution":{ zh: "重要性分布" },
    "system.atomTypes":   { zh: "原子类型" },
    "system.activeSessions":{ zh: "活跃会话" },
    "system.versionBackups":{ zh: "版本备份" },
    "system.noActiveSessions":{ zh: "暂无活跃会话" },
    "system.noSessions":  { zh: "暂无会话" },
    "system.noBackups":   { zh: "暂无备份" },
    "system.noAtoms":     { zh: "暂无原子数据" },
    "system.files":       { zh: "个文件" },
    "system.messages":    { zh: "条消息" },
    "system.lastActive":  { zh: "最后活跃" },
    "system.fetchFailed": { zh: "获取系统数据失败" },
    "system.atomFactual": { zh: "事实" },
    "system.atomEpisodic":{ zh: "事件" },
    "system.atomPreference":{ zh: "偏好" },
    "system.atomRelational":{ zh: "关系" },
    "system.atomPlanned": { zh: "计划" },

    /* ---- Atom labels ---- */
    "atom.entity":        { zh: "实体" },
    "atom.event":         { zh: "事件" },
    "atom.preference":    { zh: "偏好" },
    "atom.topic":         { zh: "主题" },

    /* ---- Memory Detail ---- */
    "detail.viewTitle":   { zh: "记忆详情" },
    "detail.editTitle":   { zh: "编辑记忆" },
    "detail.content":     { zh: "内容" },
    "detail.metadata":    { zh: "元数据" },
    "detail.graphContext":{ zh: "知识图谱关联" },
    "detail.keyFacts":    { zh: "关键事实" },
    "detail.topics":      { zh: "主题" },
    "detail.editHistory": { zh: "编辑历史" },
    "detail.editBtn":     { zh: "编辑" },
    "detail.deleteBtn":   { zh: "删除" },
    "detail.saveBtn":     { zh: "保存修改" },
    "detail.cancelBtn":   { zh: "取消" },
    "detail.memoryTitle": { zh: "记忆 #{0}" },
    "detail.editingTitle":{ zh: "正在编辑记忆 #{0}" },
    "detail.sessionId":   { zh: "会话 ID" },
    "detail.personaId":   { zh: "人格 ID" },
    "detail.updated":     { zh: "更新时间" },
    "detail.updateReason":{ zh: "更新原因（可选）" },
    "detail.reasonPh":    { zh: "说明本次更新的原因" },
    "detail.contentHint": { zh: "编辑内容将创建新记忆（ID 会变更）" },
    "detail.noGraphData": { zh: "暂无图谱数据" },
    "detail.noChanges":   { zh: "没有检测到修改" },
    "detail.contentRequired":{ zh: "记忆内容不能为空" },
    "detail.contentUpdated":{ zh: "内容已更新（新 ID：{0}）" },
    "detail.statusUpdated":{ zh: "状态 → {0}" },
    "detail.typeUpdated": { zh: "类型 → {0}" },
    "detail.importanceUpdated":{ zh: "重要性 → {0}" },
    "detail.nodeMemories":{ zh: "关联记忆" },
    "detail.nodeDegree":  { zh: "连接度" },
    "detail.nodeEntries": { zh: "条目" },
    "detail.nodeWeight":  { zh: "权重" },

    /* ---- Confirm dialog ---- */
    "confirm.deleteTitle":{ zh: "确认删除？" },
    "confirm.deleteMessage":{ zh: "即将删除记忆 #{0}。此操作无法撤销。" },
    "memory.deleted":     { zh: "记忆已删除" },
    "memory.deleteFailed":{ zh: "删除记忆失败" },

    /* ---- Graph 2D ---- */
    "graph2d.noData":     { zh: "暂无图谱数据" },
    "graph2d.loading":    { zh: "加载图谱中..." },
    "graph2d.moduleFail": { zh: "2D 图谱模块未加载，请刷新页面重试。" },

  };

  /* ---- Engine ---- */
  /**
   * @param {string} key
   * @param {...(string|number)} args - positional replacements for {0}, {1}, ...
   */
  window.t = function (key, ...args) {
    const entry = MSG[key];
    let template = entry ? (entry.zh || key) : key;
    args.forEach((arg, i) => {
      template = template.replace(new RegExp("\\{" + i + "\\}", "g"), String(arg ?? ""));
    });
    return template;
  };

  function applyI18n() {
    // data-i18n → textContent
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = window.t(el.getAttribute("data-i18n"));
    });
    // data-i18n-placeholder → placeholder
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      el.setAttribute("placeholder", window.t(el.getAttribute("data-i18n-placeholder")));
    });
    // data-i18n-title → title
    document.querySelectorAll("[data-i18n-title]").forEach((el) => {
      el.setAttribute("title", window.t(el.getAttribute("data-i18n-title")));
    });
    // data-i18n-aria → aria-label
    document.querySelectorAll("[data-i18n-aria]").forEach((el) => {
      el.setAttribute("aria-label", window.t(el.getAttribute("data-i18n-aria")));
    });
  }

  // bootstrap
  document.documentElement.setAttribute("lang", "zh-CN");
  document.addEventListener("DOMContentLoaded", () => {
    applyI18n();
  });
})();
