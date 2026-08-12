# LivingMemoryCM 测试说明

## 1. 测试目的

本地单元测试只验证**代码与函数功能**：领域算法、状态机、校验规则和边界条件。

它们是“函数对不对”的证明，不是“AstrBot 里跑不跑得起来”的证明。测试名称、断言和文档不得把本地函数测试描述成 AstrBot 集成验收。

## 2. 覆盖边界

| 范围 | 本地测试 | 说明 |
| --- | --- | --- |
| 纯函数、领域模型、状态机、校验规则 | ✅ | 直接验证输入、输出和状态迁移 |
| SQLite/FAISS 临时数据与失败回滚 | ✅ | 只使用 pytest 临时目录和测试替身 |
| CM 游标、图谱、召回策略、命令/API 领域输出 | ✅ | 验证本插件逻辑，不声称验证宿主调度 |
| Dashboard 静态资源与 JavaScript 语法 | ✅ | 不等同于浏览器交互验收 |
| AstrBot reload、Provider 调用、平台发送、WebUI 交互、完整消息流 | ❌ | 交给远端部署端验收 |

`tests/conftest.py` 只 stub `astrbot.api` 中导入所需的最小类型、装饰器和公开句柄。它不 mock `astrbot.core`，不构造 Context、Provider、平台、事件总线、生命周期或浏览器运行时。

测试会读取目标 AstrBot 的 `astrbot.core` 源码类型和基类，因此可以发现导入接口不兼容；但这仍不代表插件能在真实 AstrBot 中完成加载和消息处理。标记为 `integration` 的用例会被本地测试入口拒绝收集，避免把集成测试悄悄塞回单元测试目录。

## 3. 验收分工

1. **本地代码验收**：`pytest` 全绿、全部 Dashboard JavaScript 通过 `node --check`、Python 通过 `compileall`、JSON 配置通过 `json.tool`、Git 工作树通过 `git diff --check`。
2. **远端部署验收**：AstrBot reload、真实 Provider 调用、ChatMemory Hook 顺序、平台消息发送、Plugin Page 浏览器交互和现有数据升级。
3. **用户验收**：beta 部署给实际用户，覆盖长会话、昵称变化、跨重启、异常 Provider 和日常页面操作等真实使用场景。

本地验收通过后只能声明“代码级回归通过”。远端和用户验收完成前，不得声明“部署验证通过”或“生产可用性已证明”。

## 4. 写代码和测试时的要求

- 新逻辑尽量落在**纯函数或领域层**，使其可以本地测试。
- AstrBot 集成层保持薄，只负责事件参数转换、依赖接线和结果转发，复杂状态与规则不放在 Hook 中。
- 测试失败时修正实现或修正错误的测试前提；不允许通过放宽断言、吞异常、扩大容差或删除关键场景来适配错误实现。
- 新用例优先加入已有领域测试文件；只有形成独立领域且现有文件无法清晰容纳时才新建 `test_*.py`。
- 测试替身只实现被测对象明确依赖的协议，不复制 AstrBot 完整运行时。
- 所有数据库和索引用例必须使用 pytest 临时目录，禁止读取或修改 `data/data` 和部署端 `plugin_data`。

## 5. 必要环境

测试使用目标 AstrBot 支持的 Python 版本；本次隔离验收使用 Python 3.13，并需要：

1. `requirements.txt` 中的插件运行依赖。
2. `requirements-test.txt` 中的 pytest 依赖，以及读取目标 AstrBot core 类型所需的
   直接和递归导入依赖；这些依赖只用于让真实 core 类型可导入，不会启动 AstrBot
   运行时。
3. AstrBot 4.26.7 的源码或桌面版 backend，供 `astrbot.core` 类型导入。
4. Node.js，用于 Dashboard JavaScript 语法检查。

`pytest` 不加入插件运行依赖。测试入口会设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，并只显式加载 `pytest-asyncio`，避免 AstrBot 数据目录中的无关 pytest 插件影响收集。

## 6. 可复现运行方式

代码级 pytest：

```powershell
python tests\run_tests.py --astrbot-source "D:\path\to\AstrBot-source-root"
```

桌面版也可以使用：

```powershell
python tests\run_tests.py --astrbot-backend "C:\path\to\AstrBot\backend"
```

其余本地验收项：

```powershell
Get-ChildItem pages\dashboard -Recurse -Filter *.js |
    ForEach-Object { node --check $_.FullName }

python -m compileall -q .
python -m json.tool _conf_schema.json > $null
python -m json.tool core\i18n\zh.json > $null
git diff --check
```

当前源码快照不是 Git checkout 时，`git diff --check` 无法产生有意义的基线结果；应在正式 Git 工作树或提交前 CI 中执行，不能把“目录里没有 `.git`”写成检查通过。

测试入口会切换到自动清理的临时工作目录，避免 AstrBot 导入过程在插件源码树生成宿主配置。`sys.dont_write_bytecode` 和禁用 pytest cache 也用于减少测试残留。

## 7. 当前测试组织

测试已按领域归并为 16 个 `test_*.py`，不再为单个小函数持续拆分文件。领域包括：备份/初始化、命令、会话、调度、图谱、工具、记忆原子、记忆引擎、处理器、分层服务、Page API、CM 反思/召回和通用工具。

2026-08-13 的代码级结果：

```text
435 passed
```

Python 3.13 通过 AstrBot 已声明的 `audioop-lts` 兼容依赖完成导入；本次没有 pytest
warning。该结果仍不等同于远端部署验收。

## 8. 本地测试不能证明的事项

- AstrBot 能否成功 reload 插件及正确执行 terminate。
- Provider API Key、限流、网络和真实 Embedding/LLM 响应是否正常。
- ChatMemory 与 LivingMemoryCM 的真实 Hook 调度顺序。
- 平台消息是否实际发送成功。
- Plugin Page 在真实浏览器和 AstrBot Bridge 中是否可交互。
- 现有生产数据库、FAISS 索引和升级备份是否满足部署要求。

这些事项必须分别进入远端部署验收和用户 beta 验收。
