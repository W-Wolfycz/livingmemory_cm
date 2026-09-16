"""Dashboard 前端纯逻辑回归：HTML 转义与 API 重试策略。

用 Node 直接 import 页面模块（复制为 .mjs 后加载，模块本身无依赖、无 DOM
顶层代码），因此测的是真实实现而不是源码字符串。Node 不可用时整体跳过——
本地验收里 `node --check` 仍会覆盖语法。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node 不可用，跳过前端行为测试")

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "pages" / "dashboard"
UTILS_JS = DASHBOARD_DIR / "modules" / "utils.js"
API_CLIENT_JS = DASHBOARD_DIR / "modules" / "api-client.js"

_HARNESS = r"""
import { pathToFileURL } from "node:url";

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok: Boolean(ok), detail: String(detail) });
}

// ---------- 1) esc()：属性上下文防 breakout ----------
const utils = await import(pathToFileURL(process.env.UTILS_MJS).href);

check("esc 转义双引号", !utils.esc('"').includes('"'), utils.esc('"'));
check("esc 转义单引号", !utils.esc("'").includes("'"), utils.esc("'"));
check(
  "esc 转义 & < >",
  utils.esc("<a & b>") === "&lt;a &amp; b&gt;",
  utils.esc("<a & b>")
);
check("esc 保留普通文本与中文", utils.esc("人格 记忆") === "人格 记忆");

const payload = '"><img src=x onerror=alert(1)>';
const attr = '<input type="text" value="' + utils.esc(payload) + '" />';
check(
  "payload 拼进 value 属性后无法提前闭合",
  attr.indexOf('"><img') === -1 && attr.indexOf("onerror=alert") !== -1,
  attr
);

// ---------- 2) ApiClient 重试策略 ----------
globalThis.window = { AstrBotPluginPage: null, t: (k) => k };
const clientModule = await import(pathToFileURL(process.env.CLIENT_MJS).href);
// 退避立即执行：只验证重试次数，不为单测引入真实等待
globalThis.setTimeout = (fn) => {
  fn();
  return 0;
};

function makeBridge({ failTimes = 0, failForever = false } = {}) {
  const calls = { get: 0, post: 0 };
  const bridge = {
    async apiGet() {
      calls.get += 1;
      if (failForever || calls.get <= failTimes) throw new Error("bridge timeout");
      return { status: "ok", data: { ok: true } };
    },
    async apiPost() {
      calls.post += 1;
      if (failForever || calls.post <= failTimes) throw new Error("bridge timeout");
      return { status: "ok", data: { ok: true } };
    },
    async ready() {
      return {};
    },
    getContext() {
      return {};
    },
  };
  return { bridge, calls };
}

async function withClient(options, invoke) {
  const { bridge, calls } = makeBridge(options);
  globalThis.window.AstrBotPluginPage = bridge;
  const client = new clientModule.ApiClient();
  await invoke(client);
  return calls;
}

let calls = await withClient({ failForever: true }, (c) =>
  c.post("memories/update", {}).catch(() => {})
);
check("POST 默认不重试：只调用 1 次", calls.post === 1, JSON.stringify(calls));

calls = await withClient({ failForever: true }, (c) => c.get("stats").catch(() => {}));
check("GET 默认重试 2 次：共调用 3 次", calls.get === 3, JSON.stringify(calls));

calls = await withClient({ failForever: true }, (c) =>
  c.post("recall/test", {}, { retries: 2 }).catch(() => {})
);
check("POST 显式 retries:2：共调用 3 次", calls.post === 3, JSON.stringify(calls));

calls = await withClient({ failForever: true }, (c) =>
  c.request("x", { method: "GET", retries: 0 }).catch(() => {})
);
check("GET 显式 retries:0：只调用 1 次", calls.get === 1, JSON.stringify(calls));

calls = await withClient({ failTimes: 1 }, (c) => c.get("stats"));
check("GET 失败一次后成功：2 次调用并返回数据", calls.get === 2, JSON.stringify(calls));

calls = await withClient({}, (c) => c.post("recall/test", {}, { retries: 2 }));
check("POST 成功路径：1 次调用即返回", calls.post === 1, JSON.stringify(calls));

console.log(JSON.stringify(results));
"""


def _run_harness(tmp_path: Path) -> list[dict]:
    utils_mjs = tmp_path / "utils.mjs"
    client_mjs = tmp_path / "api-client.mjs"
    utils_mjs.write_text(UTILS_JS.read_text(encoding="utf-8"), encoding="utf-8")
    client_mjs.write_text(API_CLIENT_JS.read_text(encoding="utf-8"), encoding="utf-8")
    harness = tmp_path / "harness.mjs"
    harness.write_text(_HARNESS, encoding="utf-8")

    completed = subprocess.run(
        [NODE, str(harness)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": "/usr/bin:/bin",
            "UTILS_MJS": str(utils_mjs),
            "CLIENT_MJS": str(client_mjs),
        },
        cwd=str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_dashboard_escape_and_retry_policy(tmp_path: Path) -> None:
    checks = _run_harness(tmp_path)

    failed = [item for item in checks if not item["ok"]]
    assert checks, "harness 未产出结果"
    assert not failed, "前端行为断言失败: " + "; ".join(
        f"{item['name']} → {item['detail']}" for item in failed
    )


def test_dashboard_write_calls_declare_no_retry() -> None:
    """写路径必须显式声明不重试，避免桥接超时后重复创建/重复副作用。"""
    peek = (DASHBOARD_DIR / "modules" / "peek-panel.js").read_text(encoding="utf-8")
    memory_page = (DASHBOARD_DIR / "modules" / "memory-page.js").read_text(
        encoding="utf-8"
    )
    assert peek.count("{ retries: 0 }") == 5, "peek-panel 的 4 次 update + 1 次删除需显式不重试"
    assert memory_page.count("{ retries: 0 }") == 2
