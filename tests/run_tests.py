"""LivingMemoryCM 代码级测试入口，不启动 AstrBot，也不导入真实 AstrBot core。

本地测试只验证领域逻辑与存储协议；`astrbot.api` / `astrbot.core` 由
`tests/conftest.py` 提供最小 fake 类型树。AstrBot core 兼容性由 Windows 测试端
reload/部署验收，不由本入口读取真实 backend 源码保证。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "运行 LivingMemoryCM 函数、模块、存储与数据协议测试。"
            "默认不读取任何 AstrBot 源码/backend，兼容性由部署端验收。"
        )
    )
    parser.add_argument(
        "--astrbot-source",
        type=Path,
        default=None,
        help=(
            "【仅调试用】可选：额外把包含 astrbot/ 包的源码根目录加入 sys.path。"
            "默认不添加；本地测试使用 conftest.py 的 fake astrbot 类型树，"
            "不再依赖真实 AstrBot core。"
        ),
    )
    return parser.parse_args()


def _append_if_directory(path: Path) -> None:
    if path.is_dir():
        sys.path.append(str(path))


def main() -> int:
    # 测试是一次性校验，不在源码树写入无意义的 ``__pycache__``。
    sys.dont_write_bytecode = True

    args = _parse_args()
    workspace = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(workspace))

    # 仅当显式传入时把 AstrBot 源码加入 sys.path（调试/对比用）；默认不注入。
    if args.astrbot_source is not None and (args.astrbot_source / "astrbot").is_dir():
        _append_if_directory(args.astrbot_source)

    # 避免 AstrBot 数据目录中的第三方 pytest 插件被自动加载，
    # 例如与本项目无关的 langsmith 插件及其可选依赖。
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    try:
        import pytest
    except ImportError as exc:
        raise SystemExit(
            "缺少测试依赖。请先执行："
            "python -m pip install -r requirements-test.txt"
        ) from exc

    tests_dir = Path(__file__).resolve().parent
    original_cwd = Path.cwd()
    # 在临时目录中运行，隔离任何测试对当前目录的副作用（例如缓存目录）。
    with tempfile.TemporaryDirectory(prefix="livingmemory_cm_tests_") as temp_dir:
        try:
            os.chdir(temp_dir)
            return int(
                pytest.main(
                    [
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "-p",
                        "pytest_asyncio.plugin",
                        str(tests_dir),
                    ]
                )
            )
        finally:
            # Windows cannot remove a directory while it is the process cwd.
            os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
