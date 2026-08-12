"""LivingMemoryCM 代码级测试入口，不启动 AstrBot。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _default_astrbot_backend() -> Path | None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate = Path(local_app_data) / "AstrBot" / "backend"
        if candidate.is_dir():
            return candidate
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 LivingMemoryCM 函数、模块、存储与数据协议测试。"
    )
    parser.add_argument(
        "--astrbot-backend",
        type=Path,
        default=_default_astrbot_backend(),
        help="AstrBot backend 目录；Windows 默认从 LOCALAPPDATA/AstrBot/backend 解析。",
    )
    parser.add_argument(
        "--astrbot-source",
        type=Path,
        default=(
            Path(os.environ["ASTRBOT_SOURCE"])
            if os.environ.get("ASTRBOT_SOURCE")
            else None
        ),
        help="包含 astrbot/ 包的源码根目录；优先于桌面版 backend/app。",
    )
    return parser.parse_args()


def _append_if_directory(path: Path) -> None:
    if path.is_dir():
        sys.path.append(str(path))


def _normalize_backend_path(value: Path | None) -> Path | None:
    """兼容 Windows 原生路径和 WSL 的 /mnt/<drive>/ 路径。"""
    if value is None:
        return None
    raw = str(value)
    # Windows Python may normalize a WSL argument into ``\\mnt\\c\\...``
    # before this function sees it. Handle both spellings.
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/mnt/") and len(normalized) > 6:
        drive = normalized[5]
        remainder = normalized[6:].replace("/", "\\")
        return Path(f"{drive.upper()}:\\{remainder}")
    return value


def main() -> int:
    # 测试是一次性校验，不在源码树写入无意义的 ``__pycache__``。
    sys.dont_write_bytecode = True

    args = _parse_args()
    workspace = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(workspace))

    source = _normalize_backend_path(args.astrbot_source)
    backend = _normalize_backend_path(args.astrbot_backend)
    if source is not None and (source / "astrbot").is_dir():
        _append_if_directory(source)
    elif backend is not None and (backend / "app" / "astrbot").is_dir():
        _append_if_directory(backend / "app")
    else:
        raise SystemExit(
            "未找到 AstrBot core 源码。请传入 --astrbot-source <源码根目录> "
            "或 --astrbot-backend <桌面版 backend>；测试只读取类型与接口，"
            "不会启动 AstrBot。"
        )

    if backend is not None:
        _append_if_directory(backend / "python" / "Lib" / "site-packages")

    # AstrBot 桌面版会把部分插件依赖安装到用户数据目录。
    _append_if_directory(Path.home() / ".astrbot" / "data" / "site-packages")

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
    # AstrBot import may initialize host defaults relative to the current
    # directory. Isolate those side effects from the plugin source tree.
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
