"""FAISS 运行时检查、路径兼容和索引健康检查。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata as importlib_metadata
from typing import Any

from astrbot.core.provider.provider import EmbeddingProvider

from ..base.exceptions import InitializationError
from ...log import logger, tag


_FAISS_GENERIC_FALLBACK_MARKERS = (
    "illegal instruction",
    "optimized",
    "avx",
    "simd",
    "dll load failed",
    "cannot open shared object file",
    "could not load library",
    "image not found",
    "symbol not found",
    "undefined symbol",
)


def _faiss_error_details(result: Any) -> str:
    """提取 FAISS 子进程探针的可诊断错误，不输出本机路径或密钥。"""
    details = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
    returncode = getattr(result, "returncode", 0)
    if returncode < 0:
        details = f"进程被信号 {-returncode} 终止。{details}".strip()
    return details


def _is_faiss_binding_mismatch(details: str) -> bool:
    """识别 Python 封装与 FAISS 二进制扩展不匹配。"""
    lowered = details.lower()
    return "superkmeans" in lowered or (
        "python binding" in lowered and "mismatch" in lowered
    )


def _should_try_faiss_generic(result: Any) -> bool:
    """仅在可能由 CPU 指令集/动态库选择导致时尝试 generic。"""
    if getattr(result, "returncode", 0) < 0:
        return True
    details = _faiss_error_details(result).lower()
    return any(marker in details for marker in _FAISS_GENERIC_FALLBACK_MARKERS)


def _installed_faiss_version() -> str:
    """读取安装元数据，不导入可能已损坏的 faiss 模块。"""
    try:
        return importlib_metadata.version("faiss-cpu")
    except importlib_metadata.PackageNotFoundError:
        return "未知"


class FaissBootstrapService:
    """封装 AstrBot FaissVecDB 加载及 Windows 非 ASCII 路径兼容。"""

    def __init__(self) -> None:
        self._vec_db_class: Any = None

    @staticmethod
    def needs_bridge(path: str | os.PathLike[str]) -> bool:
        """判断是否需要纯 ASCII 临时文件桥接。"""
        normalized = os.fspath(path)
        if isinstance(normalized, bytes):
            normalized = os.fsdecode(normalized)
        return os.name == "nt" and not normalized.isascii()

    @staticmethod
    def safe_temp_dir() -> str:
        """返回保证纯 ASCII 且可写的临时目录。"""
        if os.name == "nt":
            root = os.environ.get("SystemRoot", r"C:\Windows")
            temp_dir = os.path.join(root, "Temp")
            if (
                temp_dir.isascii()
                and os.path.isdir(temp_dir)
                and os.access(temp_dir, os.W_OK)
            ):
                return temp_dir
            tmp = tempfile.gettempdir()
            if tmp.isascii():
                return tmp
            raise OSError("FaissBootstrapService: 无法找到可写的纯 ASCII 临时目录")
        return tempfile.gettempdir()

    @classmethod
    def make_temp_file(cls, prefix: str) -> str:
        """创建 FAISS 桥接临时文件。"""
        fd, path = tempfile.mkstemp(
            prefix=f"{prefix}_",
            suffix=".faiss",
            dir=cls.safe_temp_dir(),
        )
        os.close(fd)
        return path

    @staticmethod
    def sanitize_path(path: str | os.PathLike[str]) -> str:
        """脱敏路径中的非 ASCII 部分，避免日志泄露本机用户名。"""
        value = os.fsdecode(os.fspath(path))
        if value.isascii():
            return value
        parts: list[str] = []
        for char in value:
            if char.isascii():
                parts.append(char)
            elif not parts or parts[-1] != "[***]":
                parts.append("[***]")
        return "".join(parts)

    def check_runtime(self) -> None:
        """在加载 FaissVecDB 前确认当前 Python 可安全导入 faiss。"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import faiss"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InitializationError(
                "FAISS 运行时检查失败，无法安全初始化向量数据库。"
                "请确认 faiss-cpu 已正确安装，或改用兼容当前 CPU 的 FAISS 包。"
            ) from exc

        if result.returncode == 0:
            return

        details = _faiss_error_details(result)
        if _is_faiss_binding_mismatch(details):
            raise InitializationError(
                "FAISS 初始化失败：Python 封装与二进制扩展不匹配"
                f"（检测到 faiss-cpu {_installed_faiss_version()}）。"
                "这不是 Embedding Provider 配置问题。请在同一 Python 环境中"
                "干净安装兼容版本，避免 faiss-cpu 1.14.2。"
                f"{' 原始错误: ' + details if details else ''}"
            )

        if not _should_try_faiss_generic(result):
            raise InitializationError(
                "FAISS 初始化失败，faiss-cpu 无法在当前 Python 环境中加载。"
                "请检查安装是否完整，并确保 Python 封装与二进制扩展来自同一版本。"
                f"{' 原始错误: ' + details if details else ''}"
            )

        # 只有明确属于 CPU 指令集或动态库选择问题时，才探测 generic 扩展。
        generic_env = os.environ.copy()
        generic_env["FAISS_OPT_LEVEL"] = "generic"
        try:
            generic_result = subprocess.run(
                [sys.executable, "-c", "import faiss"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env=generic_env,
            )
        except (OSError, subprocess.TimeoutExpired):
            generic_result = None

        if generic_result is not None and generic_result.returncode == 0:
            os.environ["FAISS_OPT_LEVEL"] = "generic"
            logger.warning(
                f"{tag('init')} FAISS 默认优化扩展加载失败，已回退到 generic 指令集兼容模式。"
            )
            return

        if generic_result is not None:
            generic_details = _faiss_error_details(generic_result)
            if generic_details and generic_details != details:
                details = f"{details}；generic 模式: {generic_details}".strip("；")
        raise InitializationError(
            "FAISS 初始化失败，当前 CPU 或运行环境可能不兼容 faiss-cpu。"
            "已尝试 generic 指令集兼容模式；请重新安装兼容版本的 FAISS，"
            "或更换运行环境。"
            f"{' 原始错误: ' + details if details else ''}"
        )

    def load_vec_db_class(self):
        """加载并缓存 AstrBot 的 FaissVecDB 类。"""
        if self._vec_db_class is not None:
            return self._vec_db_class

        self.check_runtime()
        try:
            import faiss as _faiss

            original_read = _faiss.read_index
            original_write = _faiss.write_index

            def patched_read_index(path: str, *args, **kwargs):
                if isinstance(path, (str, bytes, os.PathLike)) and self.needs_bridge(path):
                    temp_path = self.make_temp_file("_faiss_read")
                    try:
                        shutil.copy2(path, temp_path)
                        return original_read(temp_path, *args, **kwargs)
                    finally:
                        self._remove_temp_file(temp_path)
                return original_read(path, *args, **kwargs)

            def patched_write_index(index, path, *args, **kwargs) -> None:
                if isinstance(path, (str, bytes, os.PathLike)) and self.needs_bridge(path):
                    target_path = os.fsdecode(os.fspath(path))
                    dirname = os.path.dirname(target_path)
                    if dirname:
                        os.makedirs(dirname, exist_ok=True)
                    temp_path = self.make_temp_file("_faiss_write")
                    try:
                        original_write(index, temp_path, *args, **kwargs)
                        try:
                            os.replace(temp_path, path)
                        except OSError:
                            shutil.copy2(temp_path, path)
                            self._remove_temp_file(temp_path)
                    finally:
                        self._remove_temp_file(temp_path)
                    return
                original_write(index, path, *args, **kwargs)

            _faiss.read_index = patched_read_index
            _faiss.write_index = patched_write_index

            from astrbot.core.db.vec_db.faiss_impl.vec_db import (
                FaissVecDB as loaded_vec_db_class,
            )
        except (ImportError, ModuleNotFoundError, SystemError, OSError) as exc:
            raise InitializationError(
                "FAISS 初始化失败，无法加载 AstrBot FaissVecDB。"
                "请检查 faiss-cpu 安装状态和 CPU 指令集兼容性。"
            ) from exc

        self._vec_db_class = loaded_vec_db_class
        return loaded_vec_db_class

    async def check_and_fix_dimension_mismatch(
        self,
        index_path: str,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        """兼容旧调用方：只检查，不删除不兼容的正式索引。"""
        if not os.path.exists(index_path):
            return

        try:
            if os.path.getsize(index_path) == 0:
                raise InitializationError("FAISS 索引文件为空。")
        except OSError:
            raise InitializationError("FAISS 索引文件不可访问。")

        try:
            import faiss  # noqa: F401
        except (ImportError, ModuleNotFoundError, SystemError, OSError) as exc:
            raise InitializationError(
                "FAISS 初始化失败，无法读取索引文件。"
                "请检查 faiss-cpu 安装状态和 CPU 指令集兼容性。"
            ) from exc

        try:
            old_index = self.faiss_read_index_safe(index_path)
        except InitializationError:
            raise
        except Exception as exc:
            raise InitializationError("FAISS 索引文件损坏或无法读取。") from exc

        old_dim = old_index.d
        new_dim = embedding_provider.get_dim()
        if old_dim == new_dim:
            return

        logger.warning(
            f"{tag('init')} 检测到 FAISS 索引维度不匹配: "
            f"索引维度={old_dim}, 当前 Embedding Provider 维度={new_dim}"
        )
        logger.warning(
            f"{tag('init')} 这通常由 Embedding 模型切换导致。"
            "正式索引不会被删除；应由影子索引服务完成重建后原子切换。"
        )
        raise InitializationError(
            f"FAISS 索引维度不匹配（索引 {old_dim}，当前 {new_dim}）；"
            "影子重建服务未执行。"
        )

    def faiss_read_index_safe(self, index_path: str):
        """通过 ASCII 临时路径桥接 FAISS read_index。"""
        if not self.needs_bridge(index_path):
            import faiss

            return faiss.read_index(index_path)
        temp_path = self.make_temp_file("_faiss_read")
        try:
            shutil.copy2(index_path, temp_path)
            import faiss

            return faiss.read_index(temp_path)
        finally:
            self._remove_temp_file(temp_path)

    def faiss_write_index_safe(self, index: Any, index_path: str) -> None:
        """写入 FAISS 索引，必要时经 ASCII 临时文件桥接。"""
        import faiss

        if not self.needs_bridge(index_path):
            faiss.write_index(index, index_path)
            return
        temp_path = self.make_temp_file("_faiss_write")
        try:
            faiss.write_index(index, temp_path)
            shutil.copy2(temp_path, index_path)
        finally:
            self._remove_temp_file(temp_path)

    @staticmethod
    def _remove_temp_file(path: str) -> None:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


__all__ = ["FaissBootstrapService"]
