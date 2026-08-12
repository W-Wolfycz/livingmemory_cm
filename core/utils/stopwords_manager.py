"""
停用词管理器 - 管理停用词表
"""

from pathlib import Path

from ...log import logger, tag

from ..models.default_stopwords import DEFAULT_STOPWORDS


class StopwordsManager:
    """停用词管理器"""

    def __init__(
        self,
        stopwords_dir: str | None = None,
    ):
        """
        初始化停用词管理器

        Args:
            stopwords_dir: 停用词文件存储目录（可选，如果未提供则使用内置停用词）
        """
        # 获取内置停用词目录（仓库中的 static/stopwords）
        self.builtin_stopwords_dir = (
            Path(__file__).parent.parent.parent / "static" / "stopwords"
        )

        # 用户自定义停用词目录（用于保存用户添加的停用词）
        if stopwords_dir:
            self.custom_stopwords_dir = Path(stopwords_dir)
            self.custom_stopwords_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.custom_stopwords_dir = None

        self.stopwords: set[str] = set()
        self.custom_stopwords: set[str] = set()

    async def load_stopwords(
        self,
        source: str = "hit",
        custom_words: list | None = None,
    ) -> set[str]:
        """
        加载停用词表

        Args:
            source: 停用词来源 ("hit" 或自定义文件路径)
            custom_words: 用户自定义停用词列表

        Returns:
            Set[str]: 停用词集合
        """
        logger.info(f"{tag('stopwords')} 开始加载停用词表: source={source}")

        # 1. 加载标准停用词表
        if source == "hit":
            # 从仓库内置目录加载
            stopwords_path = await self.get_stopwords(source)
            filepath = Path(stopwords_path) if stopwords_path else None
            if filepath and filepath.exists():
                self.stopwords = await self._load_from_file(filepath)
                logger.info(f"{tag('stopwords')} 从内置目录加载停用词: {filepath}")
            else:
                logger.warning(f"{tag('stopwords')} 内置停用词文件不可用，使用后备停用词")
                self.stopwords = self._get_builtin_stopwords()
        else:
            # 使用自定义文件路径
            custom_path = Path(source)
            if custom_path.exists():
                self.stopwords = await self._load_from_file(custom_path)
            else:
                logger.error(f"{tag('stopwords')} 自定义停用词文件不存在: {source}")
                self.stopwords = self._get_builtin_stopwords()

        # 2. 添加用户自定义停用词
        if custom_words:
            self.custom_stopwords = set(custom_words)
            self.stopwords.update(self.custom_stopwords)
            logger.info(f"{tag('stopwords')} 添加自定义停用词: {len(custom_words)} 个")

        logger.info(f"{tag('stopwords')} 停用词表加载完成，共 {len(self.stopwords)} 个词")
        return self.stopwords

    async def _load_from_file(self, filepath: Path) -> set[str]:
        """
        从文件加载停用词

        Args:
            filepath: 文件路径

        Returns:
            Set[str]: 停用词集合
        """
        try:
            import aiofiles

            stopwords = set()
            async with aiofiles.open(filepath, encoding="utf-8") as f:
                async for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):  # 跳过空行和注释
                        stopwords.add(word)

            logger.debug(f"{tag('stopwords')} 从文件加载停用词: {filepath}, 共 {len(stopwords)} 个")
            return stopwords

        except Exception as e:
            logger.error(f"{tag('stopwords')} 读取停用词文件失败: {filepath}, 错误: {e}")
            return set()

    def _get_builtin_stopwords(self) -> set[str]:
        """
        获取内置的基础停用词表（作为后备方案）

        Returns:
            Set[str]: 基础停用词集合
        """
        builtin = set(DEFAULT_STOPWORDS)
        logger.warning(f"{tag('stopwords')} 使用内置停用词表（后备方案），共 {len(builtin)} 个词")
        return builtin

    async def get_stopwords(self, source: str = "hit") -> str | None:
        """
        获取停用词文件路径。

        优先返回仓库内置停用词文件；如果文件不存在，则在用户自定义目录
        写入一份内置后备停用词，避免调用方拿到不存在的路径。

        Args:
            source: 停用词来源 ("hit")

        Returns:
            停用词文件的绝对路径字符串；若发生异常则返回 None。
        """
        try:
            filename = f"stopwords_{source}.txt"
            filepath = self.builtin_stopwords_dir / filename

            # 检查内置文件是否存在
            if filepath.exists():
                return str(filepath)

            logger.warning(f"{tag('stopwords')} 内置停用词文件不存在: {filepath}")
            if self.custom_stopwords_dir:
                fallback_path = self.custom_stopwords_dir / filename
                await self._write_fallback_stopwords(fallback_path)
                return str(fallback_path)
            return None
        except Exception as e:
            logger.error(f"{tag('stopwords')} 获取停用词文件失败: {e}")
            return None

    async def _write_fallback_stopwords(self, filepath: Path) -> None:
        if filepath.exists():
            return

        try:
            import aiofiles

            filepath.parent.mkdir(parents=True, exist_ok=True)
            words = sorted(self._get_builtin_stopwords())
            async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                await f.write("# Generated fallback stopwords for LivingMemory\n")
                for word in words:
                    await f.write(f"{word}\n")
            logger.info(f"{tag('stopwords')} 已生成后备停用词文件: {filepath}")
        except Exception as e:
            logger.error(f"{tag('stopwords')} 生成后备停用词文件失败: {e}")
            raise


# 全局单例
_stopwords_manager: StopwordsManager | None = None


def get_stopwords_manager() -> StopwordsManager:
    """获取全局停用词管理器单例"""
    global _stopwords_manager
    if _stopwords_manager is None:
        _stopwords_manager = StopwordsManager()
    return _stopwords_manager
