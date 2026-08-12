"""
人格信息处理模块
"""

from typing import TYPE_CHECKING, Any

import aiosqlite

from ...log import logger, tag

if TYPE_CHECKING:
    from .utils import PageApiUtils


class PersonaHandler:
    """人格信息处理器"""

    def __init__(self, utils: "PageApiUtils"):
        """
        初始化人格信息处理器

        Args:
            utils: PageApiUtils 工具实例
        """
        self.utils = utils

    async def list_personas(self, memory_engine) -> dict[str, Any]:
        """
        列出数据库里所有 distinct persona_id，供前端下拉使用

        Returns:
            包含 persona_id 列表的字典
        """
        db_path = getattr(memory_engine, "db_path", None)
        if not db_path:
            return self.utils.error("MemoryEngine db_path unavailable")

        try:
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT DISTINCT json_extract(metadata, '$.persona_id') AS p "
                    "FROM documents "
                    "WHERE json_extract(metadata, '$.persona_id') IS NOT NULL "
                    "AND json_extract(metadata, '$.persona_id') != '' "
                    "ORDER BY p"
                )
                rows = await cursor.fetchall()
            items = [row[0] for row in rows if row[0]]
            return self.utils.ok({"items": items})
        except Exception as exc:
            logger.error(f"{tag('page')} 列出 persona 失败: {exc}", exc_info=True)
            return self.utils.error(str(exc))
