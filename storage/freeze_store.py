"""冻结 persona 的持久化状态。

冻结（freeze）语义：该 persona 的记忆进入冷存储状态——
不参与召回，且在冻结期间免疫自动清理（重要性照常衰减）；
解冻（unfreeze）后进入一段宽限期，同样免疫清理，避免"解冻即被删"。

fail-closed 约定：状态文件"存在但读不出来"（损坏、截断、权限不足、
结构非法）与"文件不存在"是两码事。前者抛
:class:`FreezeStateUnreadable`，让调用方拒绝写入、跳过清理、放弃召回——
宁可这一轮什么都不做，也不能把"读不到"当成"没有冻结"，否则冻结记录会被
空视图覆盖丢失、冻结记忆会重新被召回乃至被自动删除。
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..log import logger, tag


class FreezeStateUnreadable(RuntimeError):
    """冻结状态文件存在但无法读取/解析。

    调用方必须按 fail-closed 处理：不写入、不清理、不按"未冻结"继续。
    """


def _iso(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _parse_iso(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class FreezeStore:
    """以独立 JSON 文件记录冻结列表（persona_id → 冻结/宽限信息）。"""

    SCHEMA = 1

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def load(self) -> dict[str, dict[str, Any]]:
        """读取冻结记录。

        文件不存在（从未冻结过）返回空表；文件存在但内容不可读、为空或结构
        非法时抛 :class:`FreezeStateUnreadable`，由调用方按 fail-closed 处理。
        """
        if not self.path.exists():
            return {}
        try:
            raw = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        except OSError as exc:
            raise FreezeStateUnreadable(
                f"冻结状态文件无法读取: {self.path.name}: {exc}"
            ) from exc
        if not raw.strip():
            # 空文件只可能来自外部清空/截断（本类始终原子写入非空内容）
            raise FreezeStateUnreadable(
                f"冻结状态文件内容为空: {self.path.name}"
            )
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise FreezeStateUnreadable(
                f"冻结状态文件解析失败: {self.path.name}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise FreezeStateUnreadable(
                f"冻结状态文件根节点不是对象: {self.path.name}"
            )
        personas = payload.get("personas")
        if not isinstance(personas, dict):
            raise FreezeStateUnreadable(
                f"冻结状态文件缺少合法的 personas 表: {self.path.name}"
            )
        records: dict[str, dict[str, Any]] = {}
        for key, value in personas.items():
            if not isinstance(value, dict):
                raise FreezeStateUnreadable(
                    f"冻结状态文件记录格式非法: {self.path.name}"
                )
            records[str(key)] = value
        return records

    async def _write(self, personas: dict[str, dict[str, Any]]) -> None:
        payload = {"schema": self.SCHEMA, "personas": personas}
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

        def _atomic_write() -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                dir=str(self.path.parent),
                prefix=f".{self.path.name}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.path)
            except BaseException:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                raise

        await asyncio.to_thread(_atomic_write)

    async def freeze(self, persona_id: str) -> dict[str, Any]:
        """冻结指定 persona；重复冻结只刷新冻结时间。"""
        persona = str(persona_id or "").strip()
        if not persona:
            raise ValueError("persona 不能为空")
        async with self._lock:
            personas = await self.load()
            record = {
                "frozen_at": _iso(time.time()),
                "grace_until": None,
            }
            personas[persona] = record
            await self._write(personas)
        logger.info(f"{tag('freeze')} 已冻结 persona {persona}")
        return record

    async def unfreeze(
        self,
        persona_id: str,
        *,
        grace_days: float = 0.0,
    ) -> dict[str, Any]:
        """解冻指定 persona；grace_days > 0 时保留一段清理宽限期。

        对从未冻结过的 persona 是 no-op（不写入任何记录）。
        """
        persona = str(persona_id or "").strip()
        if not persona:
            raise ValueError("persona 不能为空")
        now = time.time()
        grace_until = (
            _iso(now + max(0.0, float(grace_days)) * 86400.0)
            if grace_days and float(grace_days) > 0
            else None
        )
        async with self._lock:
            personas = await self.load()
            if persona not in personas:
                # 从未冻结过的 persona：解冻是 no-op。绝不能因为 grace_days>0
                # 就凭空写入一条豁免记录（那等于给它白送免删窗口）。
                return {"frozen_at": None, "grace_until": None}
            if grace_until is None:
                personas.pop(persona, None)
            else:
                personas[persona] = {
                    "frozen_at": personas.get(persona, {}).get("frozen_at"),
                    "grace_until": grace_until,
                }
            await self._write(personas)
        logger.info(
            f"{tag('freeze')} 已解冻 persona {persona}"
            + (f"（清理宽限至 {grace_until}）" if grace_until else "")
        )
        return {"frozen_at": None, "grace_until": grace_until}

    async def frozen_personas(self) -> set[str]:
        """当前处于冻结状态的 persona（解冻宽限期内不算）。

        宽限时间无法解析时按"仍冻结"处理（fail-closed），避免坏值让冻结
        persona 重新参与召回。
        """
        personas = await self.load()
        frozen: set[str] = set()
        for persona, record in personas.items():
            grace_raw = record.get("grace_until")
            if grace_raw:
                if _parse_iso(grace_raw) is None:
                    frozen.add(persona)
                continue
            if record.get("frozen_at"):
                frozen.add(persona)
        return frozen

    async def protected_personas(self) -> set[str]:
        """冻结中 + 仍在解冻宽限期内的 persona（自动清理要跳过）。

        冻结期继续正常衰减（久未使用就该变旧），但绝不自动删除；
        解冻后保留一段宽限期，避免"解冻即被清理"。宽限时间无法解析时同样
        计入保护名单（fail-closed：宁可多保护一轮，也不误删）。
        """
        now = time.time()
        personas = await self.load()
        protected: set[str] = set()
        for persona, record in personas.items():
            grace_raw = record.get("grace_until")
            if grace_raw:
                grace_until = _parse_iso(grace_raw)
                if grace_until is None or grace_until > now:
                    protected.add(persona)
                continue
            if record.get("frozen_at"):
                protected.add(persona)
        return protected

    async def entries(self) -> list[dict[str, Any]]:
        """冻结/宽限记录列表，供命令与页面展示。"""
        now = time.time()
        personas = await self.load()
        items: list[dict[str, Any]] = []
        for persona, record in sorted(personas.items()):
            grace_raw = record.get("grace_until")
            grace_until = _parse_iso(grace_raw)
            if grace_raw:
                # 坏值按"仍在宽限"展示，与 protected_personas 的保护口径一致
                state = "grace" if grace_until is None or grace_until > now else "expired"
            elif record.get("frozen_at"):
                state = "frozen"
            else:
                state = "expired"
            items.append(
                {
                    "persona_id": persona,
                    "frozen_at": record.get("frozen_at"),
                    "grace_until": grace_raw,
                    "state": state,
                }
            )
        return items

    async def grace_expired_personas(self) -> list[str]:
        """宽限期已过、需要从状态文件中清理的记录。

        只清理宽限时间可解析且确实已过期的记录；无法解析的坏值保留在文件里
        （并按 :meth:`protected_personas` 继续受保护），避免静默丢掉保护。
        """
        now = time.time()
        personas = await self.load()
        expired = [
            persona
            for persona, record in personas.items()
            if record.get("grace_until")
            and (_parse_iso(record.get("grace_until")) or float("inf")) <= now
        ]
        if not expired:
            return []
        async with self._lock:
            personas = await self.load()
            for persona in expired:
                personas.pop(persona, None)
            await self._write(personas)
        return expired


__all__ = ["FreezeStore", "FreezeStateUnreadable"]
