"""DecayScheduler domain and state-machine tests."""

import asyncio
import builtins
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from livingmemory_cm.core.schedulers.decay_scheduler import DecayScheduler


@pytest.fixture
def mock_memory_engine():
    """创建mock的MemoryEngine"""
    engine = Mock()
    engine.apply_daily_decay = AsyncMock(return_value=10)
    engine.cleanup_old_memories = AsyncMock(return_value=5)
    engine.maintain_storage = AsyncMock(
        return_value={"success": True, "bytes_reclaimed": 1024 * 1024}
    )
    engine.config = {
        "cleanup_days_threshold": 30,
        "cleanup_importance_threshold": 0.3,
    }
    return engine


@pytest.fixture
def mock_db_migration():
    """创建mock的DBMigration"""
    migration = Mock()
    migration.db_path = "/tmp/test.db"
    migration.create_backup = AsyncMock(return_value="/tmp/backup.db")
    return migration


class TestDecaySchedulerStateManagement:
    """测试状态文件管理"""

    @pytest.mark.asyncio
    async def test_load_state_with_corrupted_json(
        self, tmp_path, mock_memory_engine
    ):
        """测试损坏的JSON文件返回空字典"""
        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        # 写入损坏的JSON
        scheduler._state_file.write_text("not a valid json{", encoding="utf-8")

        state = await scheduler._load_state()
        assert state == {}

class TestDecaySchedulerMissedDaysCalculation:
    """测试错过天数计算"""

    @pytest.mark.asyncio
    async def test_calculate_missed_days_no_previous_run(
        self, tmp_path, mock_memory_engine
    ):
        """测试从未执行过时返回0"""
        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        missed = await scheduler._calculate_missed_days()
        assert missed == 0

    @pytest.mark.asyncio
    async def test_calculate_missed_days_same_day(self, tmp_path, mock_memory_engine):
        """测试同一天不算错过"""
        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        today = datetime.now().strftime("%Y-%m-%d")
        await scheduler._set_last_decay_date(today)

        missed = await scheduler._calculate_missed_days()
        assert missed == 0

class TestDecaySchedulerExecution:
    """测试衰减执行"""

    @pytest.mark.asyncio
    async def test_execute_decay_with_zero_rate(self, tmp_path, mock_memory_engine):
        """测试衰减率为0时跳过衰减"""
        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.0,  # 衰减率为0
            data_dir=str(tmp_path),
        )

        success = await scheduler._execute_decay(1)
        assert success is True
        mock_memory_engine.apply_daily_decay.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_decay_multiple_days(self, tmp_path, mock_memory_engine):
        """测试补偿多天衰减"""
        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        success = await scheduler._execute_decay(5)

        assert success is True
        mock_memory_engine.apply_daily_decay.assert_called_once_with(0.01, 5)

    @pytest.mark.asyncio
    async def test_execute_decay_with_auto_cleanup_disabled(
        self, tmp_path, mock_memory_engine
    ):
        """测试 cleanup_days_threshold == 0 时不执行清理"""
        mock_memory_engine.config["cleanup_days_threshold"] = 0

        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        success = await scheduler._execute_decay(1)

        assert success is True
        mock_memory_engine.cleanup_old_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_decay_handles_cleanup_error(
        self, tmp_path, mock_memory_engine
    ):
        """测试清理失败时仍然继续执行"""
        mock_memory_engine.cleanup_old_memories = AsyncMock(
            side_effect=Exception("Cleanup failed")
        )

        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        success = await scheduler._execute_decay(1)

        # 清理失败不影响整体成功
        assert success is True
        mock_memory_engine.apply_daily_decay.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_decay_handles_maintenance_error(
        self, tmp_path, mock_memory_engine
    ):
        """测试存储维护失败时记录警告"""
        mock_memory_engine.maintain_storage = AsyncMock(
            side_effect=Exception("Maintenance failed")
        )

        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        success = await scheduler._execute_decay(1)

        # 维护失败不影响整体成功
        assert success is True

    @pytest.mark.asyncio
    async def test_execute_decay_handles_decay_error(
        self, tmp_path, mock_memory_engine
    ):
        """测试衰减本身失败时返回False"""
        mock_memory_engine.apply_daily_decay = AsyncMock(
            side_effect=Exception("Decay failed")
        )

        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        success = await scheduler._execute_decay(1)

        assert success is False


class TestDecaySchedulerBackup:
    """测试备份功能"""

    @pytest.mark.asyncio
    async def test_cleanup_old_backups(self, tmp_path, mock_memory_engine, mock_db_migration):
        """测试清理过期备份"""
        # 创建backup目录和测试文件
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # 创建一个旧备份文件（10天前）
        old_backup = backup_dir / "test_backup_old.db"
        old_backup.touch()
        old_time = datetime.now().timestamp() - 10 * 86400
        import os
        os.utime(old_backup, (old_time, old_time))

        # 创建一个新备份文件（1天前）
        new_backup = backup_dir / "test_backup_new.db"
        new_backup.touch()

        mock_db_migration.db_path = str(tmp_path / "test.db")

        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
            db_migration=mock_db_migration,
            backup_keep_days=7,
        )

        await scheduler._cleanup_old_backups()

        # 旧文件应该被删除，新文件应该保留
        assert not old_backup.exists()
        assert new_backup.exists()


class TestDecaySchedulerScheduling:
    """测试调度逻辑"""

    @pytest.mark.asyncio
    async def test_check_and_execute_already_run_today(
        self, tmp_path, mock_memory_engine
    ):
        """测试今天已执行时跳过"""
        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        # 设置为今天已执行
        today = datetime.now().strftime("%Y-%m-%d")
        await scheduler._set_last_decay_date(today)

        await scheduler._check_and_execute()

        # 不应该再次执行衰减
        mock_memory_engine.apply_daily_decay.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_execute_with_missed_days(
        self, tmp_path, mock_memory_engine
    ):
        """测试有错过天数时补偿执行"""
        scheduler = DecayScheduler(
            memory_engine=mock_memory_engine,
            decay_rate=0.01,
            data_dir=str(tmp_path),
        )

        # 设置为3天前
        three_days_ago = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        await scheduler._set_last_decay_date(three_days_ago)

        await scheduler._check_and_execute()

        # 应该执行3天的衰减（错过2天 + 今天1天）
        mock_memory_engine.apply_daily_decay.assert_called_once_with(0.01, 3)

@pytest.mark.asyncio
async def test_state_io_falls_back_without_aiofiles(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """可选 aiofiles 缺失时仍使用标准库安全读写状态文件。"""
    real_import = builtins.__import__

    def import_without_aiofiles(name, *args, **kwargs):
        if name == "aiofiles":
            raise ImportError("aiofiles intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_aiofiles)
    scheduler = DecayScheduler(
        memory_engine=Mock(),
        decay_rate=0.01,
        data_dir=str(tmp_path),
    )

    await scheduler._save_state({"last_decay_date": "2026-06-01"})

    assert scheduler._state_file.exists()
    assert await scheduler._load_state() == {"last_decay_date": "2026-06-01"}
