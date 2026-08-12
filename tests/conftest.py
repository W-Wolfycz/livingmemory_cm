"""Local unit-test boundary for LivingMemoryCM.

Only the small public ``astrbot.api`` surface imported by the plugin is
stubbed here.  ``astrbot.core`` is deliberately left untouched: these tests
do not pretend to provide an AstrBot runtime, lifecycle, provider, platform,
or browser environment.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import astrbot as _astrbot_package
import pytest


def _identity_decorator(function: Callable[..., Any]) -> Callable[..., Any]:
    return function


def _decorator(*_args: Any, **_kwargs: Any):
    return _identity_decorator


def _command_group(*_args: Any, **_kwargs: Any):
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        function.command = _decorator  # type: ignore[attr-defined]
        return function

    return decorate


class _PermissionType(Enum):
    ADMIN = "admin"
    MEMBER = "member"


class _MessageType(Enum):
    FRIEND_MESSAGE = "friend"
    GROUP_MESSAGE = "group"


class _AstrMessageEvent:
    """Type marker only; tests provide their own event fakes."""


class _AstrBotMessage:
    """Type marker required by AstrBot's StarTools annotations."""


class _MessageMember:
    """Type marker required by AstrBot's StarTools annotations."""


class _MessageEventResult:
    """Type marker only; tests assert domain output, not AstrBot rendering."""


class _ProviderRequest:
    """Type marker only; tests pass explicit lightweight request objects."""


class _Context:
    """Minimal type marker used by annotations and ``isinstance`` guards."""


class _Star:
    def __init__(self, context: _Context | None = None) -> None:
        self.context = context


class _StarTools:
    @staticmethod
    def get_data_dir(plugin_name: str) -> Path:
        return Path.cwd() / ".astrbot-test-data" / plugin_name


def _install_astrbot_api_stub() -> None:
    api = ModuleType("astrbot.api")
    api.__path__ = []  # type: ignore[attr-defined]
    api.logger = logging.getLogger("livingmemory_cm.tests")
    # Some imported AstrBot core type modules reference the public shared-
    # preferences handle at module scope.  Domain tests never call it.
    api.sp = object()

    event = ModuleType("astrbot.api.event")
    event.__path__ = []  # type: ignore[attr-defined]
    event.AstrMessageEvent = _AstrMessageEvent
    event.MessageEventResult = _MessageEventResult

    event_filter = ModuleType("astrbot.api.event.filter")
    event_filter.PermissionType = _PermissionType
    event_filter.permission_type = _decorator
    event_filter.on_llm_request = _decorator
    event_filter.on_decorating_result = _decorator
    event_filter.after_message_sent = _decorator
    event_filter.command_group = _command_group
    event.filter = event_filter

    provider = ModuleType("astrbot.api.provider")
    provider.ProviderRequest = _ProviderRequest

    platform = ModuleType("astrbot.api.platform")
    platform.AstrBotMessage = _AstrBotMessage
    platform.MessageMember = _MessageMember
    platform.MessageType = _MessageType

    star = ModuleType("astrbot.api.star")
    star.Context = _Context
    star.Star = _Star
    star.StarTools = _StarTools
    star.register = _decorator

    api.event = event
    api.provider = provider
    api.platform = platform
    api.star = star

    sys.modules.update(
        {
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.event.filter": event_filter,
            "astrbot.api.provider": provider,
            "astrbot.api.platform": platform,
            "astrbot.api.star": star,
        }
    )
    _astrbot_package.api = api


_install_astrbot_api_stub()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    integration_items = [
        item.nodeid for item in items if item.get_closest_marker("integration")
    ]
    if integration_items:
        formatted = "\n".join(f"- {nodeid}" for nodeid in integration_items)
        raise pytest.UsageError(
            "本地 tests/ 不接收 AstrBot 集成测试；请移交远端部署验收：\n"
            f"{formatted}"
        )
