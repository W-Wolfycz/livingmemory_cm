"""Local unit-test boundary for LivingMemoryCM.

Only the small public ``astrbot.api`` surface imported by the plugin is
stubbed here, plus a *minimal* ``astrbot.core`` type tree and a bare
``quart.request`` placeholder.

These fakes only provide the interfaces the plugin's own code imports at
module scope; they deliberately do **not** simulate AstrBot runtime behaviour
(lifecycle, provider, platform, event bus, browser, web request handling).
AstrBot core compatibility is verified on the Windows test/reload side, not by
these local tests.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Generic, TypeVar

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


# ---------------------------------------------------------------------------
# Minimal fake ``astrbot.core`` tree.
#
# The plugin imports these names at module scope; the fakes below only model
# the small amount of behaviour the plugin actually relies on (field storage,
# ``mark_as_temp``, subclassing, ``isinstance`` guards).  They are *not* a
# stand-in for the real AstrBot runtime: AstrBot compatibility is accepted on
# the Windows test/reload side.
# ---------------------------------------------------------------------------


class _TextPart:
    """Minimal stand-in for ``astrbot.core.agent.message.TextPart``.

    Real AstrBot stores ``text`` on a pydantic model and exposes
    ``mark_as_temp()`` to keep a part out of persisted history.  Tests only
    need construction + the marker method.
    """

    def __init__(self, text: str = "") -> None:
        self.text = text
        self._no_save = False

    def mark_as_temp(self, _no_save: bool = True) -> "_TextPart":
        self._no_save = _no_save
        return self


_TContext = TypeVar("_TContext")


class _ContextWrapper(Generic[_TContext]):
    """Type marker for ``astrbot.core.agent.run_context.ContextWrapper``.

    The plugin uses this only in annotations; the real class is a generic
    pydantic dataclass carrying ``context`` / ``messages``.
    """


class _AstrAgentContext:
    """Type marker for ``astrbot.core.astr_agent_context.AstrAgentContext``.

    Real AstrBot models this as ``context`` + ``event`` + ``extra``; the plugin
    only references it in tool type annotations.
    """


class _FunctionTool(Generic[_TContext]):
    """Minimal base for ``astrbot.core.agent.tool.FunctionTool``.

    The real class is a pydantic dataclass holding ``name``/``description``/
    ``parameters`` plus handler bookkeeping.  This fake deliberately omits that
    machinery so the local suite has no dependency on jsonschema/mcp/deprecated;
    plugin subclasses declare their own fields and remain plain pydantic
    dataclasses.
    """

    async def call(self, context: _ContextWrapper[_TContext], **kwargs: Any) -> Any:
        raise NotImplementedError(
            "FunctionTool.call() must be implemented by subclasses."
        )


class _ToolExecResult:
    """Type placeholder for ``astrbot.core.agent.tool.ToolExecResult``.

    In real AstrBot this is ``str | mcp.types.CallToolResult``; the plugin
    always returns JSON strings, so the fake only needs to be a resolvable name
    for return annotations.
    """


class _Provider:
    """Minimal base for ``astrbot.core.provider.provider.Provider``."""


class _EmbeddingProvider(_Provider):
    """Minimal base for ``astrbot.core.provider.provider.EmbeddingProvider``.

    The plugin only does ``isinstance(provider, EmbeddingProvider)`` guards;
    tests supply their own fake providers.
    """


class _FaissVecDB:
    """Type/factory placeholder for ``astrbot.core.db.vec_db.faiss_impl.vec_db.FaissVecDB``.

    Real AstrBot lazily imports this only after ``faiss`` is available; tests
    substitute their own fake vector stores.
    """


class _Plain:
    """Plain-text component marker (``astrbot.core.message.components.Plain``)."""

    def __init__(self, text: str = "") -> None:
        self.text = text


class _At:
    """At component marker."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _AtAll:
    """AtAll component marker."""


class _Face:
    """Face component marker."""


class _File:
    """File component marker."""


class _Forward:
    """Forward component marker."""


class _Image:
    """Image component marker."""


class _Record:
    """Record component marker."""


class _Reply:
    """Reply component marker."""


class _Video:
    """Video component marker."""


def _make_module(name: str, namespace: dict[str, Any] | None = None) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    if namespace:
        module.__dict__.update(namespace)
    sys.modules[name] = module
    return module


def _install_astrbot_package() -> None:
    """Create a synthetic ``astrbot`` package so no real AstrBot import is needed."""
    astrbot_pkg = _make_module("astrbot")
    astrbot_pkg.__path__ = []  # type: ignore[attr-defined]
    astrbot_pkg.__version__ = "fake-local-testing"


def _install_astrbot_api_stub() -> None:
    api = _make_module("astrbot.api")
    api.logger = logging.getLogger("livingmemory_cm.tests")
    # Some imported AstrBot core type modules reference the public shared-
    # preferences handle at module scope.  Domain tests never call it.
    api.sp = object()

    event = _make_module("astrbot.api.event")
    event.AstrMessageEvent = _AstrMessageEvent
    event.MessageEventResult = _MessageEventResult

    event_filter = _make_module("astrbot.api.event.filter")
    event_filter.PermissionType = _PermissionType
    event_filter.permission_type = _decorator
    event_filter.on_llm_request = _decorator
    event_filter.on_decorating_result = _decorator
    event_filter.after_message_sent = _decorator
    event_filter.command_group = _command_group
    event.filter = event_filter

    provider = _make_module("astrbot.api.provider")
    provider.ProviderRequest = _ProviderRequest

    platform = _make_module("astrbot.api.platform")
    platform.AstrBotMessage = _AstrBotMessage
    platform.MessageMember = _MessageMember
    platform.MessageType = _MessageType

    star = _make_module("astrbot.api.star")
    star.Context = _Context
    star.Star = _Star
    star.StarTools = _StarTools
    star.register = _decorator

    api.event = event
    api.provider = provider
    api.platform = platform
    api.star = star

    import astrbot as _astrbot_package  # noqa: F401  (synthetic, installed above)

    _astrbot_package.api = api


def _install_astrbot_core_fake() -> None:
    """Install the minimal ``astrbot.core`` fake tree used by plugin imports."""

    agent = _make_module("astrbot.core.agent")
    message = _make_module("astrbot.core.agent.message")
    message.TextPart = _TextPart
    message.ContentPart = _TextPart  # minimal alias used by annotations

    run_context = _make_module("astrbot.core.agent.run_context")
    run_context.ContextWrapper = _ContextWrapper
    run_context.TContext = _TContext

    tool = _make_module("astrbot.core.agent.tool")
    tool.FunctionTool = _FunctionTool
    tool.ToolExecResult = _ToolExecResult
    tool.ToolSchema = _FunctionTool  # minimal alias for subclass annotations

    astr_agent_context = _make_module("astrbot.core.astr_agent_context")
    astr_agent_context.AstrAgentContext = _AstrAgentContext
    astr_agent_context.AgentContextWrapper = _ContextWrapper[
        _AstrAgentContext
    ]

    agent.message = message
    agent.run_context = run_context
    agent.tool = tool

    provider_pkg = _make_module("astrbot.core.provider")
    provider = _make_module("astrbot.core.provider.provider")
    provider.Provider = _Provider
    provider.EmbeddingProvider = _EmbeddingProvider
    provider_pkg.provider = provider

    message_pkg = _make_module("astrbot.core.message")
    components = _make_module("astrbot.core.message.components")
    components.At = _At
    components.AtAll = _AtAll
    components.Face = _Face
    components.File = _File
    components.Forward = _Forward
    components.Image = _Image
    components.Plain = _Plain
    components.Record = _Record
    components.Reply = _Reply
    components.Video = _Video
    message_pkg.components = components

    db_pkg = _make_module("astrbot.core.db")
    vec_db_pkg = _make_module("astrbot.core.db.vec_db")
    faiss_impl_pkg = _make_module("astrbot.core.db.vec_db.faiss_impl")
    vec_db = _make_module("astrbot.core.db.vec_db.faiss_impl.vec_db")
    vec_db.FaissVecDB = _FaissVecDB
    db_pkg.vec_db = vec_db_pkg
    vec_db_pkg.faiss_impl = faiss_impl_pkg
    faiss_impl_pkg.vec_db = vec_db

    core_pkg = _make_module("astrbot.core")
    core_pkg.agent = agent
    core_pkg.astr_agent_context = astr_agent_context
    core_pkg.provider = provider_pkg
    core_pkg.message = message_pkg
    core_pkg.db = db_pkg

    import astrbot as _astrbot_package  # noqa: F401

    _astrbot_package.core = core_pkg


def _install_quart_stub() -> None:
    """Provide a bare ``quart.request`` so ``page_api_modules`` can be imported.

    The plugin's page-API handlers read ``quart.request`` at request time; tests
    replace each module's ``request`` with their own mock via
    ``_patch_page_request`` before calling handlers, so the real Quart request
    context is never exercised locally.  Keeping this stub means the local suite
    does not need to install the web framework that only AstrBot hosts need.
    """
    quart = _make_module("quart")
    quart.request = object()  # replaced per-module by tests before handler calls


_install_astrbot_package()
_install_astrbot_api_stub()
_install_astrbot_core_fake()
_install_quart_stub()


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
