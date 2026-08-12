"""
Backend i18n module for bot command responses (CM-only fork: zh-only).
Loads JSON translation file from core/i18n/zh.json.
"""

import json
from pathlib import Path

from ..log import logger, tag

_translations: dict = {}


def init(language: str = "zh"):
    """Load zh translations. language 参数保留为兼容签名，不再使用。"""
    global _translations
    path = Path(__file__).parent / "i18n" / "zh.json"
    try:
        with open(path, encoding="utf-8") as f:
            _translations = json.load(f)
    except Exception as exc:
        logger.error(f"{tag('i18n')} Failed to load i18n zh.json: {exc}")
        _translations = {}


def _get(data: dict, key: str):
    parts = key.split(".")
    for part in parts:
        if isinstance(data, dict) and part in data:
            data = data[part]
        else:
            return None
    return data


def t(key: str, **kwargs) -> str:
    """Get translated string by dot-notation key."""
    value = _get(_translations, key)
    if value is None:
        logger.warning(f"{tag('i18n')} i18n key missing: {key}")
        return key
    if not isinstance(value, str):
        return str(value)
    try:
        return value.format(**kwargs)
    except Exception as exc:
        logger.warning(f"{tag('i18n')} i18n format error for key '{key}': {exc}")
        return value


def t_list(key: str) -> list[str]:
    """Get translated list of strings by dot-notation key."""
    value = _get(_translations, key)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
