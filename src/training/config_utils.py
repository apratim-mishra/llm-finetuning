"""
Training config utilities.

Provides:
- environment variable expansion for YAML configs
- deep-merge support (e.g., apply hardware "profile" overrides)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def deep_merge(base: Any, override: Any) -> Any:
    """
    Recursively merge `override` into `base`.

    - dict: keys merged recursively
    - list/scalar: override replaces base
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _expand_env(cfg)


def load_config_with_profile(
    config_path: str,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a main YAML config and optionally merge a profile YAML on top.

    Profile resolution:
    - if `profile_path` is provided, it is resolved relative to the main config file
      when given as a relative path.
    - if the main config contains a top-level `profile:` key, it is applied first,
      then `profile_path` (CLI) is applied last.
    """
    config_file = Path(config_path)
    cfg = _load_yaml(config_file)

    base_dir = config_file.parent

    inline_profile = cfg.pop("profile", None)
    if inline_profile:
        inline_path = Path(inline_profile)
        if not inline_path.is_absolute():
            inline_path = base_dir / inline_path
        cfg = deep_merge(cfg, _load_yaml(inline_path))

    if profile_path:
        p = Path(profile_path)
        if not p.is_absolute():
            p = base_dir / p
        cfg = deep_merge(cfg, _load_yaml(p))

    return cfg

