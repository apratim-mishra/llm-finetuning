"""
Dataset manifest utilities.

Used by data preparation scripts to produce reproducible metadata alongside generated JSONL files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for a file in a streaming manner."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_info(path: Path) -> Dict[str, Any]:
    """Return path/size/hash metadata for a file."""
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_files_manifest(files: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Build a manifest mapping logical names to file metadata."""
    return {name: file_info(path) for name, path in files.items() if path.exists()}


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Write a manifest JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)

