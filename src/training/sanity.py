"""
Training-time environment checks.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from rich.console import Console


def require_cuda(console: Optional[Console] = None, env_allow_cpu: str = "LLM_FT_ALLOW_CPU") -> None:
    """
    Fail fast if CUDA isn't available.

    Set `LLM_FT_ALLOW_CPU=1` to bypass (useful for debugging only).
    """
    if torch.cuda.is_available():
        return

    allow = os.environ.get(env_allow_cpu, "").lower() in {"1", "true", "yes"}
    if allow:
        if console:
            console.print(f"[yellow]Warning: CUDA not available; continuing due to {env_allow_cpu}=1[/yellow]")
        return

    if console:
        console.print("[red]CUDA not available. This script is intended for NVIDIA GPUs.[/red]")
        console.print(f"[dim]Set {env_allow_cpu}=1 to bypass for debugging.[/dim]")
    raise SystemExit(1)

