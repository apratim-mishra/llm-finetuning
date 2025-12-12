"""
Preference data utilities (e.g., for DPO).

These helpers let you build preference pairs from multiple candidate completions and a scoring
function (reward).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float


def select_preference_pair(
    prompt: str,
    candidates: List[str],
    ground_truth: str,
    reward_fn: Callable[[List[str], List[str]], List[float]],
    min_reward_gap: float = 0.0,
) -> Optional[PreferencePair]:
    """
    Select (chosen, rejected) from candidates by scoring against ground truth.

    Returns None if:
    - fewer than 2 candidates
    - all candidates collapse to the same string
    - reward gap is below `min_reward_gap`
    """
    if len(candidates) < 2:
        return None

    normalized = [c.strip() for c in candidates]
    unique = list(dict.fromkeys([c for c in normalized if c]))
    if len(unique) < 2:
        return None

    gts = [ground_truth for _ in unique]
    rewards = reward_fn(unique, gts)
    if not rewards or len(rewards) != len(unique):
        return None

    best_idx = max(range(len(unique)), key=lambda i: rewards[i])
    worst_idx = min(range(len(unique)), key=lambda i: rewards[i])

    chosen = unique[best_idx]
    rejected = unique[worst_idx]
    chosen_reward = float(rewards[best_idx])
    rejected_reward = float(rewards[worst_idx])

    if chosen.strip() == rejected.strip():
        return None

    if (chosen_reward - rejected_reward) < min_reward_gap:
        return None

    return PreferencePair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        chosen_reward=chosen_reward,
        rejected_reward=rejected_reward,
    )

