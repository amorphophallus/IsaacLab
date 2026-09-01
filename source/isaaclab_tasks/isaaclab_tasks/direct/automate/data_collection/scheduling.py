"""Pure scheduling helpers for vectorized AutoMate collection."""

from __future__ import annotations

from collections.abc import Sequence


def classify_batch_results(successes: Sequence[bool], remaining_successes: int) -> list[str]:
    """Classify an ordered batch without exceeding the selected-success quota.

    Successful environments are selected in stable environment-index order until
    ``remaining_successes`` is exhausted. Further successes are retained only as
    excluded attempt-manifest evidence.
    """

    if remaining_successes < 0:
        raise ValueError("remaining_successes must be non-negative.")

    classifications: list[str] = []
    selected = 0
    for success in successes:
        if not isinstance(success, bool):
            raise TypeError(f"success flags must be bool, received {type(success).__name__}.")
        if not success:
            classifications.append("failure")
        elif selected < remaining_successes:
            classifications.append("selected")
            selected += 1
        else:
            classifications.append("excluded")
    return classifications
