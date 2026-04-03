"""Shared prompt formatting helpers for model-facing text."""

from __future__ import annotations

ASSISTANT_TURN_PREFIX = "\n\nASSISTANT:\n"


def ensure_assistant_turn(prompt: str) -> str:
    """Normalize prompts so generation always starts at an explicit assistant turn."""

    stripped = prompt.rstrip()
    if stripped.endswith("ASSISTANT:"):
        return stripped + "\n"
    return stripped + ASSISTANT_TURN_PREFIX
