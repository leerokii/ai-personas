"""Persona construction methods A–D.

Each method returns a list of persona dicts. A persona carries whatever context
that method conditions on; survey.py turns a persona + question into a prompt.

Method A (naive baseline) is implemented here. Methods B, C, D are added in
later phases. Every persona dict carries at least:
    persona_id : int
    method     : "A" | "B" | "C" | "D"
    system     : str | None   # system prompt, if the method uses one
"""

from __future__ import annotations


def method_a(n: int) -> list[dict]:
    """Method A — naive baseline.

    All n personas are identical: there is no per-persona conditioning. Each is
    just an independent sample of the same prompt ("a random US adult woman"),
    drawn at temperature 1.0. The variation between the n responses comes
    entirely from sampling, which is exactly what this baseline is meant to test.
    """
    return [{"persona_id": i, "method": "A", "system": None} for i in range(n)]
