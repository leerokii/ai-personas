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

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"

# Fixed seed so the 100 sampled profiles are identical across the 3 repeat runs
# and across re-runs (reproducibility — "seed everything seedable").
DEFAULT_SEED = 42


def method_a(n: int) -> list[dict]:
    """Method A — naive baseline.

    All n personas are identical: there is no per-persona conditioning. Each is
    just an independent sample of the same prompt ("a random US adult woman"),
    drawn at temperature 1.0. The variation between the n responses comes
    entirely from sampling, which is exactly what this baseline is meant to test.
    """
    return [{"persona_id": i, "method": "A", "system": None} for i in range(n)]


def load_marginals(path: Path = DATA / "demographics_women.csv") -> dict[str, tuple]:
    """Read the demographic marginals CSV into {variable: (categories, weights)}.

    The CSV's '#'-prefixed header comments are skipped. Each variable's rows give
    independent category percentages; Method B samples each variable on its own,
    which approximates the joint distribution under an independence assumption
    (documented limitation — we only have published marginals, not crosstabs).
    """
    df = pd.read_csv(path, comment="#")
    out: dict[str, tuple] = {}
    for var, grp in df.groupby("variable", sort=False):
        out[var] = (grp["category"].tolist(), grp["pct"].astype(float).tolist())
    return out


def _render_system_b(template: str, profile: dict) -> str:
    return template.format(**profile)


def method_b(n: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """Method B — demographic conditioning.

    Sample n demographic profiles from the published marginals (independently per
    variable), and turn each into a system prompt. Sampling is seeded, so the
    same n personas recur across repeat runs.
    """
    rng = random.Random(seed)
    marginals = load_marginals()
    template = (PROMPTS / "method_b.txt").read_text()

    personas = []
    for i in range(n):
        profile = {
            var: rng.choices(cats, weights=weights, k=1)[0]
            for var, (cats, weights) in marginals.items()
        }
        personas.append(
            {
                "persona_id": i,
                "method": "B",
                "system": _render_system_b(template, profile),
                "profile": profile,
            }
        )
    return personas
