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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"

# Fixed seed so the 100 sampled profiles are identical across the 3 repeat runs
# and across re-runs (reproducibility — "seed everything seedable").
DEFAULT_SEED = 42


def method_a(n: int, client=None) -> list[dict]:
    """Method A — naive baseline.

    All n personas are identical: there is no per-persona conditioning. Each is
    just an independent sample of the same prompt ("a random US adult woman"),
    drawn at temperature 1.0. The variation between the n responses comes
    entirely from sampling, which is exactly what this baseline is meant to test.

    `client` is accepted for a uniform builder signature but unused (no API calls).
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


def sample_profiles(n: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """Sample n demographic profiles from the published marginals.

    Each variable is sampled independently (we have published marginals, not
    crosstabs — documented limitation). Seeded, so persona_id i is the same
    profile every time. Methods B, C, and D all build on these same profiles, so
    persona_id aligns across methods and they stay comparable.
    """
    rng = random.Random(seed)
    marginals = load_marginals()
    profiles = []
    for _ in range(n):
        profiles.append(
            {
                var: rng.choices(cats, weights=weights, k=1)[0]
                for var, (cats, weights) in marginals.items()
            }
        )
    return profiles


def method_b(n: int, client=None, seed: int = DEFAULT_SEED) -> list[dict]:
    """Method B — demographic conditioning.

    Turn each sampled demographic profile into a system prompt. `client` is
    accepted for a uniform builder signature but unused (no API calls).
    """
    template = (PROMPTS / "method_b.txt").read_text()
    return [
        {
            "persona_id": i,
            "method": "B",
            "system": template.format(**profile),
            "profile": profile,
        }
        for i, profile in enumerate(sample_profiles(n, seed))
    ]


def method_c(
    n: int,
    client,
    seed: int = DEFAULT_SEED,
    max_tokens: int = 300,
    max_workers: int = 8,
) -> list[dict]:
    """Method C — narrative persona expansion.

    For each Method B demographic profile, have the model generate a ~150-word
    first-person backstory, then use that narrative as the survey system prompt.
    Generation is concurrent; results stay ordered so persona_id i still maps to
    profile i (and to Method B's persona i). The narrative IS the conditioning.
    """
    profiles = sample_profiles(n, seed)
    gen_template = (PROMPTS / "method_c.txt").read_text()
    persona_template = (PROMPTS / "method_c_persona.txt").read_text()

    def generate(profile: dict) -> str:
        return client.complete(
            gen_template.format(**profile), max_tokens=max_tokens
        ).strip()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        narratives = list(ex.map(generate, profiles))

    return [
        {
            "persona_id": i,
            "method": "C",
            "system": persona_template.format(narrative=narr),
            "profile": profile,
            "narrative": narr,
        }
        for i, (profile, narr) in enumerate(zip(profiles, narratives))
    ]
