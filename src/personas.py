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
import yaml

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


# Human-readable phrasing for each sampled behavioral trait, injected into the
# Method D narrative-generation prompt so the backstory anchors to the fact.
BEHAVIOR_PHRASES = {
    "ai_awareness": {
        "heard_a_lot": "She has heard or read a lot about AI.",
        "heard_a_little": "She has heard or read a little about AI.",
        "nothing_at_all": "She has heard nothing at all about AI.",
    },
    "ai_usage_frequency": {
        "almost_constantly_or_several_a_day": "She interacts with AI almost constantly or several times a day.",
        "about_once_a_day": "She interacts with AI about once a day.",
        "several_times_a_week": "She interacts with AI several times a week.",
        "less_often": "She interacts with AI less than several times a week — rarely.",
    },
    "ai_sentiment": {
        "more_concerned": "She feels more concerned than excited about the increased use of AI in daily life.",
        "more_excited": "She feels more excited than concerned about the increased use of AI in daily life.",
        "equally": "She feels equally concerned and excited about the increased use of AI in daily life.",
    },
    "willingness_ai_assist": {
        "a_lot": "She would let AI assist a lot with her day-to-day tasks.",
        "a_little": "She would let AI assist a little with her day-to-day tasks.",
        "not_at_all": "She would not let AI assist with her day-to-day tasks at all.",
    },
    "control_attitudes": {
        "want_more_control": "She would like more control over how AI is used in her life.",
        "comfortable": "She is comfortable with the amount of control she has over how AI is used in her life.",
        "not_sure": "She is not sure how she feels about the control she has over how AI is used in her life.",
    },
}


def load_behaviors(path: Path = DATA / "behaviors_women.yaml") -> dict[str, tuple]:
    """Read the Pew behavioral distributions into {measure: (categories, weights)}.

    Uses the directly measured `women` block for each measure.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    out: dict[str, tuple] = {}
    for measure, block in data.items():
        w = block["women"]
        out[measure] = (list(w.keys()), [float(v) for v in w.values()])
    return out


def sample_behaviors(n: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """Sample n personas' behavioral traits from the Pew women distributions.

    Each measure is sampled independently (published marginals, not behaviour x
    demographic crosstabs — documented limitation). Uses a distinct RNG stream
    from the demographic sampling so the two are reproducible and aligned by
    index without interfering.
    """
    rng = random.Random(seed + 1)
    behaviors = load_behaviors()
    return [
        {m: rng.choices(cats, weights=w, k=1)[0] for m, (cats, w) in behaviors.items()}
        for _ in range(n)
    ]


def method_d(
    n: int,
    client,
    seed: int = DEFAULT_SEED,
    max_tokens: int = 300,
    max_workers: int = 8,
) -> list[dict]:
    """Method D — behaviorally grounded personas.

    Take the same seeded demographic profiles as Methods B and C, layer on
    behavioral traits sampled from Pew's published women distributions, and
    generate a narrative anchored to those empirical behavioral facts. The
    narrative is the survey system prompt; the sampled traits are persisted so
    the behavioral-consistency check can compare stated behavior to answers.
    """
    profiles = sample_profiles(n, seed)
    behaviors = sample_behaviors(n, seed)
    gen_template = (PROMPTS / "method_d.txt").read_text()
    persona_template = (PROMPTS / "method_c_persona.txt").read_text()

    def render_gen(profile: dict, traits: dict) -> str:
        fields = dict(profile)
        fields["b_awareness"] = BEHAVIOR_PHRASES["ai_awareness"][traits["ai_awareness"]]
        fields["b_usage"] = BEHAVIOR_PHRASES["ai_usage_frequency"][traits["ai_usage_frequency"]]
        fields["b_sentiment"] = BEHAVIOR_PHRASES["ai_sentiment"][traits["ai_sentiment"]]
        fields["b_willingness"] = BEHAVIOR_PHRASES["willingness_ai_assist"][traits["willingness_ai_assist"]]
        fields["b_control"] = BEHAVIOR_PHRASES["control_attitudes"][traits["control_attitudes"]]
        return gen_template.format(**fields)

    def generate(pt: tuple) -> str:
        profile, traits = pt
        return client.complete(render_gen(profile, traits), max_tokens=max_tokens).strip()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        narratives = list(ex.map(generate, zip(profiles, behaviors)))

    return [
        {
            "persona_id": i,
            "method": "D",
            "system": persona_template.format(narrative=narr),
            "profile": profile,
            "traits": traits,
            "narrative": narr,
        }
        for i, (profile, traits, narr) in enumerate(
            zip(profiles, behaviors, narratives)
        )
    ]
