"""Evaluation: response distributions, entropy, and Figure 1.

Phase 3 scope: response-diversity (entropy per question) for whatever methods are
present in the loaded results, plus a draft of Figure 1 (response distribution by
method). Written to generalize to all four methods — pass a JSONL per method and
the functions group by the `method` column.

The remaining evaluation dimensions (demographic coherence, internal consistency,
behavioral consistency, face validity) are added in Phase 5.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
FIGURES = ROOT / "results" / "figures"


def load_results(paths) -> pd.DataFrame:
    """Load one or more results JSONL files into a single DataFrame."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [pd.read_json(p, lines=True) for p in paths]
    return pd.concat(frames, ignore_index=True)


def load_question_options() -> dict[str, list[str]]:
    """Map question_id -> ordered list of option letters (e.g. q1 -> [A..E])."""
    with open(PROMPTS / "survey_questions.yaml") as f:
        qs = yaml.safe_load(f)["questions"]
    return {q["id"]: [o["letter"] for o in q["options"]] for q in qs}


def answer_shares(
    df: pd.DataFrame, question_id: str, letters: list[str], method: str | None = None
) -> pd.Series:
    """Normalized share of each option letter for a question (pooled across reps).

    Only parsed answers are counted. Returns a Series indexed by `letters` (so
    options with zero responses appear as 0.0), summing to 1.0.
    """
    sub = df[(df["question_id"] == question_id) & (df["parse_ok"])]
    if method is not None:
        sub = sub[sub["method"] == method]
    counts = sub["parsed_answer"].value_counts()
    shares = pd.Series({L: counts.get(L, 0) for L in letters}, dtype=float)
    total = shares.sum()
    return shares / total if total else shares


def entropy_bits(shares: pd.Series) -> float:
    """Shannon entropy in bits of a distribution (zeros ignored)."""
    h = -sum(p * math.log2(p) for p in shares if p > 0)
    return float(h + 0.0)  # normalize -0.0 -> 0.0 for clean display


def normalized_entropy(shares: pd.Series) -> float:
    """Entropy normalized to [0, 1] by the max possible (log2 of #options)."""
    k = len(shares)
    if k <= 1:
        return 0.0
    return entropy_bits(shares) / math.log2(k)


def entropy_table(df: pd.DataFrame, options: dict[str, list[str]]) -> pd.DataFrame:
    """Per method x question: entropy (bits), normalized entropy, modal option/share."""
    rows = []
    for method in sorted(df["method"].unique()):
        for qid, letters in options.items():
            shares = answer_shares(df, qid, letters, method=method)
            modal = shares.idxmax()
            rows.append(
                {
                    "method": method,
                    "question": qid,
                    "entropy_bits": round(entropy_bits(shares), 3),
                    "norm_entropy": round(normalized_entropy(shares), 3),
                    "modal_option": modal,
                    "modal_share": round(float(shares.max()), 3),
                    "n_options": len(letters),
                }
            )
    return pd.DataFrame(rows)


def plot_figure1(
    df: pd.DataFrame, options: dict[str, list[str]], out_path: Path | None = None
) -> Path:
    """Figure 1 (draft): response distribution by method — stacked option shares
    per question, one bar group per method. With a single method present this is
    one bar per question; it extends to four bars per question once Methods B–D
    are added.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = sorted(df["method"].unique())
    qids = list(options.keys())
    # union of option letters across questions, for a stable colour mapping
    all_letters = sorted({L for letters in options.values() for L in letters})
    cmap = plt.get_cmap("viridis", len(all_letters))
    colours = {L: cmap(i) for i, L in enumerate(all_letters)}

    n_methods = len(methods)
    group_w = 0.8
    bar_w = group_w / n_methods
    x = np.arange(len(qids))

    fig, ax = plt.subplots(figsize=(11, 6))
    for mi, method in enumerate(methods):
        offset = (mi - (n_methods - 1) / 2) * bar_w
        bottoms = np.zeros(len(qids))
        for L in all_letters:
            heights = []
            for qid in qids:
                letters = options[qid]
                if L in letters:
                    shares = answer_shares(df, qid, letters, method=method)
                    heights.append(float(shares.get(L, 0.0)))
                else:
                    heights.append(0.0)
            heights = np.array(heights)
            ax.bar(
                x + offset,
                heights,
                bar_w,
                bottom=bottoms,
                color=colours[L],
                edgecolor="white",
                linewidth=0.4,
                label=L if mi == 0 else None,
            )
            bottoms += heights
        # label each method group under its bars
        for xi in x:
            ax.text(
                xi + offset,
                -0.04,
                method.replace("method_", "").upper(),
                ha="center",
                va="top",
                fontsize=7,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([q.upper() for q in qids])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Response share")
    ax.set_title(
        "Figure 1 (draft) — Response distribution by method\n"
        f"Methods present: {', '.join(m.replace('method_','').upper() for m in methods)}"
    )
    ax.legend(title="Option", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out_path = out_path or FIGURES / "figure1_response_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] or [ROOT / "results" / "raw" / "method_a_full.jsonl"]
    df = load_results(paths)
    options = load_question_options()
    table = entropy_table(df, options)
    print(table.to_string(index=False))
    fig = plot_figure1(df, options)
    print(f"\nFigure 1 saved to {fig}")
