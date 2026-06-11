"""Evaluation: response distributions, entropy, and Figure 1.

Phase 3 scope: response-diversity (entropy per question) for whatever methods are
present in the loaded results, plus a draft of Figure 1 (response distribution by
method). Written to generalize to all four methods — pass a JSONL per method and
the functions group by the `method` column.

The remaining evaluation dimensions (demographic coherence, internal consistency,
behavioral consistency, face validity) are added in Phase 5.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
RESULTS_RAW = ROOT / "results" / "raw"
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


# ===========================================================================
# Dimension 1 — Response diversity (entropy). entropy_table() above is the core;
# this pivots it into a method x question comparison grid.
# ===========================================================================
def entropy_comparison(df: pd.DataFrame, options: dict) -> pd.DataFrame:
    tbl = entropy_table(df, options)
    return tbl.pivot(index="method", columns="question", values="entropy_bits")


# ===========================================================================
# Dimension 2 — Demographic coherence. Split a method's answers by a demographic
# variable (from the persona sidecar) and show answer shares per subgroup.
# ===========================================================================
def load_personas(method: str) -> pd.DataFrame:
    """Load a method's persona sidecar, flattening profile fields into columns."""
    path = RESULTS_RAW / f"method_{method.lower()}_personas.jsonl"
    pers = pd.read_json(path, lines=True)
    if "profile" in pers.columns:
        prof = pers["profile"].apply(pd.Series)
        pers = pd.concat([pers.drop(columns=["profile"]), prof], axis=1)
    return pers


def subgroup_shares(
    method: str, qid: str, letters: list[str], by: str, order: list[str]
) -> pd.DataFrame:
    """Row-normalized answer shares (%) for one question, split by `by` subgroup."""
    resp = load_results(RESULTS_RAW / f"method_{method.lower()}_full.jsonl")
    pers = load_personas(method)
    df = resp.merge(pers[["persona_id", by]], on="persona_id")
    q = df[(df.question_id == qid) & (df.parse_ok)]
    ct = pd.crosstab(q[by], q.parsed_answer)
    for L in letters:  # ensure all option columns present
        if L not in ct.columns:
            ct[L] = 0
    ct = ct[letters].reindex(order).fillna(0).astype(int)
    return (ct.div(ct.sum(axis=1), axis=0) * 100).round(0).astype(int)


# ===========================================================================
# Dimension 3 — Internal consistency across the 3 repeat runs.
# For each method x question: the share of the overall-modal answer in each rep,
# then mean +/- range across reps (stability of the distribution). Also a
# test-retest agreement for fixed-persona methods (B/C/D).
# ===========================================================================
def internal_consistency(df: pd.DataFrame, options: dict) -> pd.DataFrame:
    reps = sorted(df["rep"].unique())
    rows = []
    for method in sorted(df["method"].unique()):
        for qid, letters in options.items():
            modal = answer_shares(df, qid, letters, method).idxmax()
            per_rep = []
            for rep in reps:
                sub = df[
                    (df.method == method)
                    & (df.question_id == qid)
                    & (df.rep == rep)
                    & (df.parse_ok)
                ]
                per_rep.append((sub.parsed_answer == modal).mean())
            per_rep = np.array(per_rep)
            rows.append(
                {
                    "method": method,
                    "question": qid,
                    "modal_option": modal,
                    "modal_share_mean": round(float(per_rep.mean()), 3),
                    "modal_share_range": round(float(per_rep.max() - per_rep.min()), 3),
                    "per_rep": [round(float(x), 3) for x in per_rep],
                }
            )
    return pd.DataFrame(rows)


def test_retest_agreement(method: str, options: dict) -> float:
    """For a fixed-persona method (B/C/D): mean fraction of personas whose answer
    is identical across all 3 repeat runs, averaged over questions. Not meaningful
    for Method A (each rep is an independent sample, not the same persona)."""
    resp = load_results(RESULTS_RAW / f"method_{method.lower()}_full.jsonl")
    agrees = []
    for qid in options:
        q = resp[(resp.question_id == qid) & (resp.parse_ok)]
        piv = q.pivot_table(
            index="persona_id", columns="rep", values="parsed_answer", aggfunc="first"
        )
        all_same = piv.apply(lambda r: r.nunique() == 1, axis=1)
        agrees.append(all_same.mean())
    return float(np.mean(agrees))


# ===========================================================================
# Dimension 4 — Behavioral consistency (Method D). Split answers by a stated
# behavioral trait (from the sidecar `traits`) and show answer shares.
# ===========================================================================
def trait_shares(
    method: str, qid: str, trait: str, letters: list[str], trait_order: list[str]
) -> pd.DataFrame:
    resp = load_results(RESULTS_RAW / f"method_{method.lower()}_full.jsonl")
    pers = pd.read_json(RESULTS_RAW / f"method_{method.lower()}_personas.jsonl", lines=True)
    pers[trait] = pers["traits"].apply(lambda t: t[trait])
    df = resp.merge(pers[["persona_id", trait]], on="persona_id")
    q = df[(df.question_id == qid) & (df.parse_ok)]
    ct = pd.crosstab(q[trait], q.parsed_answer)
    for L in letters:
        if L not in ct.columns:
            ct[L] = 0
    ct = ct[letters].reindex(trait_order).fillna(0).astype(int)
    return (ct.div(ct.sum(axis=1), axis=0) * 100).round(0).astype(int)


# ===========================================================================
# Dimension 5 — Face validity. LLM-as-judge rates sampled narratives on four
# 1-5 scales with a one-line justification. Needs an API client.
# ===========================================================================
def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def face_validity(
    client, methods=("C", "D"), k: int = 5, seed: int = 99, max_tokens: int = 220
) -> pd.DataFrame:
    judge = (PROMPTS / "face_validity_judge.txt").read_text()
    rng = random.Random(seed)
    rows = []
    for m in methods:
        pers = pd.read_json(RESULTS_RAW / f"method_{m.lower()}_personas.jsonl", lines=True)
        idx = sorted(rng.sample(range(len(pers)), k))
        for i in idx:
            row = pers.iloc[i]
            resp = client.complete(
                judge.format(narrative=row["narrative"]),
                max_tokens=max_tokens,
                temperature=0.0,
            )
            data = _extract_json(resp)
            rows.append(
                {
                    "method": m,
                    "persona_id": int(row["persona_id"]),
                    "realistic": data.get("realistic"),
                    "generic": data.get("generic"),
                    "stereotyped": data.get("stereotyped"),
                    "caricature": data.get("caricature"),
                    "justification": data.get("justification", ""),
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# Figure 2 — entropy comparison across methods (grouped bars per question).
# ===========================================================================
def plot_figure2(df: pd.DataFrame, options: dict, out_path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tbl = entropy_table(df, options)
    methods = sorted(df["method"].unique())
    qids = list(options.keys())
    x = np.arange(len(qids))
    n = len(methods)
    w = 0.8 / n
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(11, 6))
    for mi, method in enumerate(methods):
        vals = [
            tbl[(tbl.method == method) & (tbl.question == q)]["entropy_bits"].iloc[0]
            for q in qids
        ]
        ax.bar(
            x + (mi - (n - 1) / 2) * w,
            vals,
            w,
            label=method,
            color=cmap(mi),
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([q.upper() for q in qids])
    ax.set_ylabel("Response entropy (bits)")
    ax.set_title("Figure 2 — Response entropy by method and question\n(higher = more diverse; 0 = total mode collapse)")
    ax.legend(title="Method")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_path = out_path or FIGURES / "figure2_entropy_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# Figure 3 — demographic coherence on one question, split by subgroup, all four
# methods (small multiples). Method A has no demographic conditioning, so it gets
# a single "all personas" bar as the no-conditioning baseline.
# ===========================================================================
def plot_figure3(
    qid: str,
    options: dict,
    by: str = "age_band",
    order: list[str] | None = None,
    out_path: Path | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    letters = options[qid]
    methods = ["A", "B", "C", "D"]
    cmap = plt.get_cmap("viridis", len(letters))
    colours = {L: cmap(i) for i, L in enumerate(letters)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, method in zip(axes.flat, methods):
        if method == "A":
            # No demographics — show the single overall distribution.
            df = load_results(RESULTS_RAW / "method_a_full.jsonl")
            shares = answer_shares(df, qid, letters, method="A") * 100
            groups = ["all"]
            data = shares.to_frame("all").T
        else:
            data = subgroup_shares(method, qid, letters, by, order)
            groups = list(data.index)
        bottoms = np.zeros(len(groups))
        xs = np.arange(len(groups))
        for L in letters:
            vals = data[L].values if L in data.columns else np.zeros(len(groups))
            ax.bar(xs, vals, 0.7, bottom=bottoms, color=colours[L], edgecolor="white", label=L)
            bottoms += vals
        ax.set_xticks(xs)
        ax.set_xticklabels(groups, rotation=0, fontsize=8)
        ax.set_title(f"Method {method}")
        ax.set_ylim(0, 100)
    axes.flat[0].set_ylabel("Response share (%)")
    axes.flat[2].set_ylabel("Response share (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[L]) for L in letters]
    fig.legend(handles, letters, title="Option", loc="center right")
    fig.suptitle(
        f"Figure 3 — Demographic coherence on {qid.upper()} by {by}, across methods",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.96])
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_path = out_path or FIGURES / "figure3_demographic_coherence.png"
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
