"""Orchestration + cost guard for running a method's full survey panel.

Builds the full task list (personas x questions x repeats), runs the survey
calls concurrently, logs every raw response to JSONL as it completes, and prints
running cost. A pre-run estimate is printed and — above a threshold — must be
confirmed before any calls are made.

Usage:
    python src/run.py method_a --estimate        # print cost estimate only
    python src/run.py method_a                    # run full Method A
    python src/run.py method_a --personas 5 --questions 1 --repeats 1   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from client import Client  # noqa: E402
import personas  # noqa: E402
import survey  # noqa: E402

RESULTS_RAW = ROOT.parent / "results" / "raw"

# Per-call token assumptions for the pre-run estimate, calibrated from smoke
# tests. Input tokens per SURVEY call vary by method because the persona system
# prompt grows: none (A), a short demographic profile (B), a ~150-word narrative
# (C, D). Output is a single letter (~5 tokens).
SURVEY_INPUT_TOKENS = {
    "method_a": 110,
    "method_b": 185,
    "method_c": 290,
    "method_d": 290,
}
SURVEY_OUTPUT_TOKENS = 5

# Methods C and D also make one narrative-GENERATION call per persona.
GEN_METHODS = {"method_c", "method_d"}
GEN_INPUT_TOKENS = 130
GEN_OUTPUT_TOKENS = 300

# Above this estimated cost, require explicit confirmation before running.
CONFIRM_THRESHOLD_USD = 1.00


def estimate_cost_usd(method, n_personas, n_questions, n_repeats, pricing):
    """Estimate total cost (survey + any narrative generation) and the call counts."""
    n_survey = n_personas * n_questions * n_repeats
    s_in = SURVEY_INPUT_TOKENS.get(method, 150)
    cost = (
        n_survey * s_in / 1_000_000 * pricing["input_per_mtok"]
        + n_survey * SURVEY_OUTPUT_TOKENS / 1_000_000 * pricing["output_per_mtok"]
    )
    n_gen = n_personas if method in GEN_METHODS else 0
    cost += (
        n_gen * GEN_INPUT_TOKENS / 1_000_000 * pricing["input_per_mtok"]
        + n_gen * GEN_OUTPUT_TOKENS / 1_000_000 * pricing["output_per_mtok"]
    )
    return cost, n_survey, n_gen


# Persona builders by method, paired with the user-message template each uses.
# Method A embeds its framing in the user prompt (method_a.txt). Methods B-D put
# the persona in a system prompt and share the question-only user template.
METHOD_BUILDERS = {
    "method_a": (personas.method_a, "method_a.txt"),
    "method_b": (personas.method_b, "survey_question.txt"),
    "method_c": (personas.method_c, "survey_question.txt"),
}


def build_tasks(panel: list[dict], questions: list[dict], n_repeats: int):
    """Return a flat list of (rep, persona, question) tuples to administer."""
    tasks = []
    for rep in range(n_repeats):
        for p in panel:
            for q in questions:
                tasks.append((rep, p, q))
    return tasks


def write_personas_sidecar(method: str, panel: list[dict]) -> Path:
    """Persist persona metadata (profile / traits / narrative) alongside results.

    The survey JSONL keys responses by persona_id only; this sidecar maps each
    persona_id to its demographic profile (Method B), behavioural traits (D), and
    narrative (C/D), so analyses can split responses by subgroup. The `system`
    prompt text itself is omitted to keep the file readable.
    """
    path = RESULTS_RAW / f"{method}_personas.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in panel:
            rec = {"persona_id": p["persona_id"], "method": p["method"]}
            for key in ("profile", "traits", "narrative"):
                if key in p:
                    rec[key] = p[key]
            f.write(json.dumps(rec) + "\n")
    return path


def run_method(
    method: str,
    n_personas: int,
    n_questions: int,
    n_repeats: int,
    out_path: Path,
    max_workers: int = 8,
    confirm: bool = True,
) -> None:
    client = Client()
    builder, template_name = METHOD_BUILDERS[method]
    template = survey.load_template(template_name)
    questions = survey.load_questions()[:n_questions]

    pricing = {
        "input_per_mtok": client.tracker.input_per_mtok,
        "output_per_mtok": client.tracker.output_per_mtok,
    }
    est, n_survey, n_gen = estimate_cost_usd(
        method, n_personas, len(questions), n_repeats, pricing
    )
    gen_note = f" + {n_gen} generation" if n_gen else ""
    print(
        f"[estimate] {method}: {n_survey} survey{gen_note} calls "
        f"({n_personas} personas x {len(questions)} questions x {n_repeats} repeats) "
        f"~= ${est:.2f}  (cap ${client.tracker.budget_cap_usd:.2f})"
    )
    if confirm and est > CONFIRM_THRESHOLD_USD:
        print(
            f"[guard] estimate ${est:.2f} exceeds ${CONFIRM_THRESHOLD_USD:.2f} "
            "confirm-threshold; re-run with --yes to proceed."
        )
        return

    # Build the panel AFTER the confirm gate — for C/D this is where narrative
    # generation (API calls) happens, so it must not run before approval.
    if n_gen:
        print(f"[personas] generating {n_personas} narratives ...")
    panel = builder(n_personas, client)
    tasks = build_tasks(panel, questions, n_repeats)
    n_calls = len(tasks)

    sidecar = write_personas_sidecar(method, panel)
    print(f"[personas] wrote {len(panel)} persona records to {sidecar}")

    # Pre-render each question's prompt + valid letters once.
    rendered = {
        q["id"]: (survey.render_prompt(template, q), survey.valid_letters(q["options"]))
        for q in questions
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    done = {"n": 0}

    def worker(task):
        rep, p, q = task
        prompt, letters = rendered[q["id"]]
        res = survey.administer_question(
            client, prompt, letters, system=p.get("system")
        )
        return {
            "method": p["method"],
            "persona_id": p["persona_id"],
            "question_id": q["id"],
            "rep": rep,
            "raw_response": res["attempts"][-1]["raw"],
            "parsed_answer": res["parsed_answer"],
            "parse_ok": res["parse_ok"],
            "n_attempts": res["n_attempts"],
            "model": client.model,
            "temperature": client.temperature,
        }

    n_parse_fail = 0
    with open(out_path, "w") as f:  # truncate / fresh run
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, t) for t in tasks]
            for fut in as_completed(futures):
                rec = fut.result()
                if not rec["parse_ok"]:
                    n_parse_fail += 1
                with write_lock:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done["n"] += 1
                    if done["n"] % 250 == 0 or done["n"] == n_calls:
                        print(
                            f"  {done['n']}/{n_calls} done — "
                            f"{client.tracker.summary()}"
                        )

    print(
        f"[done] {method}: {n_calls} calls, "
        f"{n_calls - n_parse_fail}/{n_calls} parsed "
        f"({100 * (n_calls - n_parse_fail) / n_calls:.1f}%), "
        f"logged to {out_path}"
    )
    client.print_cost(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=list(METHOD_BUILDERS))
    ap.add_argument("--personas", type=int, default=100)
    ap.add_argument("--questions", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--estimate", action="store_true", help="print estimate only")
    ap.add_argument("--yes", action="store_true", help="skip confirm threshold")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else RESULTS_RAW / f"{args.method}_full.jsonl"

    if args.estimate:
        client = Client()
        pricing = {
            "input_per_mtok": client.tracker.input_per_mtok,
            "output_per_mtok": client.tracker.output_per_mtok,
        }
        est, n_survey, n_gen = estimate_cost_usd(
            args.method, args.personas, args.questions, args.repeats, pricing
        )
        gen_note = f" + {n_gen} generation" if n_gen else ""
        print(
            f"[estimate] {args.method}: {n_survey} survey{gen_note} calls ~= ${est:.2f}"
        )
        return

    run_method(
        args.method,
        args.personas,
        args.questions,
        args.repeats,
        out,
        max_workers=args.workers,
        confirm=not args.yes,
    )


if __name__ == "__main__":
    main()
