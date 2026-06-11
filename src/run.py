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

# Per-call token assumptions for the pre-run estimate (from the smoke test:
# ~109 input + 4 output tokens for a single-question Method A call; rounded up a
# little to cover question-length variation across the five questions).
EST_INPUT_TOKENS = 120
EST_OUTPUT_TOKENS = 5

# Above this estimated cost, require explicit confirmation before running.
CONFIRM_THRESHOLD_USD = 1.00


def estimate_cost_usd(n_calls: int, pricing: dict) -> float:
    return (
        n_calls * EST_INPUT_TOKENS / 1_000_000 * pricing["input_per_mtok"]
        + n_calls * EST_OUTPUT_TOKENS / 1_000_000 * pricing["output_per_mtok"]
    )


# Persona builders by method. Methods B–D are added in later phases.
METHOD_BUILDERS = {
    "method_a": (personas.method_a, "method_a.txt"),
}


def build_tasks(method: str, n_personas: int, questions: list[dict], n_repeats: int):
    """Return a flat list of (rep, persona, question) tuples to administer."""
    builder, _ = METHOD_BUILDERS[method]
    panel = builder(n_personas)
    tasks = []
    for rep in range(n_repeats):
        for p in panel:
            for q in questions:
                tasks.append((rep, p, q))
    return tasks


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
    _, template_name = METHOD_BUILDERS[method]
    template = survey.load_template(template_name)
    questions = survey.load_questions()[:n_questions]
    tasks = build_tasks(method, n_personas, questions, n_repeats)
    n_calls = len(tasks)

    pricing = {
        "input_per_mtok": client.tracker.input_per_mtok,
        "output_per_mtok": client.tracker.output_per_mtok,
    }
    est = estimate_cost_usd(n_calls, pricing)
    print(
        f"[estimate] {method}: {n_calls} calls "
        f"({n_personas} personas x {n_questions} questions x {n_repeats} repeats) "
        f"~= ${est:.2f}  (cap ${client.tracker.budget_cap_usd:.2f})"
    )
    if confirm and est > CONFIRM_THRESHOLD_USD:
        print(
            f"[guard] estimate ${est:.2f} exceeds ${CONFIRM_THRESHOLD_USD:.2f} "
            "confirm-threshold; re-run with --yes to proceed."
        )
        return

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
        res = survey.administer_question(client, prompt, letters)
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
        n_calls = args.personas * args.questions * args.repeats
        est = estimate_cost_usd(n_calls, pricing)
        print(f"[estimate] {args.method}: {n_calls} calls ~= ${est:.2f}")
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
