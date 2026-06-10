"""Survey administration: render prompts, call the model, parse single-select answers.

Responsibilities:
  - load the five survey questions from prompts/survey_questions.yaml
  - render a method prompt template + question + options into a full prompt
  - administer one question to one persona, parsing a single option letter,
    retrying once on a parse failure
  - append raw + parsed results to a JSONL log

Parsing is deliberately strict-then-lenient: an exact single letter first, then a
leading "C." / "C)" form, then an isolated letter token. If none match, the call
is retried once; a second failure is logged with parsed_answer=None.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"


def load_questions(path: Path = PROMPTS / "survey_questions.yaml") -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["questions"]


def load_template(name: str) -> str:
    """Load a method prompt template from prompts/<name>."""
    return (PROMPTS / name).read_text()


def format_options(options: list[dict]) -> str:
    return "\n".join(f"{o['letter']}. {o['text']}" for o in options)


def valid_letters(options: list[dict]) -> list[str]:
    return [o["letter"] for o in options]


def render_prompt(template: str, question: dict) -> str:
    return template.format(
        question=question["text"],
        options=format_options(question["options"]),
    )


def parse_letter(text: str, letters: list[str]) -> str | None:
    """Extract a single chosen option letter from a model response, or None."""
    if not text:
        return None
    s = text.strip().upper()
    letterset = set(letters)
    cls = "".join(letters)

    # 1. The whole response is exactly one valid letter.
    if s in letterset:
        return s
    # 2. Leading letter followed by a delimiter: "C.", "C)", "C:", "C -".
    m = re.match(rf"\s*([{cls}])[\.\):\-\s]", s)
    if m:
        return m.group(1)
    # 3. An isolated letter token anywhere (not embedded inside a word).
    m = re.search(rf"(?<![A-Z])([{cls}])(?![A-Z])", s)
    if m:
        return m.group(1)
    return None


def administer_question(
    client,
    prompt: str,
    letters: list[str],
    max_tokens: int = 16,
    max_attempts: int = 2,
) -> dict:
    """Send `prompt`, parse an option letter, retry once on parse failure.

    Returns a dict with the final parsed answer, whether parsing succeeded, the
    number of attempts, and the raw text of each attempt.
    """
    attempts: list[dict] = []
    parsed: str | None = None
    for attempt in range(1, max_attempts + 1):
        raw = client.complete(prompt, max_tokens=max_tokens)
        parsed = parse_letter(raw, letters)
        attempts.append({"attempt": attempt, "raw": raw, "parsed": parsed})
        if parsed is not None:
            break
    return {
        "parsed_answer": parsed,
        "parse_ok": parsed is not None,
        "n_attempts": len(attempts),
        "attempts": attempts,
    }


def log_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to a JSONL file, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
