# LLM Persona Construction Methods

Testing four ways of building LLM personas of US women aged 18–35, surveyed on
questions about using AI to find online communities. Each method produces a
100-persona panel; the methods are compared on five evaluation dimensions to
recommend which a senior decision-maker should trust for a consequential
product decision.

See `PROJECT_BRIEF.md` for the full specification.

## Status

Scaffolding in place (Phase 1). Methods, survey administration, evaluation, and
findings are built in subsequent phases — this section is rewritten last.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your Anthropic API key into .env
```

Configuration (model, temperature, panel size, repeats, budget cap) lives in
`config.yaml`. The default model is `claude-sonnet-4-6`.

## Layout

```
src/        client.py (API wrapper + cost tracking); methods, survey, eval added later
prompts/    versioned prompt text files (one per method)
data/        reference demographic/behavioral distributions (Methods B and D)
results/raw/      raw model responses as JSONL
results/figures/  generated figures
config.yaml  all tunable parameters
```
