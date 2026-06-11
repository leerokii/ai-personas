# LLM Persona Construction Methods

Testing four ways of building LLM personas of **US women (all ages, 18+)**, surveyed
on questions about using AI to find online communities. Each method produces a
100-persona panel; the four panels answer the same five single-select questions
(100 personas × 5 questions × 3 repeat runs each), and the methods are compared on
five evaluation dimensions to recommend which a senior decision-maker should trust
for a consequential product decision.

See `PROJECT_BRIEF.md` for the full specification.

---

## Problem framing

A developer is weighing whether to build an AI-powered tool that helps US women
find online communities, and — if so — what design choices to make (how proactive
to be, what kinds of communities to surface, how to describe them, what would make
users walk away). The honest way to answer is to ask a representative panel of real
women. Absent that, the question becomes whether a panel of **LLM personas** can
stand in for them, and — critically — **how you build those personas changes the
answer you get back.** A naive prompt, a demographically-weighted panel, a panel of
richly-imagined individuals, and a panel grounded in real behavioral data will not
agree, even on the same five questions. This study makes that disagreement visible
and asks which construction method a decision-maker should actually trust.

The framing borrows from research on how young women collectively imagine and shape
their relationships to algorithmic platforms (Bucher's *algorithmic imaginaries*;
Hogan's exhibition/artefact distinction). This task asks the inverse question:
whether language models can credibly *simulate* those imaginaries back to us, and
where they break. The four methods are deliberately a ladder — each rung adds one
variable (demographics, then narrative, then empirical behavioral grounding) — so
that the comparison isolates what each layer buys and what it costs.

---

## The four methods

| Method | Name | What conditions each persona | Persona-construction cost |
|---|---|---|---|
| **A** | Naive baseline | Nothing — one prompt, *"a random US adult woman"*, sampled 100× at temperature 1.0 | none (no API calls) |
| **B** | Demographic conditioning | A demographic profile (age, race/ethnicity, education, income, region) sampled from US Census ACS marginals, delivered as a system prompt | none (local sampling) |
| **C** | Narrative persona expansion | A ~150-word first-person backstory the model generates *from* the Method B profile | 100 generation calls |
| **D** | Behaviorally grounded | The Method B profile **plus** five behavioral traits (AI awareness, usage, sentiment, willingness, control) sampled from Pew's published distributions for women, with the narrative anchored to those empirical facts | 100 generation calls |

Persona IDs are aligned across B, C and D (same seed, same sampled demographics), so
the *same* demographic profile can be traced through all three richer methods.

Data sources (in `data/`): demographic marginals from **US Census ACS 2023**
(`demographics_women.csv`); behavioral distributions from **Pew Research Center ATP
Wave 173**, June 2025 (`behaviors_women.yaml`, every figure page-cited to the report
or topline). The three Pew source PDFs are committed for provenance.

---

## Findings — five-dimension synthesis

All numbers are from the runs in `results/raw/`; the harness is `src/evaluate.py`;
the figures are in `results/figures/`.

### 1. Response diversity (entropy) — *Figure 2*

Entropy (bits) per method × question. Higher = more diverse; 0 = total mode collapse.

| | q1 usage | q2 comfort | q3 type | q4 friction | q5 tone |
|---|---|---|---|---|---|
| A | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 |
| **B** | 0.00 | **1.56** | **1.69** | **0.94** | 0.00 |
| C | 0.08 | 1.00 | 0.97 | 0.11 | 0.00 |
| **D** | **0.16** | 1.13 | 1.27 | 0.57 | **0.56** |

Method A collapses to a single answer on 4 of 5 questions (only 3 of its 1,500
answers deviate from the mode). Method B unlocks the most diversity on the
attitude/preference questions. **Method D is the only method that produces any
diversity on q1 and q5**, the two questions every other method fully collapses.
Method C is *narrower* than B — narrative richness reduced diversity (see below).

### 2. Demographic coherence — *Figure 3*

Do subgroups answer in plausible directions? Tested on **two** questions (q3 community
type and q2 comfort), split by age band. The pattern **replicates**: only Method B
holds a clean, monotonic age gradient; the narrative methods flatten it.

On q2 comfort, Method B runs from 96% comfortable (18–29) to 80% *un*comfortable
(65+) — a textbook gradient. Method D sits at ~63–79% "somewhat uncomfortable" at
*every* age — the gradient is gone. The narratives inject vivid "local community /
neighborhood" texture (book clubs, church, neighbors) that pulls answers toward
"Local" regardless of demographics, overwriting the structure Method B captured.

### 3. Internal consistency (3 repeat runs)

Distribution stability is high for all methods (modal-share range ≤ 0.07 across the
3 reps). Test-retest agreement — the fraction of personas giving an *identical*
answer across all 3 reps — confirms the hypothesis that richer context makes
personas more deterministic:

| Method C | Method D | Method B |
|---|---|---|
| **0.97** | 0.96 | 0.89 |

(Method A: not applicable — its "personas" are independent samples, not fixed
individuals.) Importantly, the 3 repeat runs are themselves a reproducibility
replicate: they show that re-administering the survey yields near-identical
*aggregate* distributions even though individual responses are stochastic.

### 4. Behavioral consistency (Method D structural validity)

Do Method D personas answer in line with their *stated* behaviors? The answer
depends on alignment between the trait and the question:

- **Stated AI usage vs q1** (does it *currently* use AI to find communities): weak.
  94–100% answer "Never" regardless of stated usage — but heavy users are the only
  group to break from "Never" at all (6% vs 0%). q1 asks about a niche behavior, so
  being a heavy general-AI user doesn't imply it.
- **Stated AI sentiment vs q2** (comfort with AI suggesting communities): **strong
  and monotonic.** % answering *comfortable*: more-concerned **0%**, equally 14%,
  more-excited **100%**.

So the behavioral layer *does* propagate to answers — strongly when the question
matches the trait, weakly when it doesn't. That contrast is itself the structural
validity result: Method D's grounding is real but only as good as the trait–question
alignment.

### 5. Face validity (LLM judge — read with the caveat below)

Five narratives each from Methods C and D, rated 1–5 by an LLM judge (the ten
narratives and per-rating justifications are in
`results/face_validity_narratives.md`):

| | realistic | generic | stereotyped | caricature |
|---|---|---|---|---|
| C | 4.4 | 2.6 | 2.4 | 1.0 |
| D | 4.4 | 2.4 | 2.4 | 1.2 |

Both methods produce **believable, individualized** narratives (realistic ≈ 4.4/5,
caricature ≈ 1.1/5) with **mild but real stereotyping** — the judge repeatedly
flagged "familiar demographic tropes" (Latina-mom, rural-Midwestern-skeptic,
Southern-retiree). That same trope-leaning is what flattens the demographic gradient
in dimension 2.

> ⚠️ **Self-evaluation bias (read this).** The face-validity ratings come from an LLM
> judge in the *same model family* (`claude-sonnet-4-6`) that wrote the narratives. A
> model rating its own family's output is **not an independent assessment** and very
> likely skews lenient — a genuinely human reader might rate "generic" or
> "stereotyped" higher. These numbers are **indicative, not definitive.** In a fuller
> study, **human ratings or a different model-family judge** is the right next step.
> Do not treat dimension 5 as load-bearing evidence on its own.

---

## Decision matrix — which method to use

**Headline: no method dominates.** There is a real tension between *demographic
faithfulness* (B wins) and *individual richness + behavioral grounding* (D wins). The
right choice depends on what the decision-maker needs.

| If you need… | Use | Why |
|---|---|---|
| Subgroup estimates that move correctly with demographics (e.g. "how does comfort differ by age?") | **B** | Only B preserves clean, monotonic demographic gradients across questions |
| A behaviorally plausible panel whose answers track stated attitudes; the broadest question coverage | **D** | Best structural validity on aligned questions; only method to avoid collapse on q1/q5 |
| Rich, readable individual personas for qualitative review / stakeholder empathy | **C or D** | Believable narratives (face validity ≈ 4.4/5); D adds empirically-sourced AI attitudes |
| A quick, cheap signal of the single most likely answer | **A** | Trivially cheap, but it returns *only* the mode and erases all minority opinion — use with extreme caution |

**Recommendation for the community-finder decision:** use **Method D as the primary
panel** (it has the strongest structural validity and the widest coverage, and its
"local community" pull is itself a *substantive signal* — these women gravitate to
local/neighborhood communities), but **cross-check every demographic-subgroup claim
against Method B**, because D flattens the very gradients a subgroup analysis depends
on. Treat A as a degenerate baseline only.

**Where the panel should NOT replace human research:** any decision that hinges on
minority opinion, on a specific behavior the survey didn't align a trait to, or on
fine demographic gradients. The methods disagree most exactly where the stakes
(privacy concerns, who feels surveilled, who is uncomfortable) are highest.

---

## Limitations

- **Variance flattening / training-data opinion skew.** Method A demonstrates the
  base failure mode: LLMs collapse toward a modal "consensus" answer and erase
  minority views. Every richer method mitigates but does not eliminate this; the
  panel's "opinions" are a reflection of training-data priors, not measured humans.
- **Stereotype amplification.** The narrative methods (C, D) lean on mild demographic
  tropes, which both reads as light stereotyping (dimension 5) and *overwrites* real
  demographic structure (dimension 2). Richer ≠ more representative.
- **No external ground truth (by design).** With no real survey of these exact
  questions, the evaluation is method-vs-method, not method-vs-truth. We can say which
  method is more internally coherent, not which is *correct*.
- **Face-validity self-evaluation bias.** Dimension 5 is judged by the same model
  family that generated the narratives — not independent; likely lenient. See the
  prominent caveat above.
- **Independence assumption in sampling.** Demographics and behaviors are sampled from
  published *marginals*, not joint crosstabs (Pew/Census don't publish the joint
  distribution for this subgroup), so cross-variable correlations are not modeled.
- **Behavioral grounding is approximate.** The behavioral distributions are Pew's
  directly-measured *women* figures (not a women×age crosstab, which Pew doesn't
  publish); see `data/behaviors_women.yaml` headers.
- **Model dependence.** All results are from `claude-sonnet-4-6`. A second model
  (stretch goal in the brief) would test whether the *patterns* — not just the
  numbers — are model-specific.
- **Reproducibility is distributional, not byte-level.** Seeded inputs (profiles,
  traits) regenerate identically; narratives and survey answers are temperature-1.0
  samples with no API seed, so raw responses vary run-to-run while aggregate findings
  are stable (evidenced by the 3 repeat runs). See *Reproducibility* below.

---

## Future work — networked personas

The natural next step is to drop the independence assumption between personas. Place
the behaviorally-grounded Method D personas on a small-world social graph, let each
persona see its neighbors' answers in a second round, and measure whether social
exposure shifts the panel — convergence, polarization, or the emergence of local
norms. This tests Artificial Societies' interconnected-persona thesis directly, and
community-formation is the precise domain where social context *should* matter most:
people decide which communities to join partly by watching what people like them do.
Out of scope to build in 48 hours; in scope to name as the right direction.

---

## Reproducibility

- **Seeded and deterministic:** demographic profiles (`sample_profiles`) and
  behavioral traits (`sample_behaviors`) regenerate byte-identically across runs and
  match the committed persona sidecars.
- **Stochastic by design:** narrative generation and all survey responses run at
  `temperature: 1.0` with no API-level seed, so the raw `results/raw/*_full.jsonl`
  files are **not** byte-identical across runs. This is intended — Method A's whole
  point is temperature-1.0 sampling — and aggregate findings are stable, as the 3
  repeat runs (dimension 3) demonstrate.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your Anthropic API key into .env
```

Configuration (model, temperature, panel size, repeats, budget cap) lives in
`config.yaml`. Default model `claude-sonnet-4-6`. Re-run a method end-to-end with
`python src/run.py method_d` (prints a cost estimate and a >$1 confirmation guard);
regenerate the evaluation and figures with
`python src/evaluate.py results/raw/method_*_full.jsonl`.

## Layout

```
src/
  client.py     Anthropic API wrapper + running cost tracker + budget guard
  personas.py   methods A–D (sampling, narrative generation)
  survey.py     prompt rendering, single-letter parsing, retry, JSONL logging
  evaluate.py   all five evaluation dimensions + Figures 1–3
  run.py        orchestration, cost estimate/guard, concurrent execution
prompts/        versioned prompt text files (one per method + judge)
data/           Census/Pew reference distributions + source PDFs
results/raw/        raw responses + persona sidecars (JSONL; gitignored)
results/figures/    Figures 1–3
config.yaml     all tunable parameters
notebook.ipynb  narrative walkthrough for reviewers
```

## References

- Santurkar et al. 2023, *Whose Opinions Do Language Models Reflect?*
- Argyle et al. 2023, *Out of One, Many*
- Park et al. 2024, generative agent simulations
- Artificial Societies Survey Eval Report, Jan 2026
- He et al., *British Journal of Psychology* (founders' AI society paper)
- Bucher 2017, algorithmic imaginaries
- Hogan 2010, exhibition/artefact (dissertation theoretical anchor)
- Pew Research Center, American Trends Panel Wave 173, *How Americans View AI and Its
  Impact on People and Society*, June 2025 (behavioral data; PDFs in `data/`)
- US Census Bureau, American Community Survey 2023 (demographic marginals)
