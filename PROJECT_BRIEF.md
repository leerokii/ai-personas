# Project Brief: LLM Persona Construction Methods
### Testing four ways of building LLM personas of US women aged 18 to 35, on questions about AI for finding online communities.

## What the Brief Asks For (Min's email, paraphrased)
- Build 100 LLM personas modelling one group of humans
- Ask them a few single-select survey questions
- Use the results to inform a consequential decision
- Try different methods of creating personas, so the panel captures the
  group's opinions as well as possible
- Present methods and findings to senior decision-makers

The unit is 100 personas. The variation is in how those 100 are built. This
brief tests four different construction methods, each producing its own
100-persona panel.

## The Consequential Decision
Should a developer build an AI-powered tool to help US women aged 18 to 35
find online communities, and if so, what design choices should they make?

Five survey questions inform this decision: usage, trust, context, friction
and tone.

## The Group
US women aged 18 to 35.

## The Five Survey Questions (single-select)
1. How often do you currently use AI tools to help you find online communities
   or groups to join?
2. How comfortable would you be if an AI suggested an online community for you
   based on your interests?
3. Which kind of community would you most want an AI to help you find?
   (e.g. hobby, professional, support, local, identity-based)
4. What would most make you stop using an AI community-finder?
   (e.g. inaccurate matches, privacy concerns, lack of human moderators,
   feeling surveilled, low-quality communities)
5. How should an AI describe a community to you when recommending it?
   (e.g. neutral facts only, warm and personal, identity-aware, like a
   friend's recommendation)

Each question gets 4 to 5 single-select options. Finalize during build.

## The Four Persona Construction Methods
Each method produces a 100-persona panel. Each rung adds one variable.

### Method A — Naive baseline
One prompt: "Answer this question as a random US woman aged 18 to 35." Sample
100 times, temperature 1.0. Same survey for all 100. Hypothesis: collapses
toward modal answers, undercounts minority opinions.

### Method B — Demographic conditioning
Generate 100 demographic profiles matching the joint distribution of US women
18 to 35 across age band, race/ethnicity, education, income, region and
urbanicity. Use US Census or Pew published marginals. Each profile becomes a
system prompt. Hypothesis: better demographic representativeness, residual
stereotyping.

### Method C — Narrative persona expansion
Feed each Method B profile to the LLM. Generate a first-person backstory
(~150 words: values, occupation, daily life, friendships, online habits).
Survey the narrative persona. Hypothesis: richer individual variation, risk
of caricature.

### Method D — Behaviorally grounded personas
Layer behavioral traits onto each Method B profile: AI usage frequency,
primary digital platforms, online community participation level, comfort
with AI tools. Source these from Pew's published distributions for women in
this age band. Narratives anchor to empirical behavioral facts.

Mirrors Artificial Societies' stated approach ("constructed from real-world
social behavior data") and tests whether the behavioral layer improves output.

## Evaluation: Five Dimensions of Method Comparison

The brief asks methods to "capture the human group's opinions as much as
possible." Without an external benchmark for these specific questions, the
evaluation compares the four methods to each other along dimensions that
matter to a senior decision-maker.

### 1. Response Diversity
- Response entropy per question, per method
- Hypothesis: Method A is low-entropy (one voice), Methods B to D higher

### 2. Demographic Coherence
- Within Methods B to D, split personas by education, age band, region
- Do subgroups answer in plausible directions given research on women and
  digital platforms (cite Bucher on algorithmic imaginaries, Hogan on
  self-presentation)
- Hypothesis: Methods C and D show stronger and more consistent subgroup
  effects than Method B alone

### 3. Internal Consistency
- 3 repeat runs per method, same personas, same questions
- Report mean ± range
- Hypothesis: methods with richer persona context (C, D) are more stable

### 4. Behavioral Consistency (Method D specific)
- Method D personas have stated behaviors (heavy AI users, light AI users)
- Do their answers align with those behaviors?
- Structural validity check on the behaviorally-grounded approach

### 5. Face Validity
- Sample 5 narratives per method (C and D)
- Rate 1 to 5: realistic, generic, stereotyped, caricature
- Brief written justification per rating

## Implementation Notes
- Model: Claude (current Sonnet) via Anthropic API. Optional stretch: rerun
  Method D on a second model.
- Force single-select output via constrained format (option letter only),
  validate, retry once on parse failure, log failures.
- Async/batched calls. 100 personas × 5 questions × 4 methods × 3 reps ≈
  6,000 calls plus Method C and D persona generation. Print cost guard.
  Cap total spend ~$20.
- Seed everything seedable. Log raw responses as JSONL in `results/raw/`.
- Prompts live in `prompts/` as versioned text files.

## Repo Layout
```
ai-personas/
  README.md
  notebook.ipynb
  src/
    personas.py        # methods A to D
    survey.py          # administration, parsing, retries
    evaluate.py        # entropy, subgroup cuts, consistency, behavioral
    run.py             # orchestration + cost guard
  prompts/
  data/                # reference distributions for Methods B and D
  results/raw/  results/figures/
  config.yaml
```

## Figures (exactly three)
1. Response distribution by method: stacked bars per question, four bars
   per question
2. Entropy comparison: how much variation each method produces per question
3. Demographic coherence on one question: split simulated personas by
   education, plot answer shares per method

## Commit Discipline
Small commits, present-tense messages, in chronological exploration order:
skeleton, baseline, methods B/C/D, evaluation harness, figures, writeup.
History should read as a research log.

## Future Work (one paragraph in README)
Name the networked-personas extension as the natural next step: place
behaviorally-grounded personas on a small-world graph, let each see neighbors'
answers in a second round, measure whether social exposure shifts the panel.
Tests Artificial Societies' interconnected-persona thesis directly.
Community-formation is the precise domain where social context should matter
most. Out of 48-hour scope to build, in scope to articulate.

## README Findings Section (write last)
- Recommendation: which method to use for the community-finder decision,
  and where the panel shouldn't replace human research
- Limitations: variance flattening, training-data opinion skew, stereotype
  amplification, model dependence, no external ground truth (by design)
- What each evaluation dimension revealed

## Dissertation Through-Line (for the presentation only)
Opening line draft: "My Oxford research examined how young women collectively
imagine and shape their relationships to algorithmic platforms. This task
asks the inverse question. Whether language models can credibly simulate
those imaginaries back to us. And where they break."

## References
- Santurkar et al. 2023, "Whose Opinions Do Language Models Reflect?"
- Argyle et al. 2023, "Out of One, Many"
- Park et al. 2024, generative agent simulations
- Artificial Societies Survey Eval Report, Jan 2026
- He et al., British Journal of Psychology (founders' AI society paper)
- Bucher 2017, algorithmic imaginaries
- Hogan 2010, exhibition/artefact (dissertation theoretical anchor)
