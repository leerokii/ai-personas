# How to Work With Me on This Project

I'm building a take-home task for Artificial Societies (Forward Deployed Engineer role).
The full spec is in PROJECT_BRIEF.md. Read it fully before doing anything.

## The Goal (in one sentence)
Build four different methods of constructing 100 LLM personas of US women (all ages, 18+),
ask each panel of 100 the same five survey questions, then compare the methods on
five dimensions to recommend which one a senior decision-maker should trust.

## Working Rules

1. **Small steps, stop often.** Don't try to build the whole project at once.
   Build, show me, get sign-off, then continue.

2. **Cost guard from the start.** Before any API call estimated to cost more
   than $1, tell me the estimated cost and wait for me to confirm. Print
   running cost after each batch.

3. **No expensive runs until pipelines are validated on small samples.** Test
   each method with 5 personas and 1 question first. If that looks right,
   then scale to 100 personas and 5 questions.

4. **Default model is claude-sonnet-4-6.** Do not switch models without my
   approval. Keep my Anthropic API key in environment variables, never in
   code or git.

5. **Commit small with present-tense messages.** The commit history should
   read as a research log. Don't squash. Don't force-push.

6. **Ask before installing non-standard packages.** I want to keep dependencies
   minimal. Allowed without asking: anthropic, pyyaml, pandas, matplotlib,
   numpy, python-dotenv, jupyter.

7. **Prompts live in `prompts/` as versioned text files**, never as inline
   strings in Python code.

8. **If you're not sure about a design choice, ask.** Don't guess.

## Phase Checklist

Work through these in order. Each phase ends with a STOP where you summarize
what you did and wait for my sign-off before continuing.

### Phase 1: Repo Skeleton
- [ ] Read PROJECT_BRIEF.md fully
- [ ] Create folder structure matching the layout in the brief
- [ ] Create requirements.txt with: anthropic, pyyaml, pandas, matplotlib,
      numpy, python-dotenv, jupyter
- [ ] Create .gitignore (.env, __pycache__, .DS_Store, *.pyc, results/raw/*,
      .ipynb_checkpoints, .venv)
- [ ] Create .env.example with one line: ANTHROPIC_API_KEY=your-key-here
- [ ] Prompt me to create my real .env file and paste my API key
- [ ] Create config.yaml with: model name, temperature, num_personas (100),
      num_repeats (3), budget_cap ($20)
- [ ] Build src/client.py with an Anthropic API wrapper that tracks running
      cost (input tokens × input price + output tokens × output price)
- [ ] Initialize git repo, first commit message: "chore: scaffold repo"
- [ ] STOP. Show me the directory tree and the client.py code.

### Phase 2: Method A Smoke Test
- [ ] Write prompts/method_a.txt (one-prompt baseline)
- [ ] Write src/personas.py with a method_a function
- [ ] Write src/survey.py to administer one question and parse a single-letter
      response with one retry on parse failure
- [ ] Run 5 personas × 1 question as a smoke test
- [ ] Log all responses to results/raw/method_a_smoke.jsonl
- [ ] Print actual cost of the smoke test
- [ ] STOP. Show me the 5 raw responses, parse success rate, and cost.

### Phase 3: Method A Full Scale
- [ ] Estimate cost of 100 personas × 5 questions × 3 repeats (1,500 calls)
- [ ] Confirm estimate with me before running
- [ ] Run full Method A
- [ ] Log all responses to results/raw/method_a_full.jsonl
- [ ] Build src/evaluate.py with entropy calculation per question
- [ ] Generate first draft of Figure 1 showing Method A response distributions
- [ ] STOP. Show me the figure, entropy numbers and total cost so far.

### Phase 4: Methods B, C, D (one at a time)
For each method, complete this sub-checklist before moving to the next:
- [ ] Write the prompt file in prompts/
- [ ] Add the method function to src/personas.py
- [ ] Smoke test with 5 personas × 1 question
- [ ] Estimate full-scale cost and confirm with me
- [ ] Run 100 personas × 5 questions × 3 repeats
- [ ] Log to results/raw/method_X_full.jsonl
- [ ] STOP. Show me sample responses and cost before moving to the next method.

### Phase 5: Full Evaluation
- [ ] Implement all 5 evaluation dimensions in src/evaluate.py:
  - [ ] Response diversity (entropy per method per question)
  - [ ] Demographic coherence (subgroup answer differences)
  - [ ] Internal consistency (mean ± range across 3 repeats)
  - [ ] Behavioral consistency (Method D only)
  - [ ] Face validity (sample 5 narratives per method, rate 1-5)
- [ ] Generate Figure 1: response distribution by method (stacked bars per
      question, four bars per question)
- [ ] Generate Figure 2: entropy comparison across methods
- [ ] Generate Figure 3: demographic coherence on one question (split by
      education)
- [ ] STOP. Show me all three figures and the evaluation tables.

### Phase 6: Writeup
- [ ] Fill in README.md with: problem framing, methods summary, findings,
      decision recommendation, limitations, future work paragraph, references
- [ ] Polish notebook.ipynb to read as a narrative walkthrough for reviewers
- [ ] Run the full pipeline end-to-end one final time to confirm reproducibility
- [ ] Print final total cost
- [ ] Final commit with message: "docs: complete writeup and findings"
- [ ] STOP. Show me the README, the notebook and the final cost.

## How to Report at Each STOP

When you hit a STOP point, use this format:

**Phase complete:** [phase name]
**What I built:** [bullet list of files created or changed]
**Results:** [key numbers, sample outputs, any figures]
**Choices I made:** [design decisions you'd want me to review]
**Estimated cost of next phase:** [dollar estimate]
**Ready to continue?** Wait for my reply.

## What "Done" Looks Like for the Whole Project

A GitHub repo containing:
- A clean README with framing, methods, findings and recommendation
- A notebook that walks a reviewer through the full study
- Source code in src/, prompts in prompts/
- All raw responses in results/raw/ as JSONL
- Three figures in results/figures/
- A complete commit history that reads as a research log
- Total spend under $20

Now read PROJECT_BRIEF.md and start Phase 1. Do not start Phase 2 until
I've signed off on Phase 1.
