# Judge calibration methodology

An LLM-as-judge scorer is an unverified classifier until you check it
against ground truth. This project treats that check as a first-class,
versioned artifact rather than an assumption.

**This project defaults the judge to `gpt-4o-mini`** -- the same cheap
tier as the task model, purely for cost. That makes this calibration
check load-bearing rather than a formality: a cheap judge grading a
cheap model's output is exactly the setup where self-preference bias and
missed subtle hallucinations are most likely to slip through unnoticed.
Don't skip running this because "it's just the mini tier, how wrong
could it be" -- that's the assumption this file exists to replace with a
number. If the numbers below come back weak, upgrade `HARNESS_JUDGE_MODEL`
to `gpt-4o` for the judge role only; the task model can stay on the mini
tier regardless.

## What's being tested

`golden/calibration_labels.jsonl` holds 14 hand-authored
`(question, candidate_answer, reference_answer, human_label)` triples,
independent of the RAG-fusion target app entirely -- no retrieval, no
adapter, no live model run. This isolates one question: **given a
question, a reference answer, and a candidate answer, does the judge's
0-10 score track a human's judgment of quality?**

The set is deliberately adversarial to the judge, not just a spread of
good/bad answers. It specifically includes:

| Case type | Why it's there |
|---|---|
| Near-verbatim correct answer | Sanity check -- if the judge fails this, nothing else matters |
| Correct paraphrase (different wording) | Catches a judge that's secretly doing string matching instead of reading for meaning |
| Fabricated/hallucinated specific ("cheaper than solar power") | The single most important failure mode: a plausible-sounding invented fact must score low even though it "reads" confident and fluent |
| Direct contradiction of the reference | Judge must not be fooled by confident tone |
| Verbose-but-correct | Checks the judge doesn't penalize (or over-reward) length independent of content |
| Excessive hedging on an answerable question | Checks the judge notices when hedging substitutes for actually answering |
| Unwarranted refusal ("I don't have enough information") on an answerable question | A refusal isn't automatically safe/correct -- it's wrong if the answer was available |
| Appropriate refusal on a genuinely out-of-corpus question | The mirror case: refusing correctly must score as *high* as answering correctly |
| Confidently hallucinating an answer to an out-of-corpus question | The other half of the refusal pair -- answering when you should have declined |
| Mixed correct-and-wrong content | Tests graded partial credit, not just a binary correct/incorrect read |

Run it with:

```bash
python -m harness.cli calibrate --provider openai
```

## Metrics reported

- **Mean absolute error** between judge score (0-1) and human label (0-1)
- **Pearson correlation** between the two series
- **Agreement within +/-0.2** -- the fraction of cases where the judge landed
  within a fairly generous tolerance of the human label

## Passing bar

The `calibrate` command exits non-zero (and prints a warning) if MAE > 0.25
or correlation < 0.5. This is a judgment call, not a law of nature --
tighten it once you have more calibration data than 14 cases.

## Results

*(Fill this in after running `calibrate --provider openai` for real --
the numbers below are from a `--provider mock` smoke run and only prove the
plumbing works, not that the judge is any good. Mock always scores 0 on
every case, since it doesn't actually reason about the question, which is
exactly why the command correctly reports the judge as uncalibrated in
that mode.)*

| Judge model | MAE | Correlation | Agreement (+/-0.2) | Notes |
|---|---|---|---|---|
| mock (plumbing check only) | 0.493 | undefined (zero variance) | 35.7% | Expected -- mock cannot reason, this only proves the harness runs end to end |
| gpt-4o-mini (default) | _TODO: run and fill in_ | | | The number that actually matters -- this is what the harness uses unless overridden |
| gpt-4o (fallback, only if the mini row above is weak) | _TODO if needed_ | | | |

## What this doesn't cover

This tests the judge's *scoring* ability on a fixed, small, hand-picked
set. It does not test:

- Whether the judge is stable across repeated calls on the *same* input
  (temperature/sampling variance) -- worth adding if you see the judge
  disagree with itself run to run.
- Whether the judge's calibration holds up on a much larger and more
  diverse set of failure modes than the 14 curated here. Fourteen cases
  catch the obvious failure classes; they are a floor, not a ceiling.
- Judge self-preference bias when the judge model is the same model as
  the one being graded. As of the current default, **this is a live risk,
  not a hypothetical one**: judge and task model are both `gpt-4o-mini`.
  The calibration set's `human_label`s are independent of any model's
  self-assessment, so a poor MAE/correlation here would be one real
  signal of this bias -- but 14 cases against one fixed candidate-answer
  set can't fully rule it out for arbitrary future outputs. If you
  upgrade the judge to `gpt-4o` while keeping the mini tier as the task
  model, this risk goes away by construction; the code doesn't yet
  enforce that as a hard rule, so it's on you to keep them different if
  this matters for your use case.
