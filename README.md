# AI Model Harness

A provider-agnostic evaluation harness for LLM-powered applications --
regression gating, judge calibration, cost/budget controls, and response
caching -- demonstrated against a real third-party RAG pipeline rather
than a toy app written to make the harness look good.

**This project is the harness.** The target app it evaluates
([Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion), MIT
licensed, pinned to commit `e6ae2d4`) is vendored, not written by this
project -- see [Target app](#target-app-and-attribution) below for exactly
what's reused versus reimplemented.

## Why this exists

An eval harness is the regression-test suite for a system whose output
isn't deterministic. You can't assert `response == "expected string"` on
an LLM call, so this harness instead:

1. Runs a versioned **golden dataset** (25 hand-authored cases in
   [`golden/rag_fusion_qa.jsonl`](golden/rag_fusion_qa.jsonl), including
   adversarial/out-of-corpus and cross-topic cases, not just happy path)
2. Scores each case with **multiple scorer types** -- deterministic
   substring/citation checks, retrieval hit-rate, embedding similarity,
   and LLM-as-judge
3. **Diffs against the previous run** for the same (task, model) pair, so
   a prompt change that fixes 3 cases and silently breaks 2 others gets
   caught instead of averaged away
4. **Gates** on a quality bar + zero pinned-regression-failures + zero
   new regressions, with a non-zero exit code a CI pipeline can act on
5. Tracks **cost, latency, and run history** over time in SQLite

The part most eval harnesses skip: an **LLM judge is an unverified
classifier** until you check it against ground truth. See
[CALIBRATION.md](CALIBRATION.md) for the methodology and results.

## Architecture

```
golden/*.jsonl ──► dataset.py ──► EvalCase[]
                                      │
                                      ▼
                    ┌───────────────────────────────┐
                    │            Runner              │  (bounded thread pool,
                    │  adapter.run() per case + all   │   per-case error isolation)
                    │      scorers per case            │
                    └───────────────┬────────────────┘
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                              ▼
  RagFusionAdapter            ModelProvider chain              Scorers
  (wraps vendored          Anthropic ──► BudgetGuard        contains,
   rag-fusion pipeline,        ──► ResponseCache            retrieval_hit_rate,
   see below)                (or MockProvider, $0)          embedding_similarity,
                                                              llm_judge (calibrated)
                                     │
                                     ▼
                              RunStore (SQLite)
                                     │
                        ┌────────────┴────────────┐
                        ▼                          ▼
                  diff.py (regressions)     report.py (Markdown)
                        │
                        ▼
                  cli.py `gate` (exit code for CI)
```

## Target app and attribution

[Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion) implements
RAG-Fusion: multi-query expansion + Reciprocal Rank Fusion over a vector
store. 957+ stars, MIT licensed, actively maintained.

**What's reused verbatim** (imported directly from the vendored copy,
never modified): `vector_search` and `reciprocal_rank_fusion` -- this
logic is already model-agnostic.

**What's reimplemented** in [`harness/adapters/rag_fusion.py`](harness/adapters/rag_fusion.py):
query expansion and answer synthesis, which the original hardcodes to
OpenAI's client. These are rewritten against this project's own
`ModelProvider` abstraction with equivalent prompts, so the harness can
run different Claude models through an otherwise-identical pipeline.

**Worth knowing:** rag-fusion already ships its own, fairly sophisticated
eval tooling (`eval/` -- retrieval metrics against NFCorpus, an LLM-judge
answer-quality eval with pairwise ranking, bootstrap confidence
intervals). This harness doesn't duplicate that; it adds what that
tooling doesn't have -- a reusable, provider-agnostic framework with
multi-model comparison, CI-gating, and regression diffing, none of which
are specific to this one repo.

The vendored copy lives at `targets/rag-fusion` (gitignored) and is
fetched, pinned to an exact commit, by `scripts/setup_target.sh`.

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv\Scripts\activate on cmd.exe
pip install -r requirements-dev.txt
bash scripts/setup_target.sh    # vendors targets/rag-fusion, pinned commit

# Free: mock provider, no API key, no network calls to any model
python -m harness.cli run --provider mock --smoke 8

pytest -q                        # 28 tests, all mock-based, $0
```

To run against real models, copy `.env.example` to `.env`, fill in
`ANTHROPIC_API_KEY`, then:

```bash
python -m harness.cli run --provider anthropic --model claude-haiku-4-5-20251001 --smoke 10
python -m harness.cli gate --model claude-haiku-4-5-20251001 --min-score 0.75
python -m harness.cli calibrate --provider anthropic
```

## Models

| Role | Model | Why |
|---|---|---|
| Cheap task model | `claude-haiku-4-5-20251001` | Fast/cheap tier, default for iteration |
| Capable task model | `claude-sonnet-5` | Comparison point -- the whole exercise is measuring the Haiku/Sonnet tradeoff on identical inputs |
| Judge | `claude-sonnet-5` | Stronger than the cheap tier being graded; calibrated in [CALIBRATION.md](CALIBRATION.md) rather than trusted blindly |
| Embeddings | `all-MiniLM-L6-v2` (local, via sentence-transformers) | Free, no API cost, run as many similarity checks as needed |

## Cost controls

Three independent mechanisms, not one:

- **`BudgetGuard`** ([`harness/core/budget.py`](harness/core/budget.py)) --
  a hard spend cap checked *before* every paid call using a worst-case
  token estimate. A run stops before overshooting the budget, not after.
  Default $5/run, set via `HARNESS_MAX_COST_USD` or `--max-cost-usd`.
- **`ResponseCache`** ([`harness/core/cache.py`](harness/core/cache.py)) --
  SQLite-backed, keyed by `(model, system, prompt, tools)`. Re-running
  after a crash, or regenerating a report, never re-spends money on a
  case that already ran.
- **`--smoke N`** -- runs the pinned regression cases plus the first N
  others, for fast/cheap iteration; reserve the full golden set for
  actual gating decisions.

Rough per-full-run cost (25 cases, Haiku + Sonnet task calls + Sonnet
judge): **under $1**, mostly the judge calls. Verify current per-token
pricing at [anthropic.com/pricing](https://anthropic.com/pricing) --
the numbers in [`harness/core/pricing.py`](harness/core/pricing.py) are
estimates for relative cost tracking, not a guarantee.

## CLI reference

| Command | Purpose |
|---|---|
| `harness run` | Execute the golden set, score it, save to SQLite, write a Markdown report |
| `harness diff` | Show regressions/improvements vs. the previous run for a (task, model) |
| `harness gate` | Exit non-zero if mean score / pinned cases / regressions fail the bar -- for CI |
| `harness calibrate` | Check the LLM judge against `golden/calibration_labels.jsonl` |
| `harness history` | Show run history (score, cost, errors) for a (task, model) over time |

Every subcommand defaults to `--provider mock` so a command copy-pasted
from this README can't accidentally spend money.

## CI

Two separate workflows, deliberately not one:

- **[`ci.yml`](.github/workflows/ci.yml)** -- runs on every push/PR, free,
  `--provider mock` throughout. Proves the harness's own code is correct.
  Tells you nothing about the target app's actual answer quality.
- **[`eval-gate.yml`](.github/workflows/eval-gate.yml)** -- the workflow
  that costs real money. Manually triggered or nightly, *not* on every
  push -- an eval-gate wired to every commit (including from forks) is an
  unbounded, unreviewed spend trigger, which is a real production mistake
  worth avoiding deliberately rather than discovering by bill.

## Testing

```bash
pytest -q
ruff check harness tests
```

28 tests, all against `MockProvider` or hand-built fakes -- zero API cost,
zero network dependency beyond the one-time sentence-transformers model
download. Covers dataset validation, every scorer's logic (including a
judge given deliberately malformed output), the budget guard's
before-vs-after accounting, cache-hit/miss behavior, the diff engine's
regression/improvement/new-case classification, and the runner's error
isolation (one bad case, or one broken scorer, must not sink the run --
this is asserted directly, not assumed).

## Known limitations

Honest scope notes, not hidden gaps:

- **Golden set is a seed, not a finished suite.** 25 cases is enough to
  exercise every scorer type and prove the methodology; a real production
  gate wants 100+, grown from cases that actually broke in the wild.
- **SQLite, single-node.** Fine for one contributor or a small team's CI;
  a concurrent-writer service would need a real database.
- **ChromaDB's ephemeral client is not proven thread-safe** for concurrent
  queries -- the adapter serializes all vector-store access behind one
  lock rather than assume otherwise. This was found by actually running
  the harness under concurrency during development, not by inspection --
  see the comment in `harness/adapters/rag_fusion.py`.
- **Judge calibration set is 14 cases.** It's adversarial and covers the
  major failure classes (hallucination, contradiction, unwarranted
  refusal, hedging), but it's a floor, not a statistically powered study.
- **No async I/O** -- concurrency is a bounded thread pool, which is
  sufficient at this scale (network-bound API calls) and much simpler to
  reason about than an asyncio rewrite would be for the size of workload
  this handles.

## License

This project: MIT. Vendored target app: MIT (Raudaschl/rag-fusion),
unmodified, attributed above.
