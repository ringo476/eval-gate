# AI Model Harness

A provider-agnostic evaluation harness for LLM-powered applications --
regression gating, judge calibration, cost/budget controls, and response
caching -- demonstrated against a real third-party RAG pipeline rather
than a toy app written to make the harness look good.

**This project is the harness.** It evaluates two different kinds of LLM
application through the same core (runner, scorers, gating, diffing):

- **`rag_fusion_qa`** -- a **fixed pipeline** wrapping a vendored
  third-party app ([Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion),
  MIT licensed, pinned to commit `e6ae2d4`; see
  [Target app](#target-app-and-attribution) for what's reused vs.
  reimplemented). Every run executes the same steps in the same order.
- **`tool_agent_qa`** -- a genuinely **agentic** task
  ([`harness/adapters/tool_agent.py`](harness/adapters/tool_agent.py)):
  the model decides, turn by turn, whether to call `calculator` or
  `kb_lookup`, with what arguments, whether to chain a second tool call,
  and when it has enough to answer. The adapter drives the loop and
  executes tools locally; the model controls the control flow, not a
  hardcoded step sequence. This is what separates it from the fixed
  RAG-fusion pipeline above -- see [Known limitations](#known-limitations)
  for exactly where that distinction matters.

## Why this exists

An eval harness is the regression-test suite for a system whose output
isn't deterministic. You can't assert `response == "expected string"` on
an LLM call, so this harness instead:

1. Runs a versioned **golden dataset** per task (25 cases in
   [`golden/rag_fusion_qa.jsonl`](golden/rag_fusion_qa.jsonl), 10 in
   [`golden/tool_agent_qa.jsonl`](golden/tool_agent_qa.jsonl)), including
   adversarial, out-of-scope, and multi-step cases, not just happy path
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
golden/*.jsonl ──► dataset.py ──► EvalCase[]      (harness/tasks.py maps a
                                      │            task name to its dataset +
                                      ▼             adapter + scorer set)
                    ┌───────────────────────────────┐
                    │            Runner              │  (bounded thread pool,
                    │  adapter.run() per case + all   │   per-case error isolation)
                    │      scorers per case            │
                    └───────────────┬────────────────┘
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                              ▼
  One adapter per task:        ModelProvider chain              Scorers
  RagFusionAdapter            OpenAI/Anthropic ──► BudgetGuard (per task, e.g.
  (fixed pipeline)                ──► ResponseCache             contains,
  ToolAgentAdapter             (or MockProvider, $0)            retrieval_hit_rate /
  (agentic tool-call loop,     generate() single-turn,          tool_usage,
   see below)                  generate_messages() multi-turn   embedding_similarity,
                                                                 llm_judge (calibrated))
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
query expansion and answer synthesis, which the original hardcodes
directly to OpenAI's client with no retry/caching/budget logic. These
are rewritten against this project's own `ModelProvider` abstraction
with equivalent prompts -- this project also defaults to OpenAI, so the
underlying API is the same one the original uses, but going through
`ModelProvider` adds response caching, a hard spend cap, retries, and the
ability to swap in a different model (including a different provider
entirely, e.g. `--provider anthropic`) without touching the adapter.

**Worth knowing:** rag-fusion already ships its own, fairly sophisticated
eval tooling (`eval/` -- retrieval metrics against NFCorpus, an LLM-judge
answer-quality eval with pairwise ranking, bootstrap confidence
intervals). This harness doesn't duplicate that; it adds what that
tooling doesn't have -- a reusable, provider-agnostic framework with
multi-model comparison, CI-gating, and regression diffing, none of which
are specific to this one repo.

The vendored copy lives at `targets/rag-fusion` (gitignored) and is
fetched, pinned to an exact commit, by `scripts/setup_target.sh`.

## Tool-calling agent task

`tool_agent_qa` ([`harness/adapters/tool_agent.py`](harness/adapters/tool_agent.py))
is a small agent with two tools
([`harness/adapters/tools.py`](harness/adapters/tools.py)):

- `calculator` -- arithmetic only. Parses a restricted grammar with
  Python's `ast` module rather than `eval()`-ing model-supplied text --
  the model's output is an untrusted input source here, same as any
  other external input, so the tool only allows numbers, `+ - * ( )`, and
  rejects everything else (name lookups, attribute access, calls) before
  it can be evaluated.
- `kb_lookup` -- semantic search over the same document corpus the
  RAG-fusion task uses ([`harness/corpus.py`](harness/corpus.py)), via
  the shared embedding model in
  [`harness/core/embeddings.py`](harness/core/embeddings.py).

The adapter drives a loop (`ModelProvider.generate_messages`, capped at
`MAX_STEPS = 4`): the model sees the tool results and decides whether to
call another tool or give a final answer -- it is not told how many
steps to take. The golden set includes single-tool cases, a two-hop case
that requires chaining two tool calls, a multi-tool case, a
division-by-zero error-handling case, and a no-tool-needed case that
checks the model doesn't force a tool call when neither applies.

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv\Scripts\activate on cmd.exe
pip install -r requirements-dev.txt
bash scripts/setup_target.sh    # vendors targets/rag-fusion, pinned commit

# Free: mock provider, no API key, no network calls to any model
python -m harness.cli run --task rag_fusion_qa --provider mock --smoke 8
python -m harness.cli run --task tool_agent_qa --provider mock --smoke 10

pytest -q                        # 43 tests, all mock-based, $0
```

To run against real models, copy `.env.example` to `.env`, fill in
`OPENAI_API_KEY` (get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)), then:

```bash
python -m harness.cli run --task rag_fusion_qa --provider openai --model gpt-4o-mini --smoke 10
python -m harness.cli run --task tool_agent_qa --provider openai --model gpt-4o-mini --smoke 10
python -m harness.cli gate --task rag_fusion_qa --model gpt-4o-mini --min-score 0.75
python -m harness.cli calibrate --provider openai
```

Anthropic is also supported through the same `ModelProvider` abstraction
(`--provider anthropic`, `ANTHROPIC_API_KEY` in `.env`) but isn't the
default for this project.

## Models

| Role | Model | Why |
|---|---|---|
| Task model (default) | `gpt-4o-mini` | Cheapest viable OpenAI tier -- used for both iteration and the actual run by default |
| Task model (optional comparison) | `gpt-4o` | Only added once the mini tier's results are in hand, to measure the cost/quality tradeoff on identical inputs -- not required to use this project |
| Judge (default) | `gpt-4o-mini` | Cheapest option, same tier as the task model. This is *not* trusted blindly -- see [CALIBRATION.md](CALIBRATION.md): run `harness calibrate` and check the MAE/correlation before relying on it for gating. Bump to `gpt-4o` (via `HARNESS_JUDGE_MODEL`) only if calibration shows the mini tier isn't good enough, and only for the judge role. |
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

Rough per-full-run cost with the default all-`gpt-4o-mini` setup (25 RAG
cases + 10 agent cases, task + judge calls): **a few cents**. Verify
current per-token pricing at
[platform.openai.com/docs/pricing](https://platform.openai.com/docs/pricing) --
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

43 tests, all against `MockProvider` or hand-built fakes -- zero API cost,
zero network dependency beyond the one-time sentence-transformers model
download. Covers dataset validation, every scorer's logic (including a
judge given deliberately malformed output), the budget guard's
before-vs-after accounting, cache-hit/miss behavior for both single-turn
and multi-turn (agent) calls, the diff engine's regression/improvement/
new-case classification, the runner's error isolation (one bad case, or
one broken scorer, must not sink the run), the calculator tool's refusal
to evaluate anything outside plain arithmetic, and the agent loop itself
end to end with `MockProvider`.

## Known limitations

Honest scope notes, not hidden gaps:

- **Golden sets are seeds, not finished suites.** 25 RAG cases and 10
  agent cases are enough to exercise every scorer type and prove the
  methodology; a real production gate wants 100+ per task, grown from
  cases that actually broke in the wild.
- **`MockProvider`'s tool-choice heuristic is deliberately crude**
  (checks for digits + a short list of operator words to decide
  calculator vs. kb_lookup) -- it exists to exercise the agent loop's
  code paths for $0, not to simulate good judgment. Don't read anything
  into `--provider mock` scores on `tool_agent_qa`; they measure the
  mock's heuristic, not agent quality.
- **The agent has a hard step ceiling (`MAX_STEPS = 4`).** A case that
  needs more than 4 tool-call turns to answer correctly will report a
  clean "exceeded MAX_STEPS" error rather than hang -- by design, but
  worth knowing before writing a golden case that needs deep chaining.
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
