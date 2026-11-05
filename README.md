# AI Model Harness

A provider-agnostic evaluation harness for LLM-powered applications:
**regression gating against committed baselines**, judge calibration,
repeat-sampling to separate flaky from broken, cost/budget controls, response
caching, and first-class support for **local models** (Ollama) next to hosted
ones (OpenAI, Anthropic) -- demonstrated against a real third-party RAG
pipeline rather than a toy app written to make the harness look good.

**This project is the harness.** It evaluates two different kinds of LLM
application through the same core (runner, scorers, gating, diffing):

- **`rag_fusion_qa`** -- a **fixed pipeline** wrapping a vendored
  third-party app ([Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion),
  MIT licensed, pinned to commit `e6ae2d4`; see
  [Target app](#target-app-and-attribution)). Every run executes the same
  steps in the same order.
- **`tool_agent_qa`** -- a genuinely **agentic** task
  ([`harness/adapters/tool_agent.py`](harness/adapters/tool_agent.py)):
  the model decides, turn by turn, whether to call `calculator` or
  `kb_lookup`, with what arguments, whether to chain a second call, and when it
  has enough to answer. The adapter drives the loop and executes tools locally;
  the model controls the control flow.

## Why this exists

An eval harness is the regression-test suite for a system whose output isn't
deterministic. You can't `assert response == "expected string"` on an LLM call,
so this harness instead:

1. Runs a versioned **golden dataset** per task (25 cases for RAG, 10 for the
   agent), including adversarial, out-of-scope, error-handling and multi-step
   cases, not just the happy path
2. Scores each case with **multiple scorer types** -- deterministic
   substring/retrieval/tool-usage checks, embedding similarity, and an
   LLM-as-judge
3. **Diffs against a committed baseline**, per case, so a prompt change that
   fixes 3 cases and silently breaks 2 others is caught instead of averaged away
4. **Gates** with an exit code CI can act on: quality bar + pinned cases +
   no regressions -- and refuses runs that are partial, aborted, errored or
   stale
5. Tracks **cost, latency and provenance** (git SHA, dataset hash, judge,
   settings) for every run in SQLite

The part most eval harnesses skip: an **LLM judge is an unverified classifier**
until you check it against ground truth. [CALIBRATION.md](CALIBRATION.md) has
the methodology and measured results for four judges.

## Quickstart (free: no API key needed)

```bash
python -m venv .venv
source .venv/Scripts/activate        # .venv\Scripts\activate on cmd.exe
pip install -r requirements-dev.txt
bash scripts/setup_target.sh         # vendors targets/rag-fusion at a pinned commit
python -m harness.cli doctor         # checks packages, datasets, keys, local models

# 1. Mock provider: exercises the harness itself, $0, no network
python -m harness.cli run --task tool_agent_qa --provider mock --smoke 5

# 2. A real model, locally, via Ollama (ollama pull llama3.1:8b gpt-oss:20b)
python -m harness.cli run --task tool_agent_qa --provider ollama \
    --model llama3.1:8b --judge-model gpt-oss:20b

pytest -q                            # 170+ tests, all offline, $0
```

With an OpenAI key (copy `.env.example` to `.env` and set `OPENAI_API_KEY`):

```bash
python -m harness.cli calibrate --provider openai          # is the judge trustworthy? (cents)
python -m harness.cli run --task rag_fusion_qa --provider openai --model gpt-6-luna
```

Every subcommand defaults to `--provider mock`, so a command copy-pasted from
this README can't accidentally spend money.

## How regression gating works

```
 good full run ──► harness baseline set ──► baselines/<task>__<model>.json ──► git commit (reviewed in a PR)
                                                         │
 later change ──► harness run ──► harness gate --require-baseline ──► exit 0 / 1
                                                         │
                     compares every case's pass rate to the committed file
```

Why a **committed file** and not "the previous run in the database"? Because
the database is local state:

- a CI runner starts empty, so a history-based diff never has anything to
  compare against and the gate would pass vacuously;
- the previous run might be a `--smoke` subset or a degraded run, so the bar
  would silently drift down;
- nobody approved it.

A baseline only changes when someone runs `harness baseline set` and commits
the result, so moving the bar shows up in code review like any other change.

`harness gate` **fails** unless all of these hold (the policy is a pure,
unit-tested function in [`harness/core/gate.py`](harness/core/gate.py)):

| Rule | Why |
|---|---|
| Run wasn't aborted (budget cap, fatal error) | Results are incomplete |
| Run covered the full dataset (not `--smoke`) | A subset can't vouch for the rest |
| Run used the *current* dataset file (hash check) | A stale run from before the golden set changed proves nothing |
| No errored samples (`--max-errors`) | An error is not a pass |
| Mean score ≥ `--min-score` (and pass rate ≥ `--min-pass-rate`) | Absolute quality bar |
| Every pinned case passes every sample | "Must never regress" cases |
| No case regressed vs. the baseline | The actual point |
| No baseline case is missing from the run | Deleting a failing case must not turn the gate green |
| A committed baseline exists (`--require-baseline`) | A gate that compares against nothing can't go green |

### Flaky vs. broken: `--repeats`

LLM output varies. `--repeats N --temperature 0.5` runs every case N times as
independent samples (each a genuine fresh draw -- the cache is sample-aware), so a
case's result is a pass *rate*. Regression is a rate **drop of at least
`--regression-threshold`** (default 0.5); smaller drops are reported as "flakier
than baseline" without failing the gate, and `FLAKY` cases are listed in the
report. With one sample per case (the default) a flip is always a regression.

## Multi-model comparison

Run the same task on several models and compare them side by side -- including
a local model against a hosted one:

```bash
python -m harness.cli run --task rag_fusion_qa --provider ollama --model llama3.1:8b --judge-model gpt-oss:20b
python -m harness.cli run --task rag_fusion_qa --provider openai --model gpt-6-luna --judge-provider ollama --judge-model gpt-oss:20b
python -m harness.cli compare --task rag_fusion_qa --models llama3.1:8b gpt-6-luna
```

The judge can run on a different provider from the model under test
(`--judge-provider`), which is how you keep them different to avoid
self-preference bias. The comparison table shows pass rate with a 95% interval,
mean score, cost, errors and which judge scored each model, plus the cases where
models disagree. If judge and task model are the same, the report says so and
`--strict-judge` refuses to run.

## Measured results (local models, 2026-11-05)

Real runs on Ollama (AMD Radeon RX 9070), temperature 0, one sample per case,
all judged by `gpt-oss:20b` -- a *different* model from every model under test
(it scored MAE 0.056 / correlation 0.983 in [calibration](CALIBRATION.md)). Cost: $0.

| Task | Model | Pass rate (95% CI) | Mean score |
|---|---|---|---|
| `rag_fusion_qa` (25 cases) | llama3.1:8b | **96.0%** (80-99%) | 0.932 |
| | gemma2:9b | 92.0% (75-98%) | 0.946 |
| | qwen2.5:3b | 72.0% (52-86%) | 0.892 |
| `tool_agent_qa` (10 cases) | llama3.1:8b | **100%** (72-100%) | 0.995 |
| | qwen2.5:3b | 90.0% (60-98%) | 0.960 |
| | gemma2:9b | not run -- Ollama's `gemma2` has no tool-calling support | |

(Mean score is inflated by scorers that auto-pass cases with no requirement for
that check; pass rate is the more honest headline.)

What the harness found, on real output rather than the mock:

- **It blocks a real regression, with names.** Gating the `qwen2.5:3b` run against
  the committed `llama3.1:8b` baseline -- as if someone had swapped the model --
  exits 1: `6 case(s) regressed: rf-003, rf-007, rf-009, rf-018, rf-019, rf-023`,
  four of them pinned. Notably `rf-023`, the out-of-corpus "capital of France"
  case, failed on the small model.
- **The per-scorer breakdown localizes failures.** `llama3.1:8b`'s single RAG
  failure (`rf-024`) is a *retrieval* miss -- the cross-topic query found `doc19`
  but not `doc5` -- while generation scored 100%. Without the breakdown that
  would read as "the model got one wrong".
- **Repeat sampling exposes flakiness a single run hides.** The single-sample
  agent baseline said `ta-010` ("don't force a tool call") passed. Re-run at
  temperature 0.7 with 3 samples it passes 2 of 3 (once the model rambled about
  its own tools). The diff reports it as *flakier than baseline* without failing
  the gate.
- **It found a bug in the golden set.** An early run failed `ta-010` on embedding
  similarity because its reference answer describes expected *behavior* rather
  than giving a sample answer; the case now opts out with `skip_scorers`.

With n=25 the intervals overlap (llama's 80-99% vs qwen's 52-86%), so the 8B-vs-3B
RAG gap is suggestive, not conclusive -- though it is consistent across the
specific cases above. The `gpt-6-luna` row is not filled in: it needs an API key
(see [Known limitations](#known-limitations)). The commands are in the multi-model
section above, and `harness baseline set` blesses its run for CI.

## Architecture

```
golden/*.jsonl ──► dataset.py ──► EvalCase[]      (harness/tasks.py maps a task name
(validated, hashed)                 │              to dataset + adapter + scorer set)
                                    ▼
                    ┌────────────────────────────────────┐
                    │               Runner                │  bounded thread pool; per-case
                    │  cases x repeats, adapter + scorers │  error isolation; aborts the run
                    └───────────────┬─────────────────────┘  when the budget cap is hit
                                    │
        ┌───────────────────────────┼──────────────────────────────┐
        ▼                           ▼                               ▼
 One adapter per task:       ModelProvider chain               Scorers
 RagFusionAdapter           OpenAI / Anthropic / Ollama        contains,
 (fixed pipeline)           ──► BudgetEnforcing (paid only)    retrieval_hit_rate /
 ToolAgentAdapter           ──► Caching (sample-aware)         tool_usage,
 (agentic loop)             (or Mock, $0)                      embedding_similarity,
                                    │                          llm_judge (calibrated)
                                    ▼
                             RunStore (SQLite) ◄── provenance: git sha, dataset hash, judge, settings
                                    │
       ┌──────────────┬─────────────┴────────────┬────────────────┐
       ▼              ▼                          ▼                ▼
 baselines/*.json   diff.py                 report.py        cli.py `gate`
 (committed)        (rate-based regressions) (.md + .json)   (exit code for CI)
```

[FLOW.md](FLOW.md) is the exhaustive reference: every command's call sequence
and every function.

## Target app and attribution

[Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion) implements
RAG-Fusion: multi-query expansion + Reciprocal Rank Fusion over a vector store.
MIT licensed.

**Reused verbatim** (imported directly from the vendored copy, never modified):
`vector_search` and `reciprocal_rank_fusion` -- already model-agnostic.

**Reimplemented** in [`harness/adapters/rag_fusion.py`](harness/adapters/rag_fusion.py):
query expansion and answer synthesis, which the original hardcodes to OpenAI's
client with no retry/caching/budget logic. They are rewritten against this
project's `ModelProvider` abstraction with equivalent prompts, which adds
response caching, a hard spend cap, retries, and the ability to swap in any
provider -- including a local model -- without touching the adapter.

rag-fusion ships its own eval tooling (`eval/`: retrieval metrics against
NFCorpus, pairwise answer-quality ranking). This harness doesn't duplicate that;
it adds what that tooling doesn't have -- a reusable, provider-agnostic framework
with multi-model comparison, CI gating, and regression diffing.

The vendored copy lives at `targets/rag-fusion` (gitignored) and is fetched,
pinned to an exact commit, by `scripts/setup_target.sh` (`.ps1` on Windows).

## Tool-calling agent task

`tool_agent_qa` ([`harness/adapters/tool_agent.py`](harness/adapters/tool_agent.py))
is a small agent with two tools
([`harness/adapters/tools.py`](harness/adapters/tools.py)):

- `calculator` -- arithmetic only. It parses a restricted grammar with Python's
  `ast` module rather than `eval()`-ing model-supplied text (the model's output
  is an untrusted input, same as any other): only numbers, `+ - * /` and
  parentheses are allowed, names/attributes/calls are rejected, expressions are
  length-capped, and pathologically nested input returns an error string instead
  of crashing.
- `kb_lookup` -- semantic search over the same corpus the RAG task uses
  ([`harness/corpus.py`](harness/corpus.py)) via a local embedding model.

The adapter drives a loop (`generate_messages`, capped at `MAX_STEPS = 4`): the
model sees tool results and decides whether to call another tool or answer. The
golden set has single-tool cases, a two-hop case, a multi-tool case, a
division-by-zero error case, and a no-tool-needed case that checks the model
doesn't force a tool call (`"expected_tools": []` means *no tool may be called*;
omit the key for no requirement).

## Writing golden cases

One JSON object per line:

```json
{"id": "rf-001", "task": "rag_fusion_qa",
 "input": {"query": "How does climate change affect coral reefs?"},
 "expected": {"relevant_doc_ids": ["doc10"], "answer_contains": ["bleaching"],
              "reference_answer": "Rising ocean temperatures cause mass coral bleaching..."},
 "tags": ["biodiversity", "pinned"]}
```

- Every case should exist to catch **one specific failure**, not to add volume:
  happy path across the topic space, out-of-scope/adversarial cases (graded on
  *restraint*: `relevant_doc_ids: []`, `expected_tools: []`), error handling,
  multi-step cases.
- `pinned` marks must-never-regress cases: they always run in `--smoke`, and the
  gate requires them to pass every sample.
- `reference_answer` feeds embedding similarity **and** the judge. If it
  *describes expected behavior* ("should decline") rather than being a sample
  answer, add `"skip_scorers": ["embedding_similarity"]` -- comparing a real
  answer to a description of behavior is meaningless. (A real run caught exactly
  this: a correct "I don't have a favorite color" failed on embedding similarity
  to the behavior description.)
- The loader validates types, rejects duplicate ids, and rejects a case whose
  `task` doesn't match the file.

## Models

| Role | Default | Notes |
|---|---|---|
| Hosted task model | `gpt-6-luna` | Cheapest current OpenAI tier. Lineups change fast; re-check pricing before assuming this is current. |
| Hosted comparison | `gpt-6-sol` | Added once Luna's results are in hand, to measure the cost/quality tradeoff. |
| Local task model | any Ollama model | Tool calling needs a tool-capable model (`llama3.1`, `qwen2.5`, `gpt-oss`; **not** `gemma2`). |
| Judge (hosted) | `gpt-6-luna` | Not trusted blindly -- run `harness calibrate`. Prefer a judge different from the task model. |
| Judge (local) | `gpt-oss:20b` | Best of four calibrated local judges, see [CALIBRATION.md](CALIBRATION.md). |
| Embeddings | `all-MiniLM-L6-v2` (local) | Free; used by the similarity scorer and `kb_lookup`. |

## Cost and safety controls

- **`BudgetGuard`** -- a hard spend cap checked *before* every paid call using a
  deliberately pessimistic token estimate. When it trips, the **whole run is
  aborted** (pending work cancelled, run flagged, never usable as a baseline)
  rather than recording every remaining case as a separate error. Default
  $5/run (`HARNESS_MAX_COST_USD`, `--max-cost-usd`).
- **Pricing is mandatory for paid models** -- a model missing from
  [`pricing.py`](harness/core/pricing.py) is refused (unless `--allow-unpriced`),
  because an unpriced call would silently count as $0 and defeat the cap. Local
  models are free and exempt.
- **`ResponseCache`** -- SQLite, keyed by model, sampling settings, sample index,
  prompt and tools. Re-running after a crash, or regenerating a report, never
  re-spends. Changing temperature never serves a stale answer.
- **Timeouts and retries** -- every client has a timeout (longer for local
  models); rate limits, connection errors, timeouts and 5xx are retried with
  jittered exponential backoff; models that reject `temperature` or
  `max_tokens` are adapted to once and remembered.
- **`--smoke N`** -- pinned cases plus the first N others, for cheap iteration.
  Smoke runs are flagged partial, can't become baselines, and the gate refuses them.

Rough cost of a full hosted run with the all-`gpt-6-luna` setup (25 RAG + 10
agent cases, task + judge calls): **a few cents**. A local-only run costs nothing.

## CLI reference

| Command | Purpose |
|---|---|
| `harness doctor` | Check packages, config, datasets, keys, baselines, local models |
| `harness run` | Execute the golden set, score it, save to SQLite, write `.md` + `.json` reports |
| `harness baseline set` / `show` | Bless a full run as the committed baseline / inspect it |
| `harness diff` | Regressions vs. the baseline for a (task, model) |
| `harness gate` | Pass/fail with an exit code, for CI |
| `harness compare` | Side-by-side comparison of models on one task |
| `harness calibrate` | Test the LLM judge against `golden/calibration_labels.jsonl` |
| `harness history` | Run history: score, cost, errors, kind (full/partial/aborted), git SHA |

Exit codes are stable: **0** success, **1** checked and failed (gate failed,
regressions, errored samples, weak judge), **2** nothing to act on (no run / no
baseline) or run aborted, **3** fatal (bad config, missing key, invalid dataset).

## Configuration

All via environment / `.env` (see [`.env.example`](.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | -- | Only for those providers |
| `HARNESS_OLLAMA_URL` | `http://localhost:11434/v1` | Local model endpoint |
| `HARNESS_MAX_COST_USD` | `5.00` | Per-run spend cap |
| `HARNESS_MAX_WORKERS` | `4` | Concurrent samples |
| `HARNESS_TEMPERATURE` | `0` | Task-model temperature (judge is always 0) |
| `HARNESS_MAX_OUTPUT_TOKENS` | `1024` | Raise for reasoning models |
| `HARNESS_JUDGE_MODEL` | per provider | Judge model |
| `HARNESS_BASELINES_DIR` / `_REPORTS_DIR` / `_RUNS_DB` / `_CACHE_DB` | `baselines` / `reports` / `runs.db` / `.harness_cache.db` | State locations |

## CI

Two workflows, deliberately separate:

- **[`ci.yml`](.github/workflows/ci.yml)** -- every push/PR, free, `--provider
  mock`: lint, the full test suite, environment check, smoke runs of both tasks.
  Proves the harness's own code is correct; says nothing about answer quality.
- **[`eval-gate.yml`](.github/workflows/eval-gate.yml)** -- the one that costs
  money. Nightly and on demand, never on push (an eval gate wired to every
  commit, including from forks, is an unbounded spend trigger). It skips with a
  warning until the `OPENAI_API_KEY` secret exists, gates each task against the
  committed baseline with `--require-baseline`, publishes the report as the job
  summary, and -- when no baseline is committed yet -- uploads a candidate
  baseline as an artifact for you to review and commit.

The workflow files are validated as YAML but have not been run on GitHub from
this repository yet; the first real run needs the secret and a baseline.

## Testing

```bash
pytest -q
ruff check harness tests
```

170+ tests, all offline and free. They cover dataset validation; every scorer
(including a judge handed malformed output and prompt-injection text); the
budget guard; cache scoping by settings and sample; provider behavior against a
fake client (parameter adaptation, retries, missing usage, tool-call parsing,
Ollama preflight, pricing refusal); the runner's repeat sampling, error
isolation and budget abort; the store including migration of an old database;
baseline round-trips; diffing; **every gate rule**; the calculator's refusal to
evaluate anything but arithmetic; report rendering; and end-to-end CLI runs of the
whole run -> bless -> gate -> regress -> fix cycle with real argument parsing,
SQLite and files.

## Known limitations

Honest scope notes, not hidden gaps:

- **Golden sets are seeds, not finished suites.** 25 + 10 cases exercise every
  scorer and prove the methodology; a production gate wants 100+ per task, grown
  from cases that actually broke. With n=25 a pass rate carries a wide interval
  (the reports print it), so small differences between models are not evidence.
- **The OpenAI path is tested against a fake client, not the live API.** Local
  (Ollama) runs were executed for real; `gpt-6-luna` has not been, pending an API
  key. Run `calibrate --provider openai` before trusting Luna to judge.
- **A baseline is only as meaningful as the run behind it.** `baseline set`
  refuses partial, aborted and errored runs, but cannot tell whether the model
  was good, only that it was complete.
- **Judge calibration is one annotator, 31 cases**, written by the same author as
  the judge prompt -- see [CALIBRATION.md](CALIBRATION.md) for what that does and
  doesn't establish.
- **`MockProvider`'s tool-choice heuristic is deliberately crude.** It exists to
  exercise code paths for $0; its scores measure the heuristic, not agent quality.
- **The agent has a hard step ceiling (`MAX_STEPS = 4`).** A case needing more
  turns reports a clean error rather than hanging.
- **SQLite, single node.** Fine for one contributor or a small team's CI; a
  concurrent-writer service would need a real database.
- **ChromaDB's ephemeral client is not proven thread-safe**, so the RAG adapter
  serializes all vector-store access behind one lock (found by running under
  concurrency, not by inspection).
- **No async I/O** -- a bounded thread pool, sufficient for network-bound calls at
  this scale and far simpler to reason about than an asyncio rewrite.

## License

This project: MIT (see [LICENSE](LICENSE)). Vendored target app: MIT
(Raudaschl/rag-fusion), unmodified, attributed above.
