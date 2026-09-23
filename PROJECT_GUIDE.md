# Project Guide

## What this project does

`eval-gate` is an evaluation harness for LLM-powered applications -- the
equivalent of a regression-test suite for a system whose output isn't
deterministic, so you can't just assert `output == expected_string`.

It runs a versioned set of test cases ("golden sets") against a target
LLM application, grades each output with multiple scoring methods
(exact/substring checks, retrieval accuracy, semantic similarity, and an
LLM-as-judge that is itself calibrated against hand-labeled ground truth
rather than trusted blindly), compares the result against the previous
run to catch regressions a raw average score would hide, and can block a
CI pipeline (`harness gate`) if quality drops or a known-important case
starts failing.

It evaluates two different kinds of application through the same core:

- **`rag_fusion_qa`** -- a fixed retrieval-augmented-generation pipeline,
  wrapping a vendored third-party app
  ([Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion))
- **`tool_agent_qa`** -- a genuine agent: the model decides, turn by
  turn, whether to call a tool, which one, and when it has enough to
  answer, rather than following a scripted sequence

## Getting an Anthropic API key

1. Go to **[console.anthropic.com](https://console.anthropic.com)** and
   sign up or log in. This is separate from a claude.ai chat subscription
   -- API access is billed separately, pay-as-you-go per token.
2. Add billing details and some credit (a few dollars is more than
   enough for this project -- see the cost estimates in README.md).
3. Go to **Settings -> API Keys** and create a new key.
4. In this project: copy `.env.example` to `.env`, paste the key in as
   `ANTHROPIC_API_KEY=sk-ant-...`. `.env` is gitignored -- it will never
   be committed or pushed.

Everything in this project defaults to `--provider mock` (a free, fake
model), so nothing requires this key until you deliberately pass
`--provider anthropic`.

## Models used

| Role | Model | Notes |
|---|---|---|
| Task model (default) | `claude-haiku-4-5-20251001` | Cheapest viable tier; what's actually being evaluated |
| Task model (optional) | `claude-sonnet-5` | Only if you want to compare a stronger model against Haiku on identical inputs |
| Judge (default) | `claude-haiku-4-5-20251001` | Same cheap tier, by default -- see CALIBRATION.md for why this is checked, not assumed |
| Judge (optional upgrade) | `claude-sonnet-5` | Set via `HARNESS_JUDGE_MODEL` in `.env` if calibration shows Haiku isn't reliable enough |
| Embeddings | `all-MiniLM-L6-v2` | Runs locally via `sentence-transformers` -- free, no API key needed, downloaded once from Hugging Face on first use |

## End-to-end flow

```
python -m harness.cli run --task rag_fusion_qa --provider anthropic --smoke 10
```

1. **`harness/cli.py: main()`** parses args, sets up logging, dispatches
   to `cmd_run()`.
2. **`cmd_run()`** loads config (`harness/config.py: load_config()` --
   reads `.env`, validates env vars, fails fast with a clear message if
   something's missing), builds a `ResponseCache`
   (`harness/core/cache.py`) and a `BudgetGuard`
   (`harness/core/budget.py`), and gets a provider for the task model and
   a separate one for the judge via `get_provider()`
   (`harness/core/providers.py`) -- this wraps the raw Anthropic client in
   budget-checking and caching layers automatically.
3. The golden dataset is loaded with `load_jsonl()`
   (`harness/core/dataset.py`), optionally trimmed to a `--smoke N`
   subset (pinned regression cases always included).
4. **`harness/tasks.py: get_task()`** looks up the `TaskSpec` for the
   requested task name and builds that task's adapter and scorer list.
5. **`harness/core/runner.py: Runner.run()`** executes every case
   concurrently (bounded thread pool), calling `adapter.run(case,
   provider)` for each one and catching any exception per-case so one bad
   case can't kill the whole run:
   - For `rag_fusion_qa`: **`harness/adapters/rag_fusion.py:
     RagFusionAdapter.run()`** -- expands the query via
     `provider.generate()`, retrieves and fuses results using
     `vector_search()` / `reciprocal_rank_fusion()` imported *directly*
     from the vendored `targets/rag-fusion/main.py` (unmodified), then
     synthesizes a final answer via another `provider.generate()` call.
   - For `tool_agent_qa`: **`harness/adapters/tool_agent.py:
     ToolAgentAdapter.run()`** -- drives a loop calling
     `provider.generate_messages()` with the running conversation; if the
     model calls a tool, `harness/adapters/tools.py: execute_tool()` runs
     it locally (`calculator` or `kb_lookup`) and the result is fed back
     in; this repeats (capped at `MAX_STEPS = 4`) until the model gives a
     final text answer instead of calling another tool.
6. Each case's output (`RunTrace`) is scored by every scorer in
   `harness/scorers/` for that task -- e.g. `contains.py`,
   `retrieval.py`, `tool_usage.py`, `embedding_sim.py`, and
   `llm_judge.py` (which itself calls the judge provider and parses its
   JSON verdict).
7. The full `RunSummary` is saved to SQLite via
   **`harness/core/store.py: RunStore.save()`**.
8. **`harness/core/diff.py: diff_runs()`** compares this run against the
   most recent previous run for the same (task, model) pair, classifying
   every case as a regression, an improvement, or unchanged.
9. **`harness/core/report.py: write_report()`** renders everything to a
   Markdown file under `reports/`.
10. **`harness gate`** (`cmd_gate()`) re-reads the latest run from
    storage and exits non-zero if the mean score is below threshold, a
    pinned case is failing, or any regression was detected -- this is the
    exit code a CI workflow checks to block a merge or deploy.
11. **`harness calibrate`** (`cmd_calibrate()`) is a separate, independent
    check: it runs the judge scorer against
    `golden/calibration_labels.jsonl` (fixed question/candidate-answer/
    human-label triples) and reports how well the judge's scores track
    human judgment -- see `CALIBRATION.md`.

## Where each concern lives

| Concern | File |
|---|---|
| Config / env var validation | `harness/config.py` |
| Cost tracking + per-token pricing | `harness/core/pricing.py` |
| Hard spend cap | `harness/core/budget.py` |
| Response caching (so reruns are free) | `harness/core/cache.py` |
| Model calls (Anthropic + mock), retries | `harness/core/providers.py` |
| Golden dataset loading/validation | `harness/core/dataset.py` |
| Concurrent execution + error isolation | `harness/core/runner.py` |
| Run history (SQLite) | `harness/core/store.py` |
| Regression detection | `harness/core/diff.py` |
| Markdown reports | `harness/core/report.py` |
| Shared embedding model | `harness/core/embeddings.py` |
| Document corpus + semantic search | `harness/corpus.py`, `harness/core/corpus_search.py` |
| Task -> adapter/dataset/scorer registry | `harness/tasks.py` |
| RAG-fusion adapter | `harness/adapters/rag_fusion.py` |
| Tool-calling agent adapter | `harness/adapters/tool_agent.py` |
| Agent's tools (calculator, kb_lookup) | `harness/adapters/tools.py` |
| Scorers | `harness/scorers/*.py` |
| CLI | `harness/cli.py` |
