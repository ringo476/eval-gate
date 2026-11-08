# Project Flow — Complete Reference

This document has three parts:

1. **What this project does** — one paragraph, no jargon.
2. **Execution traces** — for every CLI command, the exact sequence of
   function calls from the moment you press Enter to the moment it exits,
   in order, naming every function involved.
3. **Function reference** — every function and method in the codebase,
   file by file, in the order they appear in that file: what it takes,
   what it does, what it returns, anything non-obvious about it.

If you only read one section, read part 2 for the command you care about
(`run` is the one that matters most) — it tells you, in order, everything
that happens.

---

## Part 1: What this project does

This is an **evaluation harness** for LLM-powered applications — the
regression-test-suite equivalent for systems whose output isn't
deterministic (you can't `assert output == "expected string"` on a model
call). It runs a fixed, versioned set of test questions ("golden cases")
against a target app, grades every answer with several different scoring
methods, compares the result to a **committed baseline** to catch
regressions, and can block a CI pipeline if quality drops.

It evaluates **two** target apps through the same core machinery:

- **`rag_fusion_qa`** — a fixed pipeline (expand query → retrieve →
  fuse → synthesize answer) wrapping a vendored third-party project,
  [Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion).
- **`tool_agent_qa`** — a genuine agent (built for this project) with a
  calculator and a knowledge-base lookup tool, where the model itself
  decides which tool to call, in what order, and when to stop.

It runs them against any of four **providers**: `openai`, `anthropic`,
`ollama` (a local model, free), or `mock` (a fake, for testing the harness).
The model under test and the judge that grades it can be on different
providers.

Three ideas distinguish it from a script that prints a score:

- **Baselines are committed files**, not database rows, so regression
  detection works on a fresh CI runner with no history.
- **Cases can be sampled repeatedly** (`--repeats N`), so a flaky case is
  told apart from a broken one.
- **The judge is itself tested** (`calibrate`) before its scores are trusted,
  and the gate refuses runs that are partial, aborted, errored, or stale.

---

## Part 2: Execution traces

### `python -m harness.cli run --task rag_fusion_qa --provider ollama --model llama3.1:8b --judge-model gpt-oss:20b --repeats 3 --temperature 0.5`

This is the main command. Here is *everything* that happens, in order.

1. **`harness/cli.py: main(argv)`** runs first (the `if __name__ == "__main__"`
   guard calls it; `python -m harness` and the `harness` console script call
   it too). It reconfigures `sys.stdout`/`sys.stderr` to UTF-8, calls
   **`build_parser()`**, `parser.parse_args(argv)`, then
   **`setup_logging(args.verbose)`** (`harness/logging_setup.py`), then calls
   `args.func(args)` — **`cmd_run`** for `run` — inside a `try` that maps
   exceptions to exit codes (see the exit-code table in the module
   docstring: `KeyboardInterrupt` → 130, expected errors → 3).

2. **`cmd_run(args)`** (`harness/cli.py`):
   1. **`load_config()`** (`harness/config.py`) — reads `.env` (via
      `load_dotenv()` at import time) and every `HARNESS_*` / `*_API_KEY`
      env var, validates numeric ones, returns a frozen `Config`.
   2. Validates `--repeats >= 1`. Calls **`_task_settings(config, args)`** →
      a `GenerationSettings` (temperature, max output tokens). Logs a warning
      if `repeats > 1` and temperature is 0 (the samples would be identical).
   3. **`_resolve_judge(args, config, provider)`** → `(judge_provider,
      judge_model)`: `--judge-provider` or the task provider; `--judge-model`
      or `HARNESS_JUDGE_MODEL` or a per-provider default.
   4. Computes `self_judged` (judge == task model on a real provider). If so,
      logs a warning — or, with `--strict-judge`, raises `RuntimeError`
      *before* anything is spent.
   5. Constructs a **`ResponseCache`** (unless `--no-cache`) and a
      **`BudgetGuard`** (`--max-cost-usd` or `HARNESS_MAX_COST_USD`).
   6. **`_make_provider(...)`** *twice* — task model, then judge. Each call:
      `config.require_api_key(provider)` (raises an actionable error for a
      missing key; `ollama`/`mock` need none), for `ollama`
      **`check_ollama(model, url)`** (fails fast if the server is down or the
      model isn't pulled), then **`get_provider(...)`**, which builds the
      concrete provider, refuses an unpriced paid model unless
      `--allow-unpriced`, and wraps it `BudgetEnforcingProvider` (paid
      providers only) then `CachingProvider`. The judge gets
      `_judge_settings(...)`: temperature always 0, output cap at least 2048
      (reasoning models need room to think).
   7. **`get_task(args.task)`** → `TaskSpec` (dataset path + adapter factory +
      scorer factory).
   8. **`load_jsonl(path, expected_task=args.task)`** — strictly validates the
      golden set (types, duplicate ids, `human_label` range, and that every
      row's `task` matches). **`file_sha256(path)`** fingerprints the file.
   9. With `--smoke N`, **`filter_smoke(cases, n)`** keeps every pinned case
      plus the first N others.
   10. **`task_spec.adapter_factory()`** and
       **`task_spec.scorers_factory(judge_provider)`** build the adapter and
       the scorer list.
   11. **`Runner(adapter, scorers, provider, max_workers, repeats)`**, then
       **`runner.run(cases)`** — the nested trace below. All model calls,
       scoring and wall-clock time happen here.
   12. **`collect_meta(...)`** (`harness/core/runmeta.py`) attaches provenance
       to the summary: git sha / branch / dirty flag, harness version, provider,
       judge, `self_judged`, temperature, repeats, dataset path + sha256 +
       size, and `partial` (true if the dataset was subset by `--smoke`).
   13. **`RunStore(config.runs_db_path)`** (creates/migrates the schema),
       **`store.save(summary)`** — one `runs` row (incl. `meta_json`,
       `is_partial`, `is_aborted`, `repeats`) and one `case_results` row per
       *case* (pass rate across samples, representative sample's output and
       scores, a compact per-sample JSON), in one transaction.
   14. **`diff_runs(store, task, model, run_id, baselines_dir=...,
       baseline_path=args.baseline, threshold=...)`** → `resolve_baseline`
       (committed file first, else the previous *full* run) → `diff_cases`.
   15. **`write_report(summary, diff, out_dir)`** writes `<run_id>.md` and
       `<run_id>.json`; **`render_run_report`** is also printed to stdout.
   16. Prints the budget spent. Returns **2** if the run was aborted, **1** if
       any sample errored, else **0**. (`run` does not apply a quality bar —
       that's `gate`.) `finally:` closes the store and cache.

3. **Inside `runner.run(cases)`** (`harness/core/runner.py`):
   1. Generates a `run_id` via `new_run_id()`.
   2. Submits one `Runner._run_one(case, sample_idx)` per (case, sample) to a
      `ThreadPoolExecutor(max_workers)`.
   3. An adapter exception that escapes (say, a case missing its `query`) becomes
      an error trace for that one case. It never takes down the run.
      `_run_one` obtains **`self._provider_for(sample_idx)`** →
      `provider.scoped(sample_idx)`. Only `CachingProvider` does anything with
      it: it returns a view whose cache key includes the sample index, so
      sample 2 is a fresh draw rather than a replay of sample 0.
   4. `_run_one` calls **`adapter.run(case, scoped_provider)`** — the target
      app runs (traces 4 and 5 below).
   5. If `trace.error` is set, returns a `CaseResult` with no scores — no
      scorer runs on a failed sample.
   6. Otherwise calls **`scorer.score(case, trace)`** for every scorer the case
      didn't opt out of with `expected.skip_scorers`, catching any exception
      per-scorer (a broken scorer becomes a failing `ScoreResult`), **except
      `BudgetExceededError`**, which propagates. If the trace has a `failure_note`
      (the agent ran out of steps) a failing `completion` result is added: that is
      the model failing the task, so it is a failed check and not an error.
   7. Back in `run()`, completed futures are collected. The run is **aborted**
      (every pending future cancelled, `abort_reason` recorded, work already in
      flight still collected) on `BudgetExceededError`, or when
      `--max-consecutive-errors` samples in a row have errored, because a bad key
      or a dead server would fail every remaining case the same way. Ctrl-C also
      cancels queued work, so it stops spending instead of finishing the queue.
   8. Results are sorted into `(dataset order, sample_idx)` — thread
      completion order isn't deterministic — and returned as a `RunSummary`
      (`aborted=True` if the budget tripped).

4. **Inside `RagFusionAdapter.run(case, provider)`**
   (`harness/adapters/rag_fusion.py`), for `rag_fusion_qa`:
   1. **`get_target_module()`** — imports the vendored
      `targets/rag-fusion/main.py` once (cached in `_target_module`).
   2. **`self._get_collection()`** — double-checked locking around building
      the in-memory ChromaDB collection (concurrent construction crashed in
      testing).
   3. **`provider.generate(...)`** with `QUERY_EXPANSION_SYSTEM` — model call 1.
   4. **`_clean_query_lines(...)`** strips numbering/bullets, and the list is cut
      to `MAX_EXPANDED_QUERIES` (4) so a chatty reply can't fan out into dozens of
      searches.
   5. Under `self._collection_lock`: **`target.vector_search(...)`** per query,
      **`target.reciprocal_rank_fusion(...)`** (both vendored, unmodified),
      then `collection.get(ids=...)` for the top-`k` fused documents, restoring
      fused order. The whole ranking goes in `RunTrace.retrieval_ranking` for the
      retrieval metrics.
   6. **`provider.generate(...)`** with `SYNTHESIS_SYSTEM` — model call 2.
   7. **`_leg_cost(...)`** ×2 (0 for cache hits).
   8. Returns a `RunTrace`. Any exception becomes an error trace — **except
      `BudgetExceededError`, which is re-raised** for the runner to abort on.

5. **Inside `ToolAgentAdapter.run(case, provider)`**
   (`harness/adapters/tool_agent.py`), for `tool_agent_qa` — a loop:
   1. `messages = [{"role": "user", "content": query}]`.
   2. Up to `MAX_STEPS` (4) times: **`provider.generate_messages(messages,
      system=AGENT_SYSTEM, tools=TOOL_SCHEMAS)`**; if the reply has no tool
      calls it's the final answer → return a `RunTrace`. Otherwise
      **`provider.build_assistant_message(response)`** is appended, each tool
      call is run via **`execute_tool(...)`**, and
      **`provider.build_tool_result_messages(executed)`** extends `messages`.
   3. Exhausting `MAX_STEPS` returns a trace with `failure_note` set.
   4. Same exception policy as the RAG adapter.

6. **Inside `execute_tool(name, tool_input)`** (`harness/adapters/tools.py`):
   - `"calculator"` → **`calculator(expression)`**: rejects non-strings and
     anything over `MAX_EXPRESSION_CHARS` (200), then `ast.parse` +
     **`_safe_eval_node`** (only numbers, `+ - * /`, unary `+ -`;
     `RecursionError` from deep nesting becomes a clean `error:` string).
   - `"kb_lookup"` → **`kb_lookup(topic)`** → `search_corpus` → `most_similar`
     → `get_embedding_model()` + `cosine_similarity(...)`.

7. **Scoring** — what each scorer does with a trace is in Part 3
   (`harness/scorers/*`). `LLMJudgeScorer.score` builds a delimited prompt via
   `build_judge_prompt`, calls the (temperature-0) judge, parses with
   `_extract_json`, and clamps the score to 0–10. `GroundednessScorer` does the same
   with the retrieved documents in place of a reference answer.

### `python -m harness.cli gate --task rag_fusion_qa --model gpt-6-luna --min-score 0.75 --require-baseline`

1. `main()` → `cmd_gate(args)`.
2. `load_config()`, `RunStore`, then the run to check: `--run-id`, else
   **`store.latest_run_id(task, model)`**. Exit **2** if none exists.
   **`store.get_run(run_id)`** loads its row (flags + meta); exit 2 if the id is
   unknown.
3. `load_jsonl(dataset_path, expected_task=...)` → the pinned case ids;
   **`file_sha256`** of the dataset *now*.
4. **`store.case_results_for_run(run_id)`** → per-case `passed` / `pass_rate` /
   `score` / `n_samples`.
5. **`diff_runs(...)`** — committed baseline or previous full run.
6. **`evaluate_gate(...)`** (`harness/core/gate.py`) applies the policy (see
   Part 3) and returns a `GateVerdict` with `failures` and `warnings`.
7. Prints `WARN:`/`FAIL:` lines or one `PASS:` line. Exit **0** pass,
   **2** if the run was aborted, **1** otherwise.

### `python -m harness.cli baseline set --task rag_fusion_qa --model gpt-6-luna`

1. `main()` → `cmd_baseline_set(args)`.
2. The run to bless: `--run-id`, else **`store.latest_run_id(..., full_only=True)`**
   (never a smoke or aborted run). Exit 2 if none.
3. **`baseline_from_store(store, run_id, allow_errors=args.force)`** refuses an
   aborted run, a partial run, and (without `--force`) one with errored
   samples; otherwise builds a `Baseline` from the run row and its case results.
4. **`write_baseline(baseline, path)`** — `baselines/<task>__<model>.json`,
   sorted keys + trailing newline so the git diff is clean. If a baseline
   already existed, prints what it replaced.
5. Prints a reminder to review and commit the file.

### `python -m harness.cli diff --task rag_fusion_qa`

`main()` → `cmd_diff(args)` → `RunStore`, `latest_run_id`, `diff_runs`; prints
the baseline kind (committed / previous local run), regressions, flakier cases,
improvements. Exit 1 if any regression, 2 if no run exists.

### `python -m harness.cli compare --task rag_fusion_qa --models gpt-6-luna llama3.1:8b`

`main()` → `cmd_compare(args)`: for each model, **`store.latest_run_id(...,
full_only=True)`**, `get_run`, `case_results_for_run`. Exit 2 listing any model
with no full run. Otherwise **`render_comparison(task, runs, results)`** prints a
table (pass rate with 95% interval, mean score, cost, errors, judge, run id),
retrieval metrics, an exact paired test for each pair of models
(**`paired_disagreements`** + **`mcnemar_exact`**), and the cases where models
disagree; `--out` also writes it to a file. It warns if the runs used different judges.

### `python -m harness.cli dashboard`

`main()` → `cmd_dashboard(args)` → **`write_dashboard(store, out, baselines_dir)`**:
**`collect_dashboard_data`** takes each model's latest full run, its history on the
same dataset, and its baseline diff; **`render_dashboard`** writes one self-contained
HTML page. All run text passes through `html.escape`.

### `python -m harness.cli calibrate --provider ollama --judge-model gpt-oss:20b --repeats 3`

Independent of the target apps — no adapter, retrieval or tools. It tests an LLM
grader alone. `--kind judge` (default) tests correctness grading against
`golden/calibration_labels.jsonl`; `--kind groundedness` tests the hallucination check
against `golden/groundedness_labels.jsonl`.

1. `main()` → `cmd_calibrate(args)`.
2. `load_config()`, judge model (`--judge-model` / `HARNESS_JUDGE_MODEL` /
   default), `GenerationSettings(--temperature, max(--max-output-tokens, 2048))`,
   cache, budget, **`_make_provider(...)`** once.
3. `load_jsonl("golden/calibration_labels.jsonl", expected_task="judge_calibration")`.
   Each row's `candidate_answer` is a fixed, hand-authored answer.
4. For each case, `--repeats` times: a fresh
   **`LLMJudgeScorer(judge_provider.scoped(r))`** scores a synthetic `RunTrace`
   holding the candidate. A reply with no usable score is recorded as
   **invalid**, *not* as a 0 (a judge that answers in prose would otherwise look
   like it "scored the answer 0").
5. Computes MAE, Pearson correlation (`stats.pearson`), agreement within ±0.2,
   **false-accept rate** (clearly-bad answers, label ≤ 0.2, that the judge
   passes at ≥ 0.7 — the dangerous error for a regression gate), false-reject
   rate, self-inconsistency across repeats, and mean error by tag.
6. Prints the table and metrics, writes `reports/calibration-<model>-<ts>.json`.
7. Exit 1 if: > 10% invalid replies, MAE > 0.25, correlation undefined or
   < 0.5, or false-accept rate > 20%.

### `python -m harness.cli history --task rag_fusion_qa --model gpt-6-luna`

`main()` → `cmd_history(args)` → **`store.run_history(...)`**; prints run id,
time, mean score, pass rate, cost, errors, repeats, kind
(full / partial / aborted), and short git sha.

### `python -m harness.cli doctor`

`main()` → `cmd_doctor(args)`: checks the Python version, every required
package, config validity, that the vendored target exists, that each task's
golden set loads (counts cases and pinned cases), which baselines exist, which
API keys are set (never prints values), and whether Ollama is reachable and
which models it has. Exit 3 if a required check fails.

### `python scripts/demo_regression.py [--change careless|careful]`

Overwrites `rag_fusion.SYNTHESIS_SYSTEM` in memory, points the run history and reports at
a temporary directory, then calls `cli.main` for `run` and `gate --require-baseline`. No
files in the repo change, and the shared response cache means answers that didn't change
are not recomputed.

---

## Part 3: Complete function reference

Organized file by file, in the order each function/method appears in
that file.

### `harness/__init__.py`, `harness/__main__.py`

- **`__version__`** — the single source of the version (also read by
  `pyproject.toml`).
- `__main__.py` calls `cli.main()` so `python -m harness` works.

### `harness/config.py`

- **`PROVIDERS`** — `("openai", "anthropic", "ollama", "mock")`.
- **`Config`** (frozen dataclass) — both API keys (either may be `None`), cost
  cap, worker count, optional judge model, DB/baseline/report paths, Ollama
  URL, default temperature, default max output tokens.
- **`Config.require_api_key(self, provider_name)`** — returns the key for
  `openai`/`anthropic`, raising an actionable `RuntimeError` if missing;
  returns `None` for `ollama`/`mock` (no key needed); `ValueError` for anything
  else.
- **`_env_number(name, default, cast, kind)`** — parse an env var or raise
  `RuntimeError` naming the variable.
- **`load_config()`** — the only function that reads `os.environ`. Validates
  ranges (workers ≥ 1, cost > 0, temperature 0–2, output tokens ≥ 16).

### `harness/logging_setup.py`

- **`setup_logging(verbose=False)`** — one stderr handler on the `"harness"`
  logger tree; DEBUG if `verbose`, else INFO. Prompts/responses only at DEBUG.

### `harness/core/models.py`

Pure dataclasses plus computed properties:

- **`EvalCase`** — one golden row. `.is_pinned` — `"pinned"` in `.tags`.
- **`ProviderResponse`** — one model call's result; `finish_reason == "length"`
  means the output was truncated.
- **`RunTrace`** — what an adapter returns for one sample. `error` means the run
  broke (network, crash); `failure_note` means it finished but the app misbehaved in
  a way scorers can't see; `retrieval_ranking` is every retrieved doc id, best first.
- **`ScoreResult`** — one scorer's verdict (see below).
- **`ScoreResult`** also carries `applicable` (false when the case had nothing for
  this scorer to check, so it passes but stays out of averages) and `cost_usd`
  (what an LLM grader spent).
- **`CaseResult`** — **one sample** of one case (`sample_idx`).
  `.aggregate_score` (mean of *applicable* scores; 0 if errored), `.passed` (all
  scorers passed and no error), `.scoring_cost_usd`, `.total_cost_usd`.
- **`CaseAggregate`** — all samples of a case. `.n_samples`, `.n_passed`,
  `.pass_rate`, `.passed` (**strict majority** of samples, so one sample behaves
  like before), `.status` (`pass` / `fail` / `flaky`), `.mean_score`,
  `.error_count`, `.representative` (first failing sample, else the first),
  `.total_cost_usd`, `.mean_latency_ms`.
- **`RunSummary`** — one run. `.cases` (group `case_results` into
  `CaseAggregate`s), `.is_partial` (from `meta["partial"]`), `.repeats`,
  `.total_cost_usd`, `.total_latency_ms`, `.error_count` (errored *samples*),
  `.pass_rate` (fraction of *cases* passed), `.mean_score`, `.flaky_cases`.
  Also `meta`, `aborted`, `abort_reason`.

### `harness/core/stats.py`

- **`wilson_interval(successes, total, z=1.96)`** — 95% Wilson score interval;
  well-behaved at 0% / 100% and small n.
- **`intervals_overlap(a, b)`**.
- **`mcnemar_exact(only_a, only_b)`** — exact paired test for two models on the same
  cases. Only disagreements carry information; if the models were equally good each
  would be a coin flip.
- **`percentile(values, pct)`** — linear interpolation; 0 for empty.
- **`pearson(xs, ys)`** — correlation, or `None` when undefined (n < 2 or zero
  variance) instead of NaN.

### `harness/core/retrieval_metrics.py`

- **`recall_at_k`**, **`precision_at_k`**, **`reciprocal_rank`**, **`ndcg_at_k`** —
  standard definitions; `ranking` is best-first, `relevant` is the human-marked set.
- **`retrieval_metrics(ranking, relevant, ks=(1, 3, 5))`** — all of them for one query.
- **`aggregate_retrieval(case_results)`** — means over samples with both a ranking and
  a known right answer; `None` if there are none (e.g. the agent task).

### `harness/core/dashboard.py`

- **`collect_dashboard_data(store, baselines_dir, include_mock=False)`** — per task, each
  model's latest full run, same-dataset history and baseline diff. Mock runs are hidden
  by default because their scores mean nothing.
- **`render_dashboard(tasks, generated_at=None)`** — the HTML page: per-model table with
  pass-rate bars and trend, retrieval table, a case-by-model matrix (status always has an
  icon and text, not just color), and "what failed and why".
- **`write_dashboard(store, out_path, baselines_dir, include_mock=False)`**.

### `harness/core/runmeta.py`

- **`file_sha256(path)`** — SHA-256 of a text file with line endings normalized, so the
  same dataset hashes the same on Windows and in CI.
- **`_git(*args)`** — run git, `None` on any failure (not a repo, git missing).
- **`git_info()`** — `git_sha`, `git_branch`, `git_dirty`.
- **`collect_meta(**run_settings)`** — harness version + Python version + git
  info + the settings passed in.

### `harness/core/pricing.py`

- **`PRICING_PER_MTOK`** — per-model USD per million tokens.
- **`is_priced(model)`** — whether a model has an entry.
- **`estimate_cost_usd(model, input_tokens, output_tokens)`** — $0 for an
  unlisted model. Safe only for local models; paid providers refuse unpriced
  models in `get_provider`.

### `harness/core/budget.py`

- **`BudgetExceededError`**.
- **`BudgetGuard.__init__(max_cost_usd)`** — cap, `_spent = 0`, `_reserved = 0`, a lock.
- **`BudgetGuard.spent`**, **`.reserved`** — thread-safe reads.
- **`BudgetGuard.check_and_reserve(estimated_cost)`** — raises if
  `spent + reserved + estimate > cap`, otherwise holds the estimate. A real
  reservation: calls run in parallel, and a check on money already spent would let
  every in-flight worker pass at once and overshoot.
- **`BudgetGuard.release(reservation)`** — the call failed; give it back.
- **`BudgetGuard.record(actual_cost, reservation=0.0)`** — swap the hold for the real cost.

### `harness/core/cache.py`

- **`_cache_key(model, system, prompt, tools, variant="")`** — SHA-256 of the
  sorted-JSON of all five. `variant` folds in sampling settings and the
  repeat-sample index.
- **`ResponseCache.__init__(db_path)`** — opens/creates the SQLite file.
- **`ResponseCache.get(model, system, prompt, tools, variant="")`** — `None` on
  a miss, else a `ProviderResponse` (unknown stored fields are ignored, so a
  cache written by another version still loads).
- **`ResponseCache.put(..., response, variant="")`** — `INSERT OR REPLACE`;
  keeps `.raw` whenever it is JSON-safe.
- **`ResponseCache.close()`**.

### `harness/core/embeddings.py`

- **`get_embedding_model()`** — lazy, process-wide `all-MiniLM-L6-v2`.
- **`cosine_similarity(a, b)`**.
- **`most_similar(query, candidates, top_k=1)`** — top-k `(key, text,
  similarity)`.

### `harness/core/corpus_search.py`

- **`search_corpus(query, top_k=1)`** — `most_similar` over `CORPUS`.

### `harness/core/dataset.py`

- **`load_jsonl(path, expected_task=None)`** — reads line by line (skipping
  blanks and `#` comments); requires `id`/`task`/`input`/`expected`; checks
  types (`id` non-empty string, `input`/`expected` objects, `tags` list of
  strings, `human_label` number in [0, 1]); rejects duplicate ids and, if
  `expected_task` is given, any row of another task. Every error carries
  `path:line`. Raises on an empty file.
- **`filter_smoke(cases, n=10)`** — `(pinned + rest)[:max(n, len(pinned))]`.

### `harness/core/providers.py`

- **`GenerationSettings`** (frozen) — `temperature`, `max_output_tokens`,
  `timeout_s`. `.fingerprint()` is part of the cache key.
- **`ModelProvider`** (ABC) — `generate`, `generate_messages`,
  `build_assistant_message`, `build_tool_result_messages`; plus
  **`scoped(sample_idx)`** (default: return `self`).
- **`_NetworkProvider._with_retry(fn)`** — tenacity retry (exponential backoff
  with jitter, 5 attempts) on `self._retryable`, logging each retry.
- **`AnthropicProvider`** — `__init__` (lazy import, client with timeout and
  `max_retries=0` so retries are only ours), `_convert_tools`, `generate`,
  `generate_messages`, `_create` (retries once without `temperature` if the
  model rejects it), `_call` (parses text and `tool_use` blocks into both a flat
  `tool_calls` list and replayable `raw`), `build_assistant_message`,
  `build_tool_result_messages`.
- **`OpenAIProvider`** — same shape for the chat-completions format. Retries
  rate limits, connection errors/timeouts and 5xx. **`_adapt_kwargs`** adapts
  once, and remembers, when a model rejects `temperature` or the token
  parameter (`max_completion_tokens` ↔ `max_tokens`), or refuses function tools until
  `reasoning_effort` is `none`. **`_create`** applies it.
  **`_call`** parses tool calls (malformed JSON arguments become `{}`),
  estimates token usage if the server omits it, and warns on
  `finish_reason == "length"`. `build_tool_result_messages` returns one
  `role: "tool"` message per call (Anthropic combines them).
- **`OllamaProvider`** — `OpenAIProvider` pointed at Ollama's `/v1`, key
  `"ollama"`, token parameter `max_tokens`.
- **`_ollama_root(base_url)`**, **`list_ollama_models(base_url)`** (GET
  `/api/tags`; `RuntimeError` with the fix if unreachable),
  **`check_ollama(model, base_url)`** — preflight; names the `ollama pull`
  command if the model is missing.
- **`BudgetEnforcingProvider`** — `_worst_case_input_tokens` (chars ÷ 3, which
  over-estimates on purpose), `_enforce` (reserve → call → record), `generate`,
  `generate_messages`, pass-through message builders. Uses the wrapped
  provider's own `max_output_tokens` for the worst case.
- **`CachingProvider`** — `scoped(sample_idx)` (a view with a different salt),
  `_variant()` (settings fingerprint + salt), `generate`, `generate_messages`,
  `_cached_call`, pass-through builders.
- **`MockProvider`** — `$0`, deterministic; `generate_messages` uses a crude
  digits-plus-operator heuristic to fake a tool call.
- **`get_provider(provider_name, model, api_key, cache, budget=None,
  settings=None, base_url=None, allow_unpriced=False)`** — the single
  construction point. Mock → bare `MockProvider`. Ollama → `OllamaProvider`
  (timeout raised to ≥ 600 s), no budget wrapper (it's free), cache wrapper.
  Paid → requires a key and a pricing entry, then budget wrapper, then cache
  wrapper. Order matters: budget sits directly on the network call so a cache
  hit never touches it.

### `harness/core/runner.py`

- **`Runner.__init__(adapter, scorers, provider, max_workers=4, repeats=1)`**
  — rejects `repeats < 1`.
- **`Runner._provider_for(sample_idx)`** — `provider.scoped(i)` if it exists.
- **`Runner._run_one(case, sample_idx=0)`** — adapter, then scorers (see
  trace 3).
- **`Runner.run(cases)`** — fans out `cases × repeats`, aborts on
  `BudgetExceededError`, sorts, returns a `RunSummary`.

### `harness/core/store.py`

- **`RunStore.__init__(db_path)`** — connect, `_init_schema()`, `_migrate()`.
- **`_init_schema()`** — `CREATE TABLE IF NOT EXISTS` for `runs` and
  `case_results`, plus an index.
- **`_columns(table)`**, **`_migrate()`** — `PRAGMA user_version` gated; adds
  the v2 columns to an older database in place and backfills `pass_rate`.
- **`save(summary)`** — one transaction: a `runs` row and one `case_results`
  row per case.
- **`latest_run_id(task, model, before_run_id=None, full_only=False)`** —
  newest run; `before_run_id` means strictly *older* than that run;
  `full_only` skips partial and aborted runs.
- **`get_run(run_id)`** — the run row with parsed `meta` and boolean flags, or
  `None`.
- **`case_results_for_run(run_id)`** — `{case_id: {passed, score, pass_rate,
  n_samples}}`.
- **`run_history(task, model, limit=20)`**, **`close()`**.
- **`new_run_id()`**, **`now_iso()`**.

### `harness/core/baseline.py`

- **`BaselineError`**, **`Baseline`** (dataclass).
- **`slugify(text)`** — filesystem-safe (`llama3.1:8b` → `llama3.1_8b`).
- **`baseline_path(dir, task, model)`** — `<task>__<model>.json`.
- **`baseline_from_store(store, run_id, allow_errors=False)`** — builds a
  baseline from a stored run; raises `BaselineError` for an unknown, aborted or
  partial run, or one with errored samples (unless `allow_errors`). Keeps only
  reviewable provenance (git sha, dataset hash, judge, temperature, …).
- **`write_baseline(baseline, path)`** — deterministic JSON.
- **`load_baseline(path)`** — validates schema version and required fields.
- **`find_baseline(dir, task, model)`** — `None` if no file.

### `harness/core/diff.py`

- **`DiffResult`** — `regressions`, `improvements`, `degraded` (dropped but by
  less than the threshold), `unchanged_pass`/`unchanged_fail`, `new_cases`,
  `missing_cases` (in the baseline, absent now), `baseline_kind`
  (`blessed` / `previous_run` / `none`), `baseline_label`, baseline metrics,
  `threshold`.
- **`BaselineRef`** — a resolved baseline's per-case pass rates + provenance.
- **`resolve_baseline(store, task, model, current_run_id, baselines_dir,
  baseline_path)`** — explicit file (missing = error) → committed file → the
  previous full run in the DB → `None`.
- **`diff_cases(ref, current_run_id, current, threshold=0.5)`** — compares
  pass *rates*: a drop of ≥ threshold is a regression, a smaller drop is
  `degraded`, a rise of ≥ threshold is an improvement. One sample per case
  means a single flip is always a regression.
- **`diff_runs(store, task, model, current_run_id, *, baselines_dir=None,
  baseline_path=None, threshold=0.5)`** — resolve then diff.

### `harness/core/gate.py`

- **`GateVerdict`** — `ok`, `failures`, `warnings`, `mean_score`, `pass_rate`.
- **`evaluate_gate(...)`** — the policy, as a pure function: not aborted; not
  partial (unless `allow_partial`); dataset hash matches the current file; no
  errored samples beyond `max_errors`; mean score ≥ `min_score` (and pass rate
  ≥ `min_pass_rate` if given); every pinned case evaluated and passing **every**
  sample; baseline present if `require_baseline` (and a *committed* one, not
  just a previous run); no regressions; no baseline case missing from a full
  run; flakier-than-baseline cases warn (or fail with `fail_on_degraded`). Also warns
  when the judge or temperature differs from the baseline's (not like-for-like), and
  when every answer was replayed from cache (the model itself wasn't re-tested).

### `harness/core/report.py`

- **`_cell(text)`** — escapes `|` and newlines for Markdown tables.
- **`_scorer_stats(summary)`** — per-scorer pass rate over samples.
- **`_latency_stats(summary)`** — p50/p95 of uncached, non-errored samples.
- **`render_run_report(summary, diff=None)`** — Markdown: provenance, pass rate
  with 95% Wilson interval, partial/aborted/self-judged banners, baseline
  comparison, by-scorer table, per-case table, failure detail.
- **`summary_to_dict(summary, diff=None)`** — the same data for machines.
- **`write_report(summary, diff, out_dir="reports")`** — writes `.md` and
  `.json`; returns the `.md` path.
- **`render_comparison(task, runs, results)`** — multi-model table and the
  cases where models disagree, with a note that intervals are wide.

### `harness/scorers/base.py`

- **`Scorer`** (ABC) — `score(case, trace) -> ScoreResult`.

### `harness/scorers/contains.py`

- **`normalize_text(text)`** — ignores case, spacing, hyphens, thousands separators,
  subscripts and stray spaces inside decimals, so harmless formatting isn't a false alarm.
- **`ContainsScorer.score`** — each `expected.answer_contains` entry (a string, or a list
  of acceptable alternatives) must appear; score `1 − missing/required`; empty list is
  not applicable.

### `harness/scorers/citations.py`

- **`extract_citations(text)`** — literal `docN` ids only ("Document 1" usually means a
  position, so it's ignored).
- **`CitationValidityScorer.score`** — every cited id must have been retrieved. A citation
  to a document the model never saw is a fabricated source. Free; no citations means
  not applicable.

### `harness/scorers/retrieval.py`

- **`RetrievalHitRateScorer.score`** — fraction of `expected.relevant_doc_ids`
  present in the retrieved chunks; passes only at 100%; empty list auto-passes.

### `harness/scorers/embedding_sim.py`

- **`EmbeddingSimilarityScorer.__init__(threshold=0.5)`**, **`.score`** —
  cosine similarity to `expected.reference_answer`; missing reference
  auto-passes, empty output auto-fails.

### `harness/scorers/tool_usage.py`

- **`ToolUsageScorer.score`** — three meanings of `expected_tools`: **key
  absent** = no requirement; **a list** = each named tool must be called (hit
  rate; `expected_arg_contains` must appear in the arguments, else the score is
  halved); **an explicit empty list** = *no tool may be called* (this is how
  "don't force a tool when none applies" is actually enforced).

### `harness/scorers/llm_judge.py`

- **`JUDGE_SYSTEM_PROMPT`** — anchored 0–10 rubric; treats the candidate as
  untrusted data; any attempt to manipulate the grader scores 0.
- **`_neutralise(text)`** — removes the delimiter tags from untrusted text.
- **`build_judge_prompt(question, reference, candidate)`** — three
  XML-delimited fields.
- **`GROUNDEDNESS_SYSTEM_PROMPT`**, **`build_groundedness_prompt(question, documents, candidate)`**
  — judges support by the retrieved documents only, ignoring outside knowledge.
- **`_LLMGrader._grade(prompt, system)`** — shared by both graders: calls the model,
  records the cost, parses and clamps. An unparseable or non-numeric reply is a failing
  result with an `error` detail (distinguishable from a genuine low score).
- **`LLMJudgeScorer.score`** — "is it correct?", against the reference answer. An empty
  candidate fails without calling the judge.
- **`GroundednessScorer.score`** — "is it supported?", against the retrieved documents.
  Needs no reference answer; with no documents it is not applicable.
- **`_extract_json(text)`** — scans every `{` and returns the **last** JSON
  object with a `score` key, so thinking-out-loud replies, code fences and stray
  braces parse.

### `harness/adapters/base.py`

- **`Adapter`** (ABC) — `run(case, provider) -> RunTrace`. Must not raise for an
  ordinary failure (put it in `RunTrace.error`); must let `BudgetExceededError`
  propagate.

### `harness/adapters/tools.py`

- **`_safe_eval_node(node)`**, **`calculator(expression)`**,
  **`kb_lookup(topic)`**, **`execute_tool(name, tool_input)`** — see trace 6.
  `calculator` never raises: it returns an `"error: ..."` string.

### `harness/adapters/rag_fusion.py`

- **`get_target_module()`**, **`_leg_cost(model, response)`**,
  **`_clean_query_lines(raw_text)`**, **`RagFusionAdapter.__init__`**,
  **`._get_collection()`**, **`.run(case, provider)`** — see trace 4.

### `harness/adapters/tool_agent.py`

- **`_cost_of(model, response)`**, **`ToolAgentAdapter.run(case, provider)`** —
  see trace 5.

### `harness/tasks.py`

- **`TaskSpec`** (frozen) — `dataset_path`, `adapter_factory`,
  `scorers_factory`.
- **`_rag_fusion_adapter()`**, **`_rag_fusion_scorers(judge)`**,
  **`_tool_agent_adapter()`**, **`_tool_agent_scorers(judge)`** — lazy-import so
  selecting one task doesn't import the other's dependencies.
- **`get_task(task_name)`** — `ValueError` listing the known tasks.

### `harness/cli.py`

- **`_task_settings`**, **`_judge_settings`** — build `GenerationSettings`.
- **`_make_provider(...)`** — key check, Ollama preflight, `get_provider`.
- **`_resolve_judge(...)`**, **`_baseline_kwargs(...)`**.
- **`cmd_run`**, **`cmd_diff`**, **`cmd_gate`**, **`cmd_baseline_set`**,
  **`cmd_baseline_show`**, **`cmd_compare`**, **`cmd_dashboard`**, **`cmd_calibrate`**,
  **`cmd_history`**, **`cmd_doctor`** — one per subcommand; each traced in
  Part 2.
- **`build_parser()`** — shared parent parsers (`common`: task/model;
  `baseline_opts`; `gen_opts`) and one subparser per command with
  `set_defaults(func=...)`.
- **`main(argv=None)`** — entry point; maps exceptions to exit codes.

---

## Order-of-functions cheat sheet

For the command that matters most, the condensed call chain:

```
main()
 └─ build_parser() / setup_logging()
 └─ cmd_run()
     ├─ load_config()
     ├─ _task_settings() / _resolve_judge()
     ├─ ResponseCache() / BudgetGuard()
     ├─ _make_provider() x2  (task model, judge)
     │   ├─ config.require_api_key()
     │   ├─ check_ollama()               [ollama only]
     │   └─ get_provider()  → Budget → Cache wrappers
     ├─ get_task() / load_jsonl() / file_sha256() / filter_smoke()
     ├─ task_spec.adapter_factory() / scorers_factory()
     ├─ Runner(repeats=N)
     ├─ runner.run(cases)
     │   └─ per (case, sample), concurrently:  Runner._run_one()
     │       ├─ provider.scoped(i)
     │       ├─ adapter.run()          [rag_fusion_qa | tool_agent_qa]
     │       └─ scorer.score()  x each, if no error
     ├─ collect_meta()
     ├─ RunStore() → save()
     ├─ diff_runs() → resolve_baseline() → diff_cases()
     └─ write_report()  (.md + .json)  → exit 0 / 1 / 2

gate:   cmd_gate() → get_run() → diff_runs() → evaluate_gate() → exit 0 / 1 / 2
baseline set:  cmd_baseline_set() → baseline_from_store() → write_baseline()
```
