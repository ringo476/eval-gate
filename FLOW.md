# Project Flow — Complete Reference

This document replaces `PROJECT_GUIDE.md`. It has three parts:

1. **What this project does** — one paragraph, no jargon.
2. **Execution traces** — for every CLI command, the exact sequence of
   function calls from the moment you press Enter to the moment it exits,
   in order, naming every function involved.
3. **Function reference** — every function and method in the codebase,
   file by file, in the order they appear in that file: what it takes,
   what it does, what it returns, anything non-obvious about it.

If you only read one section, read part 2 for the task you care about
(`run` is the one that matters most) — it tells you, in order, everything
that happens.

---

## Part 1: What this project does

This is an **evaluation harness** for LLM-powered applications — the
regression-test-suite equivalent for systems whose output isn't
deterministic (you can't `assert output == "expected string"` on a model
call). It runs a fixed, versioned set of test questions ("golden cases")
against a target app, grades every answer with several different scoring
methods, compares the result to the previous run to catch regressions,
and can block a CI pipeline if quality drops.

It evaluates **two** target apps through the same core machinery:

- **`rag_fusion_qa`** — a fixed pipeline (expand query → retrieve →
  fuse → synthesize answer) wrapping a vendored third-party project,
  [Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion).
- **`tool_agent_qa`** — a genuine agent (built for this project) with a
  calculator and a knowledge-base lookup tool, where the model itself
  decides which tool to call, in what order, and when to stop.

---

## Part 2: Execution traces

### `python -m harness.cli run --task rag_fusion_qa --provider openai --smoke 10`

This is the main command. Here is *everything* that happens, in order.

1. **`harness/cli.py: main()`** runs first (the `if __name__ == "__main__"`
   guard at the bottom of the file calls it). It reconfigures
   `sys.stdout`/`sys.stderr` to UTF-8 so Unicode in model output can never
   crash printing. It calls **`build_parser()`** to construct the
   argparse parser, calls `parser.parse_args()` to parse your command
   line into `args`, calls **`setup_logging(args.verbose)`**
   (`harness/logging_setup.py`) to configure the logger, then looks up
   `args.func` — which `build_parser()` set to **`cmd_run`** for the
   `run` subcommand — and calls it.

2. **`cmd_run(args)`** (`harness/cli.py`) runs next:
   1. Calls **`load_config()`** (`harness/config.py`) — reads `.env` (via
      `python-dotenv`'s `load_dotenv()`, called at module import time),
      pulls every `HARNESS_*` and `*_API_KEY` env var, validates
      `HARNESS_MAX_COST_USD`/`HARNESS_MAX_WORKERS` are valid numbers, and
      returns a `Config` object.
   2. Constructs a **`ResponseCache`** (`harness/core/cache.py`), whose
      `__init__` opens (or creates) the SQLite file at
      `config.cache_db_path` and runs a `CREATE TABLE IF NOT EXISTS`.
   3. Constructs a **`BudgetGuard`** (`harness/core/budget.py`) with the
      resolved max-cost value.
   4. Calls **`config.require_api_key(args.provider)`** — raises a clear
      `RuntimeError` immediately if the relevant API key env var is
      missing, *before* any network call is attempted.
   5. Calls **`_resolve_judge_model(config, args.provider)`** — returns
      `config.judge_model` if you set `HARNESS_JUDGE_MODEL`, else looks up
      a sensible default for your provider in
      `DEFAULT_JUDGE_MODEL_BY_PROVIDER`.
   6. Calls **`get_provider(...)`** (`harness/core/providers.py`) *twice*
      — once for the task model, once for the judge model. Each call:
      constructs the real provider (`AnthropicProvider` or
      `OpenAIProvider`), wraps it in a **`BudgetEnforcingProvider`**, then
      wraps *that* in a **`CachingProvider`**. (Order matters: budget
      enforcement sits directly on the real network call, so a cache hit
      never touches the budget.)
   7. Calls **`get_task(args.task)`** (`harness/tasks.py`) — looks up the
      `TaskSpec` (dataset path + adapter factory + scorer factory) for
      the requested task name in the `TASKS` dict.
   8. Calls **`load_jsonl(dataset_path)`** (`harness/core/dataset.py`) —
      reads the golden-set file line by line, validates every row has
      `id`/`task`/`input`/`expected`, rejects duplicate ids, and returns a
      list of `EvalCase` objects.
   9. If `--smoke N` was passed, calls **`filter_smoke(cases, n=N)`** —
      keeps every pinned case plus the first N others.
   10. Calls **`task_spec.adapter_factory()`** — for `rag_fusion_qa` this
       constructs a `RagFusionAdapter`; for `tool_agent_qa`, a
       `ToolAgentAdapter`.
   11. Calls **`task_spec.scorers_factory(judge_provider)`** — builds the
       list of `Scorer` instances for this task (e.g.
       `[ContainsScorer(), RetrievalHitRateScorer(),
       EmbeddingSimilarityScorer(), LLMJudgeScorer(judge_provider)]` for
       `rag_fusion_qa`).
   12. Constructs a **`Runner`** (`harness/core/runner.py`) with the
       adapter, scorer list, task provider, and worker count.
   13. Calls **`runner.run(cases)`** — see the nested trace below. This
       is where all the actual model calls, scoring, and time is spent.
   14. Constructs a **`RunStore`** (`harness/core/store.py`), whose
       `__init__` calls **`_init_schema()`** to create the `runs` and
       `case_results` SQLite tables if they don't exist.
   15. Calls **`store.save(summary)`** — writes one row to `runs` and one
       row per case to `case_results`, in a single transaction.
   16. Calls **`diff_runs(store, task, model, run_id)`**
       (`harness/core/diff.py`) — looks up the previous run for the same
       (task, model) via **`store.latest_run_id(...)`**, pulls that run's
       per-case pass/fail via **`store.case_results_for_run(...)`**, and
       classifies every case in the current run as a regression,
       improvement, or unchanged.
   17. Calls **`write_report(summary, diff)`**
       (`harness/core/report.py`) — internally calls
       **`render_run_report(summary, diff)`** to build the Markdown
       string, then writes it to `reports/<run_id>.md`.
   18. Closes the store and cache, prints the report to stdout, prints the
       budget spent, and returns an exit code (0 if no case errored, 1
       otherwise).

3. **Inside `runner.run(cases)`** (`harness/core/runner.py`), the nested
   sequence, per case, run concurrently across a `ThreadPoolExecutor`
   with `max_workers` threads:
   1. For each case, `Runner._run_one(case)` is submitted to the pool.
   2. `_run_one` calls **`adapter.run(case, provider)`** — this is where
      the target app actually executes. For `rag_fusion_qa`, that's
      `RagFusionAdapter.run()` (traced separately below); for
      `tool_agent_qa`, `ToolAgentAdapter.run()` (also traced below).
   3. If `trace.error` is set, `_run_one` returns immediately with an
      empty `scores` list — no scorer runs on a failed case.
   4. Otherwise, `_run_one` calls **`scorer.score(case, trace)`** for
      every scorer in the task's list, catching any exception per-scorer
      so one broken scorer can't sink the case.
   5. Back in `run()`, once every future completes, results are re-sorted
      into the original dataset order (thread completion order isn't
      deterministic), wrapped into a `RunSummary`, logged, and returned.

4. **Inside `RagFusionAdapter.run(case, provider)`**
   (`harness/adapters/rag_fusion.py`), for the `rag_fusion_qa` task:
   1. Calls **`get_target_module()`** — imports the vendored
      `targets/rag-fusion/main.py` under the module name
      `rag_fusion_target` (only does the actual import once; cached in
      the module-level `_target_module` variable after that).
   2. Calls **`self._get_collection()`** — on the first call, acquires
      `self._collection_lock`, constructs a `chromadb.Client()`, and adds
      every document in `self.corpus_docs` (from `harness/corpus.py`) to
      an in-memory collection. Locked because concurrent construction
      from multiple threads produced real crashes during testing.
   3. Calls **`provider.generate(...)`** with the query-expansion prompt
      and `QUERY_EXPANSION_SYSTEM` — this is the first real model call.
   4. Calls **`_clean_query_lines(expansion_resp.text)`** — strips
      numbering/bullet artifacts from the model's line-per-query output.
   5. Under `self._collection_lock`, calls **`target.vector_search(...)`**
      (the vendored, unmodified function) once per query, then
      **`target.reciprocal_rank_fusion(...)`** (also vendored, unmodified)
      to merge the ranked lists, then `collection.get(ids=...)` to fetch
      the actual text of the top-`k` fused documents.
   6. Calls **`provider.generate(...)`** again with the synthesis prompt
      and `SYNTHESIS_SYSTEM` — the second real model call, producing the
      final answer.
   7. Calls **`_leg_cost(...)`** twice (once per model call) to compute
      actual USD spent (0 if that particular call was a cache hit).
   8. Returns a `RunTrace` with the final answer, the retrieved chunks,
      combined token counts, combined cost, and latency.
   9. Any exception anywhere in the above is caught, and a `RunTrace`
      with `error` set is returned instead — the adapter never raises.

5. **Inside `ToolAgentAdapter.run(case, provider)`**
   (`harness/adapters/tool_agent.py`), for the `tool_agent_qa` task —
   a loop, not a fixed sequence:
   1. Starts `messages = [{"role": "user", "content": query}]`.
   2. Loops up to `MAX_STEPS` (4) times. Each iteration:
      - Calls **`provider.generate_messages(messages, system=AGENT_SYSTEM,
        tools=TOOL_SCHEMAS)`** — a real model call with the full
        conversation so far.
      - If the response has no tool calls, this is the final answer:
        returns a `RunTrace` immediately.
      - Otherwise, calls **`provider.build_assistant_message(response)`**
        to get this turn's assistant message in the *correct* wire shape
        for whichever provider is in use, and appends it to `messages`.
      - For every tool call in the response, calls
        **`execute_tool(tc["name"], tc["input"])`**
        (`harness/adapters/tools.py`) — runs `calculator()` or
        `kb_lookup()` for real, locally.
      - Calls **`provider.build_tool_result_messages(executed)`** to wrap
        the tool outputs into the provider's expected message shape, and
        extends `messages` with them.
      - Loops back to call the model again with the updated conversation.
   3. If `MAX_STEPS` is exhausted with no final answer, returns a
      `RunTrace` with an explicit `error` (not a silent empty success).
   4. Same as `RagFusionAdapter`: any exception is caught and turned into
      an error trace.

6. **Inside `execute_tool(name, tool_input)`**
   (`harness/adapters/tools.py`):
   - For `"calculator"`: calls **`calculator(expression)`**, which calls
     **`ast.parse(expression, mode="eval")`** then
     **`_safe_eval_node(tree.body)`** — a recursive function that only
     allows numeric constants and `+ - * /` (via `_ALLOWED_BINOPS`) and
     unary `+`/`-` (via `_ALLOWED_UNARYOPS`), raising `CalculatorError`
     for anything else (including any name lookup or function call).
   - For `"kb_lookup"`: calls **`kb_lookup(topic)`**, which calls
     **`search_corpus(topic, top_k=1)`** (`harness/core/corpus_search.py`),
     which calls **`most_similar(...)`** (`harness/core/embeddings.py`),
     which calls **`get_embedding_model()`** (loads/caches the
     `sentence-transformers` model) and **`cosine_similarity(...)`** to
     rank every document in the corpus against the query.

### `python -m harness.cli gate --task rag_fusion_qa --min-score 0.75`

1. `main()` → `cmd_gate(args)`.
2. `load_config()`, then a **`RunStore`**, then
   **`store.latest_run_id(task, model)`** to find the run to check (or
   uses `--run-id` if given). Exits with code 2 if no run exists.
3. `load_jsonl(dataset_path)` to know which case ids are pinned
   (`EvalCase.is_pinned`).
4. **`store.case_results_for_run(run_id)`** to get every case's
   pass/fail and score.
5. **`diff_runs(...)`** again, to check for regressions against the
   *previous* run (not just the absolute score bar).
6. Computes the mean score by hand from `case_results`, checks it against
   `--min-score`, checks no pinned case is in `case_results` with
   `passed=False`, checks `diff.regressions` is empty. Prints `PASS` or
   one `FAIL` line per violated condition, returns 0 or 1.

### `python -m harness.cli diff --task rag_fusion_qa`

1. `main()` → `cmd_diff(args)`.
2. `load_config()`, `RunStore`, `store.latest_run_id(...)`,
   `diff_runs(...)`. Prints baseline/current run ids and the
   regression/improvement/unchanged counts. Returns 1 if there were any
   regressions, else 0.

### `python -m harness.cli calibrate --provider openai`

This one is independent of the target apps entirely — no adapter, no
retrieval, no tool execution. It tests the judge in isolation.

1. `main()` → `cmd_calibrate(args)`.
2. `load_config()`, `config.require_api_key(args.provider)`,
   `_resolve_judge_model(...)`, a `ResponseCache`, then
   **`get_provider(...)`** once (there's no separate "task model" here —
   the thing under test *is* the judge).
3. Constructs an **`LLMJudgeScorer(judge_provider)`**
   (`harness/scorers/llm_judge.py`).
4. Loads `golden/calibration_labels.jsonl` via `load_jsonl(...)`. Each
   row has a `candidate_answer` already written into it — a fixed,
   hand-authored answer, not something a model just produced.
5. For every case: builds a synthetic `RunTrace` whose `output_text` is
   the case's `candidate_answer`, then calls **`judge.score(case,
   trace)`** — which internally calls **`judge_provider.generate(...)`**
   with the question/reference/candidate baked into the prompt, then
   **`_extract_json(response.text)`** to pull the judge's `{"score":
   ..., "reason": ...}` out of its reply.
6. Collects `(human_label, judge_score)` pairs, computes mean absolute
   error and (if both series have nonzero variance) Pearson correlation
   via `numpy`, and Pearson-correlation-free "agreement within ±0.2".
7. Prints a table (via `tabulate`) and a verdict; returns 1 if MAE > 0.25
   or correlation < 0.5 or undefined, else 0.

### `python -m harness.cli history --task rag_fusion_qa`

1. `main()` → `cmd_history(args)`.
2. `load_config()`, `RunStore`, **`store.run_history(task, model,
   limit)`** — a plain `SELECT ... ORDER BY timestamp DESC LIMIT ?`.
   Prints the result as a table.

---

## Part 3: Complete function reference

Organized file by file, in the order each function/method appears in
that file.

### `harness/config.py`

- **`Config`** (dataclass) — holds every resolved setting for a run:
  both API keys (either may be `None`), the cost cap, worker count, the
  (optional) judge model override, and the two SQLite file paths.
- **`Config.require_api_key(self, provider_name)`** — returns the right
  key for `"openai"` or `"anthropic"`, raising a `RuntimeError` with an
  actionable message (mentions `.env.example`, where to get a key, and
  the `--provider mock` escape hatch) if it's missing. Raises
  `ValueError` for any other `provider_name`.
- **`load_config()`** — the only function in the codebase that reads
  `os.environ` directly. Parses `HARNESS_MAX_COST_USD` and
  `HARNESS_MAX_WORKERS` as numbers (raising a clear error if they aren't),
  validates they're positive/≥1, and returns a populated `Config`.

### `harness/logging_setup.py`

- **`setup_logging(verbose=False)`** — configures the `"harness"` logger
  tree with a single `StreamHandler` to stderr, `DEBUG` level if
  `verbose` else `INFO`. Called once, from `cli.main()`.

### `harness/core/models.py`

Pure dataclasses, no logic beyond a few computed properties:

- **`EvalCase`** — one golden-set row. `.is_pinned` (property) — `True`
  if `"pinned"` is in `.tags`.
- **`ProviderResponse`** — the raw result of one model call: `text`,
  `tool_calls`, token counts, `latency_ms`, `raw` (provider-specific,
  used to replay a turn), `cache_hit`.
- **`RunTrace`** — what an adapter hands back to the runner for one case:
  `output_text`, `retrieved_chunks`, `tool_calls`, `model`, token counts,
  `latency_ms`, `cost_usd`, `cache_hit`, `error` (`None` unless the
  adapter failed).
- **`ScoreResult`** — one scorer's verdict on one case: `scorer_name`,
  `score` (0–1), `passed`, `details` (a dict, only populated on
  failure/interest).
- **`CaseResult`** — one case's trace plus all its scores.
  `.aggregate_score` (property) — mean of all scores, or 0 if the trace
  errored or there are no scores. `.passed` (property) — `False` if the
  trace errored, else `all(s.passed for s in scores)`.
- **`RunSummary`** — one full run: id, task, model, timestamp, and every
  `CaseResult`. Properties: `.total_cost_usd`, `.total_latency_ms`,
  `.error_count`, `.pass_rate`, `.mean_score` — all simple aggregations
  over `case_results`.

### `harness/core/pricing.py`

- **`estimate_cost_usd(model, input_tokens, output_tokens)`** — looks up
  `model` in the `PRICING_PER_MTOK` dict and computes
  `input_tokens/1e6 * input_rate + output_tokens/1e6 * output_rate`.
  Returns `0.0` for an unrecognized model name rather than raising, so an
  unlisted future model can't crash a run (only silently under-report
  cost — check logs if a run looks suspiciously cheap).

### `harness/core/budget.py`

- **`BudgetExceededError`** — raised when a reservation would exceed the
  cap.
- **`BudgetGuard.__init__(self, max_cost_usd)`** — sets the cap, `_spent
  = 0.0`, and a `threading.Lock` (this object is shared across worker
  threads).
- **`BudgetGuard.spent`** (property) — thread-safe read of the running
  total.
- **`BudgetGuard.check_and_reserve(self, estimated_cost)`** — under the
  lock, raises `BudgetExceededError` if `spent + estimated_cost >
  max_cost_usd`; otherwise does nothing (the actual increment happens in
  `.record()`, called *after* the real call, so the "reservation" is a
  check, not a two-phase commit).
- **`BudgetGuard.record(self, actual_cost)`** — under the lock, adds
  `actual_cost` to `_spent`.

### `harness/core/cache.py`

- **`_cache_key(model, system, prompt, tools)`** — serializes the four
  inputs to sorted-key JSON and SHA-256 hashes it. `prompt` is either a
  literal prompt string (`generate()`) or a JSON-serialized message list
  (`generate_messages()`, via `CachingProvider`) — same function either
  way, the caller decides what "prompt" means for that call shape.
- **`ResponseCache.__init__(self, db_path)`** — opens/creates the SQLite
  file, creates the `response_cache` table if absent.
- **`ResponseCache.get(self, model, system, prompt, tools)`** — looks up
  the row by cache key; returns `None` on a miss, else reconstructs and
  returns a `ProviderResponse` from the stored JSON.
- **`ResponseCache.put(self, model, system, prompt, tools, response)`** —
  serializes `response` via `dataclasses.asdict`, tries `json.dumps` on
  it, and only nulls out `.raw` if that fails (i.e. `.raw` is preserved
  whenever it's already JSON-safe — needed so a cached agent turn can
  still be replayed correctly later). Writes with `INSERT OR REPLACE`.
- **`ResponseCache.close(self)`** — closes the SQLite connection.

### `harness/core/embeddings.py`

- **`get_embedding_model()`** — lazily loads and module-level-caches a
  `sentence-transformers` `SentenceTransformer("all-MiniLM-L6-v2")`, so
  the (relatively slow) model load happens at most once per process.
- **`cosine_similarity(a, b)`** — plain cosine similarity between two
  numpy vectors.
- **`most_similar(query, candidates, top_k=1)`** — embeds `query` and
  every value in the `candidates` dict in one batched `.encode()` call,
  scores each by `cosine_similarity`, and returns the top-`k`
  `(key, text, similarity)` tuples sorted descending.

### `harness/core/corpus_search.py`

- **`search_corpus(query, top_k=1)`** — thin wrapper calling
  `most_similar(query, CORPUS, top_k)` against the module-level `CORPUS`
  dict imported from `harness/corpus.py`. Used by both the `kb_lookup`
  tool and (indirectly, via the same corpus) the RAG-fusion adapter's own
  retrieval path.

### `harness/core/dataset.py`

- **`load_jsonl(path)`** — reads the file line by line (skipping blank
  lines and `#`-prefixed comments), `json.loads` each line, validates
  `id`/`task`/`input`/`expected` are all present, rejects duplicate
  `id`s, raises `FileNotFoundError`/`ValueError` with the exact line
  number on any problem, raises `ValueError` if the file parses to zero
  cases. Returns a `list[EvalCase]`.
- **`filter_smoke(cases, n=10)`** — splits `cases` into pinned and
  non-pinned, returns `(pinned + non_pinned)[:max(n, len(pinned))]` — so
  every pinned case survives a `--smoke` run no matter how small `n` is.

### `harness/core/providers.py`

- **`ModelProvider`** (ABC) — declares the four methods every provider
  must implement: `generate`, `generate_messages`,
  `build_assistant_message`, `build_tool_result_messages` (see Part 2's
  intro to this file for what each is for).
- **`AnthropicProvider.__init__(self, model, api_key)`** — lazily
  imports the `anthropic` package (so it's never required if you're not
  using this provider), constructs the client, stashes the two
  retry-relevant exception classes.
- **`AnthropicProvider._convert_tools(tools)`** (static) — maps the
  neutral `{name, description, parameters}` schema to Anthropic's
  `{name, description, input_schema}` shape.
- **`AnthropicProvider.generate(self, prompt, system, tools)`** — wraps
  `prompt` in a single user message and calls `self._call(...)`.
- **`AnthropicProvider.generate_messages(self, messages, system,
  tools)`** — calls `self._call(...)` directly with the given messages.
- **`AnthropicProvider._call(self, messages, system, tools)`** — the
  actual API call, wrapped in a `tenacity` `@retry` (exponential backoff
  with jitter, up to 5 attempts, only on rate-limit/connection errors).
  Builds the request kwargs, calls `self._client.messages.create(...)`,
  walks `resp.content` splitting text blocks from `tool_use` blocks into
  both a flat `tool_calls` list (for scorers to read) and a JSON-safe
  `raw_blocks` list (for replaying the turn later), and returns a
  `ProviderResponse`.
- **`AnthropicProvider.build_assistant_message(self, response)`** —
  `{"role": "assistant", "content": response.raw}` (Anthropic wants the
  exact original content blocks echoed back).
- **`AnthropicProvider.build_tool_result_messages(self, tool_results)`**
  — one user message containing a list of `tool_result` content blocks,
  one per executed tool call.
- **`OpenAIProvider.__init__(self, model, api_key)`** — same pattern as
  Anthropic's, lazily importing `openai`.
- **`OpenAIProvider._convert_tools(tools)`** (static) — maps the neutral
  schema to OpenAI's `{"type": "function", "function": {name,
  description, parameters}}` shape.
- **`OpenAIProvider.generate(self, prompt, system, tools)`** — builds a
  message list (optional system message + one user message) and calls
  `self._call(...)`.
- **`OpenAIProvider.generate_messages(self, messages, system, tools)`**
  — prepends a system message (if given) to the provided list and calls
  `self._call(...)`.
- **`OpenAIProvider._call(self, messages, tools)`** — same retry
  wrapping as Anthropic's. Calls
  `self._client.chat.completions.create(...)`, reads
  `resp.choices[0].message`; if it has `tool_calls`, parses each one's
  JSON-string `arguments` into a dict for the normalized `tool_calls`
  list while keeping the original JSON string in `raw` (OpenAI expects
  that exact string echoed back, not a re-serialized dict — re-encoding
  could subtly change key order/whitespace). Returns a `ProviderResponse`
  whose `.raw` is `{"content": ..., "tool_calls": ...}`.
- **`OpenAIProvider.build_assistant_message(self, response)`** —
  `{"role": "assistant", "content": ..., "tool_calls": ...}` (only
  includes the `tool_calls` key if there were any).
- **`OpenAIProvider.build_tool_result_messages(self, tool_results)`** —
  one message per tool call, each `{"role": "tool", "tool_call_id": ...,
  "content": ...}` — OpenAI's shape, one message per result, unlike
  Anthropic's single combined message.
- **`BudgetEnforcingProvider.__init__(self, inner, budget,
  estimate_cost_fn)`** — stores the wrapped provider, the shared
  `BudgetGuard`, and the pricing function.
- **`BudgetEnforcingProvider._enforce(self, worst_case_input_tokens,
  call_fn)`** — calls `budget.check_and_reserve(...)` with a worst-case
  estimate *before* calling `call_fn()`, then `budget.record(...)` with
  the real cost after. This is the actual enforcement point for both
  `generate` and `generate_messages` below.
- **`BudgetEnforcingProvider.generate(...)`** /
  **`.generate_messages(...)`** — estimate a worst-case token count from
  word-splitting the prompt/messages, then delegate to `_enforce`.
- **`BudgetEnforcingProvider.build_assistant_message(...)`** /
  **`.build_tool_result_messages(...)`** — pure pass-through to
  `self._inner`.
- **`CachingProvider.__init__(self, inner, cache)`** — stores the
  wrapped provider and the shared `ResponseCache`.
- **`CachingProvider.generate(...)`** — delegates to
  `self._cached_call(prompt, ...)`.
- **`CachingProvider.generate_messages(...)`** — serializes `messages`
  to sorted JSON as the cache key representation, delegates to
  `self._cached_call(...)`.
- **`CachingProvider._cached_call(self, key_repr, system, tools,
  call_fn)`** — checks `self._cache.get(...)`; on a hit, marks
  `cache_hit = True` and returns it without calling `call_fn`; on a
  miss, calls `call_fn()`, stores the result, marks `cache_hit = False`,
  returns it.
- **`CachingProvider.build_assistant_message(...)`** /
  **`.build_tool_result_messages(...)`** — pure pass-through, same as
  `BudgetEnforcingProvider`'s.
- **`MockProvider.__init__(self, model="mock")`** — ignores `model`,
  always names itself `"mock"`.
- **`MockProvider.generate(self, prompt, system, tools)`** — returns a
  canned `f"[mock answer for: {prompt[:60]}]"` string, instantly, at $0.
- **`MockProvider.generate_messages(self, messages, system, tools)`** —
  the only provider method with real conditional logic: checks whether
  the last message is a tool result; if so (or if no tools were offered),
  returns a canned final answer. Otherwise, inspects the *first* user
  message's text with a crude heuristic (digits + an operator word/symbol
  → `calculator`; else → `kb_lookup`) and returns a fake tool call.
  Deliberately simplistic — exists to exercise the agent loop's code
  paths for $0, not to simulate real judgment.
- **`MockProvider.build_assistant_message(...)`** /
  **`.build_tool_result_messages(...)`** — same shape as
  `AnthropicProvider`'s (an arbitrary but internally consistent choice,
  since nothing sends Mock's messages to a real API).
- **`get_provider(provider_name, model, api_key, cache, budget=None)`** —
  the single construction point every caller uses. Returns a bare
  `MockProvider()` immediately if `provider_name == "mock"`. Otherwise
  requires `api_key`, constructs the right real provider class, wraps it
  in `BudgetEnforcingProvider` (if `budget` given) then `CachingProvider`
  (if `cache` given) — that order is deliberate, see the module
  docstring.

### `harness/core/runner.py`

- **`Runner.__init__(self, adapter, scorers, provider, max_workers=4)`**
  — just stores its arguments.
- **`Runner._run_one(self, case)`** — calls `adapter.run(case,
  provider)`; if the resulting trace has an error, returns a
  `CaseResult` with an empty score list immediately (no scorer runs on a
  failed case); otherwise runs every scorer, catching per-scorer
  exceptions and substituting a synthetic failing `ScoreResult` for any
  scorer that raised.
- **`Runner.run(self, cases)`** — generates a `run_id` (via
  `new_run_id()`), submits every case's `_run_one` to a
  `ThreadPoolExecutor(max_workers)`, collects results as they complete,
  re-sorts them back into the original dataset order (thread-completion
  order isn't the same as input order), builds and returns a
  `RunSummary`.

### `harness/core/store.py`

- **`RunStore.__init__(self, db_path)`** — opens the SQLite connection,
  enables `PRAGMA foreign_keys`, calls `self._init_schema()`.
- **`RunStore._init_schema(self)`** — `CREATE TABLE IF NOT EXISTS` for
  `runs` and `case_results`, plus an index on `(task, model,
  timestamp)`.
- **`RunStore.save(self, summary)`** — in one transaction, inserts one
  row into `runs` and one row per `CaseResult` into `case_results`
  (with that case's scores serialized to a JSON column).
- **`RunStore.latest_run_id(self, task, model, before_run_id=None)`** —
  `SELECT run_id ... ORDER BY timestamp DESC LIMIT 1`, optionally
  excluding a specific run id (used to find "the run before this one").
- **`RunStore.case_results_for_run(self, run_id)`** — returns
  `{case_id: {"passed": bool, "score": float}}` for one run.
- **`RunStore.run_history(self, task, model, limit=20)`** — the raw rows
  behind `harness history`.
- **`RunStore.close(self)`** — closes the connection.
- **`new_run_id()`** — `f"{utc_timestamp}-{uuid_hex[:8]}"`.
- **`now_iso()`** — current UTC time as an ISO string, used for the
  `timestamp` column.

### `harness/core/diff.py`

- **`DiffResult`** (dataclass) — `baseline_run_id` (or `None` if this is
  the first run), `current_run_id`, and the four buckets:
  `regressions`, `improvements`, `unchanged_pass`/`unchanged_fail`
  counts, `new_cases`.
- **`diff_runs(store, task, model, current_run_id)`** — finds the
  baseline via `store.latest_run_id(..., before_run_id=current_run_id)`;
  if there isn't one, everything in the current run counts as "new" and
  the function returns early. Otherwise pulls both runs' case results
  and classifies every case in the current run: in baseline and was
  passing but now fails → regression; was failing but now passes →
  improvement; not in baseline at all → new; otherwise → one of the two
  "unchanged" counters.

### `harness/core/report.py`

- **`render_run_report(summary, diff=None)`** — builds the Markdown
  string: a header block of run stats, an optional "Regression check"
  section (only if `diff` was passed), a table of every case's
  pass/fail/score/cost/latency/error, and a "Failure detail" section
  listing the query, output, error, and every failed scorer's details
  for each failing case.
- **`write_report(summary, diff, out_dir="reports")`** — calls
  `render_run_report(...)` and writes the result to
  `reports/<run_id>.md`, creating the directory if needed. Returns the
  `Path`.

### `harness/scorers/base.py`

- **`Scorer`** (ABC) — declares `score(self, case, trace) ->
  ScoreResult`, the one method every scorer implements.

### `harness/scorers/contains.py`

- **`ContainsScorer.score(self, case, trace)`** — reads
  `case.expected["answer_contains"]` (a list of required substrings); if
  empty, auto-passes with a note; otherwise checks each substring
  case-insensitively against `trace.output_text`, scoring
  `1 - (missing_count / required_count)`.

### `harness/scorers/retrieval.py`

- **`RetrievalHitRateScorer.score(self, case, trace)`** — reads
  `case.expected["relevant_doc_ids"]`; if empty, auto-passes (this is
  how adversarial/out-of-corpus cases opt out of a retrieval
  requirement); otherwise computes the overlap between that set and the
  doc ids in `trace.retrieved_chunks`, scoring the hit rate (fraction
  found), passing only at 100%.

### `harness/scorers/embedding_sim.py`

- **`EmbeddingSimilarityScorer.__init__(self, threshold=0.5)`** — stores
  the pass/fail cosine-similarity cutoff.
- **`EmbeddingSimilarityScorer.score(self, case, trace)`** — reads
  `case.expected["reference_answer"]`; if missing, auto-passes; if
  `trace.output_text` is empty, auto-fails; otherwise calls
  `get_embedding_model()` and `cosine_similarity(...)` (both from
  `harness/core/embeddings.py`) to score the semantic similarity between
  the reference and the actual output.

### `harness/scorers/llm_judge.py`

- **`LLMJudgeScorer.__init__(self, judge_provider, pass_threshold=0.7)`**
  — stores the judge provider, the pass threshold, and a running
  `total_cost_usd` counter for this scorer instance (used by
  `cmd_calibrate` to report calibration cost).
- **`LLMJudgeScorer.score(self, case, trace)`** — builds a prompt from
  the question (`case.input["question"]` or `["query"]`), the reference
  answer, and the candidate (`case.input["candidate_answer"]` if present
  — the calibration path — else `trace.output_text`); if the candidate
  is empty, auto-fails without calling the judge; otherwise calls
  `judge_provider.generate(prompt, system=JUDGE_SYSTEM_PROMPT)`, adds the
  call's cost to `total_cost_usd` (unless it was a cache hit), calls
  `_extract_json(response.text)` to parse the verdict, and returns a
  `ScoreResult` with `score = parsed_score / 10`.
- **`_extract_json(text)`** — regex-finds the first `{...}` block in the
  judge's reply and `json.loads`s it, returning `None` on any failure
  (malformed JSON, no braces at all) rather than raising — the caller
  turns a `None` into a failing `ScoreResult` with the raw text attached
  for debugging.

### `harness/scorers/tool_usage.py`

- **`ToolUsageScorer.score(self, case, trace)`** — reads
  `case.expected["expected_tools"]`; if empty, auto-passes; otherwise
  computes the overlap between that set and the tool names actually in
  `trace.tool_calls`, scoring the hit rate. If `expected_arg_contains`
  is also set and every expected tool was called, additionally checks
  that substring appears somewhere in the stringified arguments of the
  actual calls, halving the score if not.

### `harness/adapters/base.py`

- **`Adapter`** (ABC) — declares `run(self, case, provider) ->
  RunTrace`, with a docstring contract: must never raise for an ordinary
  failure, must put failures in `RunTrace.error` instead.

### `harness/adapters/tools.py`

- **`_safe_eval_node(node)`** — recursive AST walker; see Part 2's trace
  for the allow-list logic.
- **`calculator(expression)`** — parses and safely evaluates arithmetic;
  returns the result as a string, or an `"error: ..."` string for any
  failure (division by zero, disallowed syntax, malformed expression) —
  never raises out to its caller.
- **`kb_lookup(topic)`** — semantic search over the corpus, returns the
  single best-matching document (prefixed with its id) or a
  "no relevant entry found" string.
- **`execute_tool(name, tool_input)`** — dispatches to `calculator()` or
  `kb_lookup()` by name, or returns an `"error: unknown tool"` string for
  anything else.

### `harness/adapters/rag_fusion.py`

- **`get_target_module()`** — imports and caches the vendored
  `targets/rag-fusion/main.py` as its own module (`rag_fusion_target`),
  raising a clear `RuntimeError` pointing at `scripts/setup_target.sh` if
  the file isn't there yet.
- **`_leg_cost(model, response)`** — `0.0` if the response was a cache
  hit, else `estimate_cost_usd(...)`.
- **`_clean_query_lines(raw_text)`** — strips leading `-`/`*`/digits/`.`
  from each line of the model's query-expansion output.
- **`RagFusionAdapter.__init__(self, corpus_docs, top_k=3)`** — stores
  the corpus dict, `top_k`, and initializes the lazy collection +
  its lock.
- **`RagFusionAdapter._get_collection(self)`** — double-checked-locking
  lazy init of the ChromaDB collection (see Part 2's trace for why the
  lock exists).
- **`RagFusionAdapter.run(self, case, provider)`** — the full pipeline;
  traced step by step in Part 2.

### `harness/adapters/tool_agent.py`

- **`_cost_of(model, response)`** — identical pattern to
  `rag_fusion.py`'s `_leg_cost`.
- **`ToolAgentAdapter.run(self, case, provider)`** — the agent loop;
  traced step by step in Part 2.

### `harness/tasks.py`

- **`TaskSpec`** (frozen dataclass) — `dataset_path`, `adapter_factory`
  (a zero-arg callable returning an `Adapter`), `scorers_factory` (a
  callable taking the judge provider and returning a `list[Scorer]`).
- **`_rag_fusion_adapter()`** — returns `RagFusionAdapter(CORPUS)`
  (imports `RagFusionAdapter` and `CORPUS` lazily, inside the function,
  so importing `harness.tasks` doesn't force-import ChromaDB/embeddings
  unless a task that needs them is actually selected).
- **`_rag_fusion_scorers(judge_provider)`** — returns the four-scorer
  list for `rag_fusion_qa`.
- **`_tool_agent_adapter()`** — returns `ToolAgentAdapter()`.
- **`_tool_agent_scorers(judge_provider)`** — returns the four-scorer
  list for `tool_agent_qa` (`ToolUsageScorer` in place of
  `RetrievalHitRateScorer`).
- **`get_task(task_name)`** — looks up `task_name` in the `TASKS` dict,
  raising `ValueError` (listing the known task names) if it's not there.

### `harness/cli.py`

- **`_resolve_judge_model(config, provider_name)`** — see Part 2's trace,
  step 2.5.
- **`cmd_run(args)`**, **`cmd_diff(args)`**, **`cmd_gate(args)`**,
  **`cmd_calibrate(args)`**, **`cmd_history(args)`** — one per
  subcommand; each fully traced in Part 2.
- **`build_parser()`** — constructs the `argparse.ArgumentParser` tree:
  a `common` parent parser (shared `--task`/`--model` args) and one
  subparser per command, each with `set_defaults(func=cmd_*)` so
  `main()` can dispatch generically.
- **`main()`** — the actual entry point; reconfigures stdout/stderr
  encoding, builds the parser, parses args, sets up logging, calls
  `args.func(args)` inside a `try/except` that turns any uncaught
  exception into a logged error and exit code 3 rather than a raw
  traceback.

---

## Order-of-functions cheat sheet

For the command that matters most, the condensed call chain (each `→`
is "calls"):

```
main()
 └─ build_parser()
 └─ setup_logging()
 └─ cmd_run()
     ├─ load_config()
     ├─ ResponseCache()
     ├─ BudgetGuard()
     ├─ config.require_api_key()
     ├─ _resolve_judge_model()
     ├─ get_provider()  x2  (task model, judge model)
     ├─ get_task()
     ├─ load_jsonl()
     ├─ filter_smoke()               [only if --smoke]
     ├─ task_spec.adapter_factory()
     ├─ task_spec.scorers_factory()
     ├─ Runner()
     ├─ runner.run(cases)
     │   └─ per case, concurrently:
     │       └─ Runner._run_one()
     │           ├─ adapter.run()
     │           │   ├─ [rag_fusion_qa] RagFusionAdapter.run()
     │           │   │   ├─ get_target_module()
     │           │   │   ├─ self._get_collection()
     │           │   │   ├─ provider.generate()          (query expansion)
     │           │   │   ├─ _clean_query_lines()
     │           │   │   ├─ target.vector_search()        (per query)
     │           │   │   ├─ target.reciprocal_rank_fusion()
     │           │   │   ├─ provider.generate()          (synthesis)
     │           │   │   └─ _leg_cost()  x2
     │           │   └─ [tool_agent_qa] ToolAgentAdapter.run()
     │           │       └─ loop up to MAX_STEPS:
     │           │           ├─ provider.generate_messages()
     │           │           ├─ provider.build_assistant_message()
     │           │           ├─ execute_tool()             (per tool call)
     │           │           │   ├─ calculator() → _safe_eval_node()
     │           │           │   └─ kb_lookup() → search_corpus() → most_similar()
     │           │           └─ provider.build_tool_result_messages()
     │           └─ scorer.score()  (once per scorer, if no error)
     ├─ RunStore()
     ├─ store.save()
     ├─ diff_runs()
     │   ├─ store.latest_run_id()
     │   └─ store.case_results_for_run()
     ├─ write_report()
     │   └─ render_run_report()
     └─ (print results, return exit code)
```
