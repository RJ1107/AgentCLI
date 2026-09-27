# AgentCLI experiments

Controlled experiments on AgentCLI's own design: the model is held fixed and one mechanism is
switched, so a difference in the results is the mechanism's doing, not the model's.

Every run is isolated: a fresh copy of its fixture, its own `AGENTCLI_HOME` (no user config,
memory, skills, or MCP servers), approval off, and memory, skills, MCP, and the code index
disabled. Costs are computed from each call's reported tokens (cache hits included) at
OpenRouter's prices on the day of the run, model by model. Runs that failed for
infrastructure reasons (rate limits, upstream errors) are reported separately and left out
of success rates.

| File | What it is |
| --- | --- |
| `harness.py` | Isolation, per-model usage tracking, live prices, result logging |
| `context_retention.py` | Experiment 1: compression strategies over a long session |
| `coding_tasks.py` | Experiments 2 and 3: coding tasks graded by their tests |
| `make_task_fixtures.py`, `make_hard_fixture.py` | Write the task fixtures |
| `memory_recall.py`, `memory_dataset.py` | Experiment 4: long-term memory recall |
| `report.py` | Tables from `results/*.jsonl` |
| `spend.py` | OpenRouter spend so far, for budgeting |

```bash
python evals/make_task_fixtures.py && python evals/make_hard_fixture.py
python evals/context_retention.py --repeats 2
python evals/coding_tasks.py --label workers --mode team --workers 1,2,4,6 --repeats 2
python evals/coding_tasks.py --label models --mode react --models openai/gpt-6-luna,deepseek/deepseek-v4-flash
python evals/report.py
```

## Experiment 1: compression over a long session

A 72-turn session in a 64k context window, so compression fires several times. Every turn
the agent reads a source file, which fills the context the way real work does. Every ten
turns the user states five project facts (30 in all), and at turns 60 and 65 eight of them
change ("the test database port moves from 5433 to 6544"). The facts appear in no file. At
the end the agent is asked all 30 without tools and graded on the current values; an answer
with the old value of a changed fact counts as stale.

Strategies (all else equal):

- `truncate`: drop the oldest turns, no summary.
- `summary`: summarize older turns whenever over budget.
- `layered_nohyst`: clear old tool results first, then summarize; compact only to just under
  the trigger (78% against the 80% trigger).
- `layered`: clear old tool results first, then summarize (the default).
- `layered_clear30`: the current design with the first layer at 30% instead of 50%.

The first design had `layered_nohyst` (compact only to just under the trigger) instead of
`layered_clear30`; its results are kept below.

## Experiment 2: how many parallel workers

`/team` on five tasks, each six independent modules with their own tests (`textkit`,
`numkit`, `collkit`, `parsekit`, and the harder `hardkit`), with 1, 2, 4, and 6 workers, GPT-6
Luna in every role. Measured: wall-clock time, success (all six modules pass), calls, cost.

## Experiment 3: models and tiers

The same five tasks, one agent per model, to compare success, time, and cost per solved task
across the models AgentCLI offers. The expensive models (GPT-6 Sol and Astra, Claude Opus 5.5)
ran each task once, the others twice. A live comparison of tiered `/team` against one strong
model in every role was left for later: at Opus prices it did not fit the budget.

## Results (2026-09-25)

All with GPT-6 Luna unless a model is named; about $8 of OpenRouter usage in total.

### Experiment 1: compression

**Current design** (workspace, protected 20%, clear at 50%, summarize at 80%; see the main
README), 3 runs each:

| strategy | facts kept | LLM summaries | cached input | cost/run |
|---|---|---|---|---|
| truncate | 1.3/30 | 0 | 91.5% | $0.056 |
| summary | 29.7/30 | 9.7 | 87.6% | $0.069 |
| **layered (default)** | **29.7/30** | **2.0** (+ ~20 tool-result clears) | 82.7% | **$0.066** |
| layered, clear at 30% | 30/30 | 8.7 | 88.5% | $0.067 |

- Layered keeps as much as summarizing every time, with about 80% fewer model summaries;
  most compactions only clear tool results. Total cost is about the same (4% lower): the
  clears cost some cache hits.
- Clearing at 30% never happens: with 20% of the workspace protected, clearing cannot bring
  the request below 20%, so that setting behaves like plain summarizing.
- The 64k window makes each file read about 8% of the workspace, four times its share in a
  200k workspace, so clears here are more frequent than in normal use.

**Previous design** (80% of the model's input window, compact to 55%), run to measure
hysteresis; the rows below are from that version:

| strategy | runs | facts kept | LLM summaries | cached input | cost/run |
|---|---|---|---|---|---|
| truncate | 2 | 6.0/30 | 0 | 92.8% | $0.057 |
| summary | 3 | 30/30 | 7.0 | 90.6% | $0.072 |
| layered, no hysteresis | 3 | 17.3/30 | 2.0 (+ ~7 extractive) | 89.9% | $0.063 |
| layered (compact to 55%) | 3 | 30/30 | 5.3 | 89.6% | $0.075 |

- Dropping old turns loses most of what the user said (20% kept).
- Hysteresis is what keeps the layered strategy lossless: compacting only to just under the
  trigger compresses often and a little at a time, the slices fall below the size worth a
  model summary, and the extractive fallback loses facts (58% kept).
- Layered and plain summarization keep everything; layered needs a quarter fewer model
  summaries, at about the same total cost (the requests it leaves are a little larger).
- No run answered a changed fact with its old value.

### Experiment 2: parallel workers

| workers | runs | success | time | saved vs 1 | cost/run |
|---|---|---|---|---|---|
| 1 | 9 | 100% | 254 s | – | $0.0105 |
| 2 | 9 | 100% | 151 s | 39% | $0.0111 |
| 4 | 9 | 100% | 111 s | 56% | $0.0111 |
| 6 | 9 | 100% | 87 s | 64% | $0.0096 |

Cost does not depend on the worker count (the same steps run either way); wall time drops
until the workers outnumber the steps that can run at once. From 4 to 6 adds little, so the
default went from 2 to 4 workers.

### Experiment 3: models

| model | success | cost per solved task | time |
|---|---|---|---|
| DeepSeek V4 Flash | 90% | $0.0022 | 70 s |
| GPT-6 Luna | 100% | $0.0047 | 68 s |
| DeepSeek V4.1 Flash | 100% | $0.0053 | 91 s |
| DeepSeek V4 Pro 0813 | 100% | $0.0105 | 55 s |
| Kimi K2.7 Code | 80% | $0.0342 | 36 s |
| GPT-6 Sol | 100% | $0.0504 | 62 s |
| Gemini 3.8 Flash | 60% | $0.1723 | 132 s |
| GPT-6 Astra | 100% | $0.2849 | 77 s |
| Claude Opus 5.5 | 100% | $0.4478 | 82 s |
| GLM-5.3 Flash | 70% | $0.0054 | 62 s |
| Qwen3.8 Flash | 67% | $0.0071 | 117 s |

The four easier tasks separate the models only by cost and speed; `hardkit` separates them by
success. On this set GPT-6 Luna solves everything the strongest models do at about 1% of Claude
Opus 5.5's cost per solved task, which is why it is the fast tier. Qwen3.8 Flash hit rate limits
on six runs, left out above. This table was run before the fix for truncated output (below), so
the `hardkit` failures of GLM-5.3 Flash, Qwen3.8 Flash, and DeepSeek V4 Flash include turns that
ended silently when their output was cut off.

### Experiment 4: long-term memory recall

80 memories of a made-up shop backend and 119 questions labelled with the memories that
answer them (`memory_dataset.py`): 40 in the memory's own words, 40 in other words, 15 in
English or abbreviations, 4 with several answers, and 20 that nothing answers (some share
words with memories on purpose). Keywords were written once by GPT-6 Luna with the same
instruction `save_memory` gives the agent, without seeing the questions (about $0.002);
everything else runs offline.

| method | recall@3 | MRR | English / abbreviations |
|---|---|---|---|
| previous scorer (overlap + weighted importance) | 0.833 | 0.762 | 0.40 |
| BM25 | 0.889 | 0.851 | 0.47 |
| **BM25 + saved keywords** | **0.960** | **0.924** | **1.00** |

What the model is handed with each request (tier 2):

| policy | precision | memories per request | unrelated requests given a memory |
|---|---|---|---|
| previous: top 6 of anything overlapping | 15% | 5.25 | 100% |
| **top 3, coverage >= 0.55** | **91%** | **0.55** | **5%** |

- Keywords written at save time close the gap for other words and other languages at no
  extra model call.
- Importance and recency barely change the order (MRR 0.919 with neither, 0.924 with the
  best nudge, importance 0.05 and recency 0.02), so relevance ranks and they break ties.
- Tier 2's bar was chosen by F0.5 (precision first) and tier 3's by F2 (recall first) over
  a sweep from 0 to 0.8. Tier 2 then recalls 56% of relevant memories itself; the index in
  the system prompt and `search_memory` (79% at its bar) cover the rest.
- The labels are mine and were spot-checked by a second person, who changed none.

### Defects the experiments found

- **Compaction loop.** After compaction started keeping as much recent history as fits the
  target, stubbed tool results made messages cheap in tokens, so the kept part sat at the
  message-count limit and every following call compacted a sliver again (68 compactions in
  one run, cost doubled, 6 facts lost). The message count now gets the same hysteresis.
- **Silent stop on truncated output.** A model that spent its whole output budget (reasoning,
  or a tool call writing six files at once) ended the turn with nothing, and the agent took
  that as done: 0/6 on `hardkit` with every file untouched. A cut-off reply now runs the tool
  calls that arrived whole, drops the cut-off one, and asks the model to continue in smaller
  steps. DeepSeek V4 Flash went from 1 of 2 to 2 of 2 on `hardkit`; GLM-5.3 Flash no longer
  stops, though it then ran out of the 30-turn budget.
- **No retry on rate limits.** HTTP 429 and upstream 5xx ended the turn; they are now retried
  up to three times before any output, honoring Retry-After.
