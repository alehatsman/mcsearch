# dex benchmark

Measures whether the `dex` index + `LLM_GUIDE.md` actually help Claude answer
questions about a codebase, vs. native exploration with `Read`/`Grep`/`Glob`/`Bash`.

The benchmark dogfoods on this very repo.

## What gets measured

Three axes, two modes (`dex` and `native`), per question:

| Axis    | How                                                                                   |
|---------|---------------------------------------------------------------------------------------|
| Quality | LLM-as-judge scores the answer against hand-curated ground truth, 0 / 1 / 2           |
| Cost    | Input + output tokens reported by the Anthropic API (claude `--output-format json`)   |
| Effort  | Tool-call count + wall-clock seconds                                                  |

## Layout

```
benchmark/
  README.md                  this file
  questions/questions.yml    25 questions × 5 difficulty bands (L1..L5) + ground truth
  configs/
    mcp-dex.json             MCP config attaching the local dex stdio server
    mcp-none.json            empty MCP config — pure native baseline
    system-dex.md            system prompt appended in dex mode
    system-native.md         system prompt appended in native mode
  judge/rubric.md            judge instructions (0/1/2 scale)
  scripts/
    _qload.py                tiny yaml->stdout helper
    run.sh                   runs questions × modes × N replicates → results/<ts>/runs/*.json
    judge.sh                 scores each run JSON → runs/<base>.judged.json
    report.py                aggregates runs+judgments → runs.csv + summary.md
  results/<timestamp>/       one dir per benchmark execution
```

## Question bands

| Band | Theme                          | What dex *should* be good at                    |
|------|--------------------------------|-------------------------------------------------|
| L1   | Symbol lookup (file:line)      | symbol index, FTS                                |
| L2   | Single-package summary         | per-package summaries in `LLM_GUIDE.md`          |
| L3   | Cross-package data flow        | hybrid retrieval over chunks                     |
| L4   | Architectural trace            | repo summary + graph + summaries                 |
| L5   | Refactor blast-radius          | call graph + symbol→file mapping                 |

If a band has roughly equal scores between modes, dex is offering no advantage
on that class of question.

## Prerequisites

- `claude` CLI on PATH (tested with 2.1.x)
- `jq`
- `python3` with `pyyaml`
- A built `dex` binary at `/home/aleh/.local/bin/dex` (or adjust `configs/mcp-dex.json`)
- The dex index for this repo already populated (`dex index` once before running)
- Self-hosted embed/chat endpoints up (or stub them in the config) so the
  dex MCP server's `ask` tool works

## Run it

```bash
# Full sweep: every question × both modes × 1 replicate
benchmark/scripts/run.sh -n 1

# Single question, both modes, 3 replicates (noise check)
benchmark/scripts/run.sh -q L3-04 -n 3

# Single mode only
benchmark/scripts/run.sh -m native -n 1

# Judge the latest results dir
benchmark/scripts/judge.sh benchmark/results/<timestamp>

# Aggregate
python3 benchmark/scripts/report.py benchmark/results/<timestamp>
# -> runs.csv + summary.md in that dir
```

## How fair is the comparison

Things controlled:

- Both modes use `claude --bare`: no CLAUDE.md auto-discovery, no hooks, no
  plugin sync, no auto-memory. The only system-prompt content is what we
  append explicitly from `configs/system-*.md`.
- `--strict-mcp-config` ensures `native` mode has zero MCP servers attached
  (not just dex turned off).
- `--allowed-tools` whitelists exactly the tool set per mode: dex mode gets
  the seven `mcp__dex__*` tools + `Read`; native mode gets `Read Grep Glob Bash`.
- Same model in both modes (default `sonnet`, override with `MODEL=...`).
- Same prompt phrasing per question.

Things deliberately *not* controlled (and why):

- The `LLM_GUIDE.md` file lives in the repo on disk, so `native` mode could
  technically `Read` it. That's a fair representation of reality — if dex
  publishes the guide into the repo, anyone can read it. The dex-mode system
  prompt explicitly tells the agent the guide exists; the native-mode prompt
  does not. If you want a stricter baseline, delete `LLM_GUIDE.md` before
  running native mode (and remember to restore it).
- Prompt caching is left on (CLI default). Cost numbers will under-count for
  the second-and-later replicate of the same question. Compare cold runs
  (first replicate) for token cost; later replicates are useful for tool-call
  / quality variance only.
- The dex index must be warm. Cold-start indexing cost is one-off and
  reported separately (see "Indexing cost" below) — not folded into per-query
  cost, because the user only pays it once per repo.

## Indexing cost (one-off, report separately)

```bash
time dex index /home/aleh/projects/dex
du -sh "$(dex paths cache /home/aleh/projects/dex 2>/dev/null || echo "$HOME/.cache/dex")/*"
```

Record this once per run-set in `summary.md` manually; it's the price of
admission for dex mode but is paid once, not per query.

## Threats to validity

1. **Ground truth drift.** Questions reference specific `file:line` locations.
   If the repo refactors, regenerate truth before re-running. Hash-pin in
   `manifest.json` via the recorded `git_sha`.
2. **Selection bias.** Author wrote both the questions and the dex tool. The
   5-band stratification + L2/L4 (which interrogate things `LLM_GUIDE.md` does
   *not* cover well) is the defense; if dex wins uniformly it's suspicious.
3. **Single judge.** LLM-as-judge has its own bias. Spot-check 5–10 judged
   answers by hand the first time you run. If you disagree with the judge
   often, tighten the rubric.
4. **N=1 noise.** First pass is N=1 to get a smoke-signal cheaply. Bump to
   N≥3 for the questions where modes diverge most before drawing conclusions.

## Interpreting results

Look at `summary.md` for three things:

- **Per-mode totals**: did dex move the average score up at all?
- **Per-band table**: which question classes is dex actually helping with?
  L1 (symbol lookup) should be a near-tie — grep is fine. L4/L5 (architectural
  / refactor blast-radius) is where dex *should* dominate; if it doesn't,
  something is wrong with the index or the prompt.
- **Cost per quality point**: dex mode is "worth it" if the score increase
  more than compensates for the extra context dragged in by `LLM_GUIDE.md`.

If quality is equal but token cost is higher in dex mode, dex is not pulling
its weight. If quality is higher but tool-call count is also higher, dex's
benefit comes from more interactions, not better-per-call retrieval —
investigate whether `ask` is being used vs. agents flailing on the lower-level
legs.
