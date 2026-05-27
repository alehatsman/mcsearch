# Findings — run 20260527T102558Z

Model: `sonnet` · Replicates: N=1 · Git SHA: `045b8719` · Repo under test: dex itself

## Headline

**dex 1.84 / native 1.80** (out of 2). 21 vs 20 perfect scores out of 25. Very close.

## Per band — the real signal

| Band | dex | native | Verdict |
|---|---:|---:|---|
| L1 symbol lookup | 1.8 | 2.0 | native (grep is fine) |
| L2 package summary | 2.0 | 2.0 | tie |
| L3 cross-package | 2.0 | 2.0 | tie |
| **L4 architectural** | **2.0** | **1.6** | **dex (+0.4)** |
| L5 refactor blast | 1.4 | 1.4 | tie — both struggle |

Where dex actually pulled ahead was the architectural-reasoning band (L4) — specifically:

- **L4-02** (embedding dim locked at index creation): dex 2 / native 1
- **L4-03** (nanosecond timestamps & PruneUnseen): dex 2 / native 1

These are "why was it built this way" answers that live in `LLM_GUIDE.md` +
`PIPELINE.md`/`STORAGE.md` summaries. Native gets the gist from code+comments
but misses the detail.

Where dex *lost*:

- **L1-02** (`RunStdio` location): dex 1 / native 2
- **L5-02** (rrfK blast-radius): dex 1 / native 2

## Cost

~7% more per question in dex mode ($0.098 vs $0.091). The delta is mostly
`LLM_GUIDE.md` being pulled into context.

| Mode | Avg in tok (fresh) | Avg cache-read | Avg out tok | Avg $/Q | Avg wall (s) |
|---|---:|---:|---:|---:|---:|
| dex    |   9.12 | 95814 | 1045 | 0.098 | 26.96 |
| native | 247.80 | 77090 |  875 | 0.091 | 24.20 |

## Big caveat — this benchmark is generous to native mode

The dex repo has `README.md`, `PIPELINE.md`, `STORAGE.md`, and `LLM_GUIDE.md`
checked in at the root. Native mode reads those docs and effectively gets a
hand-written codebase guide for free.

On a less-documented repo (no `*.md` doc files at root), the gap should
widen — that is where dex's *generated* summaries would actually earn their
keep. **This run does not measure that case.**

## Known limitations of this run

1. **tool_calls is 0 across the board** — `--output-format json` (non-stream)
   doesn't emit the message stream, so the report parser cannot count tool
   uses. Need `--output-format stream-json` for that. Tactical fix.
2. **N=1** — per-question scores are noisy. Bumping to N=3 on L4/L5 deltas
   would tighten the signal.
3. **Prompt caching is on** — cost numbers are warm-cache numbers, not
   first-run cost.
4. **Same model judged its own siblings' answers** — sonnet judging sonnet.
   Could be biased toward its own phrasing. Should spot-check by hand or
   swap to a different judge model.

## Next steps (prioritized)

1. **Re-run on a less-documented repo.** The real test of dex's value. Pick
   a target with no `LLM_GUIDE.md`-equivalent file checked in. Or: temporarily
   delete `PIPELINE.md`/`STORAGE.md`/`README.md` for the native runs only.
2. **Fix the tool-call counter** by switching `run.sh` to
   `--output-format stream-json` and aggregating the stream, then re-run report.
3. **Bump N=3** for the questions where modes disagreed (L1-02, L4-02, L4-03,
   L5-02, L5-04) to confirm the deltas survive noise.
4. **Investigate L5 (refactor blast-radius).** Both modes at 1.4 — this is
   the band where graph traversal *should* matter most. If dex isn't winning
   here, either the graph is under-used by `ask`, the questions are bad, or
   the model isn't surfacing the graph results into its answer.

## Raw artifacts

- `summary.md` — auto-generated aggregate tables
- `runs.csv` — one row per (qid, mode, rep) with all numbers
- `runs/<qid>__<mode>__r<n>.json` — full claude --output-format json payload + harness metadata
- `runs/<qid>__<mode>__r<n>.judged.json` — judge verdict + reason
- `manifest.json` — model, N, started time, git SHA, claude version
