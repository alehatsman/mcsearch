# Findings — run 20260527T110024Z (native fixture stripped, ground truth corrected)

Model: `sonnet` · Replicates: N=1 · Git SHA: `045b8719`
Dex root: `/home/aleh/projects/dex` (original repo, LLM_GUIDE.md & all docs intact)
Native root: `/tmp/dex-bench-native` (doc-stripped: no README.md, LLM_GUIDE.md, PIPELINE.md, STORAGE.md, docs/)

## What changed in this version

This is the same run as before, but two questions were corrected after manual
review of the dex transcripts revealed problems in *my* ground truth:

1. **L4-05 (concurrency model / flock).** The original ground truth claimed
   that both `dex index` and `dex watch` acquire the per-project flock.
   That's what the `internal/lock` package's *doc-comment* promises, but
   `cmd/dex/main.go:1768` (`cmdWatch`) does NOT actually call
   `acquireProjectLock`. Dex's answer caught this; my ground truth didn't.
   Ground truth rewritten to describe what the code actually does.
2. **L1-02 (MCP stdio "entry point").** The question was ambiguous — both
   the CLI dispatch in `cmd/dex/main.go:1961` (`cmdMCP`) and the library
   `RunStdio` function at `internal/mcp/server.go:851` qualify as
   "entry point" at different levels of the stack. Tightened the question
   to ask specifically for the function definition.

Both were re-judged against the updated ground truth/question. The judge
script was also patched to load ground truth from `questions.yml` directly
rather than from the snapshot embedded in each run JSON, so future edits
to the question set take effect on re-judge without re-running the agent.

## Headline

**dex 1.92 / native 1.72** (out of 2). 23 vs 18 perfect scores out of 25.

| Metric | Run 1 (docs intact) | Run 2 (stripped, raw) | Run 2 (corrected) |
|---|---:|---:|---:|
| dex score | 1.84 | 1.80 | **1.92** |
| native score | 1.80 | 1.68 | **1.72** |
| Δ (dex − native) | +0.04 | +0.12 | **+0.20** |
| dex $/Q | 0.098 | 0.078 | 0.079 |
| native $/Q | 0.091 | 0.089 | 0.089 |

The 0.08-point bump for dex between "stripped raw" and "stripped corrected"
came entirely from those two ground-truth bugs above. The native bump
(0.04) came from L1-02 alone.

## Per band

| Band | dex | native | Δ | Verdict |
|---|---:|---:|---:|---|
| L1 symbol lookup | 2.0 | 1.8 | +0.2 | dex slight |
| L2 package summary | 2.0 | 2.0 | 0.0 | tie |
| L3 cross-package | 2.0 | 2.0 | 0.0 | tie |
| L4 architectural | 2.0 | 1.6 | +0.4 | **dex wins** |
| L5 refactor blast | 1.6 | 1.2 | +0.4 | **dex wins** |

## Where dex wins (after correction)

| QID | Theme | dex | native |
|---|---|---:|---:|
| L1-03 | RRF constant (rrfK=60) | 2 | 1 |
| L4-03 | nanosecond timestamps & PruneUnseen | 2 | 1 |
| L4-04 | why triggers maintain chunks_fts/chunk_vecs | 2 | 1 |
| L4-05 | concurrency model (caught real impl gap) | 2 | 2 |
| L5-03 | adding a new MCP tool: which files | 2 | 1 |
| L5-04 | debounce default 500ms blast-radius | 2 | 1 |

## Real bug uncovered by L4-05

Dex's answer to L4-05 was *more correct than the human-written ground truth*:
it correctly observed that `cmdWatch` at `cmd/dex/main.go:1768` never calls
`acquireProjectLock`, despite the `internal/lock/lock.go:1-5` doc-comment
explicitly listing `dex watch` as something the lock protects. The
`Holder.Command` field at `lock.go:39` even includes `"watch"` as a valid
value, but no code path ever sets it.

**Impact:** two concurrent `dex watch` processes on the same project — or
`dex watch` running while `dex index` is running — are serialized only by
SQLite's writer lock, not by the per-project flock. The flock package's
documented invariant is silently violated.

**Fix candidate:** `cmdWatch` should call `acquireProjectLock(ctx, p, "watch",
"chunk", *waitLock, *breakLock)` near line 1796 (right after
`p.EnsureCacheDir()`, before any indexing starts), mirroring `cmdIndex`.

## Methodology lessons

1. **Ground truth is the load-bearing component.** When the benchmark is
   trying to grade an LLM's understanding of a codebase, *the questioner's
   own understanding* is the floor. Two of my 25 ground-truth answers were
   either wrong or ambiguous — an 8% bug rate. Spot-checking the dex
   transcripts caught both; just trusting the judge would have missed them.
2. **Judge-explains-why is the signal.** Every judge verdict has a `reason`
   field. When dex got 0 on L4-05 the reason said *"directly contradicts
   the ground truth"* — that's the cue to read the transcript and check
   *which* of the two is wrong. Without that field the bug would have hid
   in noise.
3. **Embedding ground truth in run JSON was wrong.** I baked the
   ground_truth into every run JSON at run time, so editing `questions.yml`
   later had no effect on re-judging. Fixed in `judge.sh`.

## Honest remaining caveats

1. **N=1** — per-question scores are still noisy. L5-01 / L5-02 (both at
   score 1 for both modes) would benefit from N=3 to see whether dex's
   `ask` is actually flailing or just unlucky.
2. **`tool_calls` still 0** — needs `--output-format stream-json` to count.
3. **Same model judges siblings** — possible self-bias.
4. **Cache effects** — dex mode caches `LLM_GUIDE.md` once and amortizes
   across questions; native fixture has no shared large doc. Real cold-cache
   numbers would widen the per-question cost gap.

## Bottom line

When native cannot lean on pre-written codebase docs, **dex wins on L4
(+0.4) and L5 (+0.4)** — the two bands that exercise architectural
reasoning and refactor scope. L1–L3 are ties because grep handles them
fine once you know what to grep for.

The most interesting outcome wasn't the score delta. It was that dex's
answer to L4-05 caught a real implementation gap (cmdWatch missing the
flock) that the human (me) had assumed worked from reading the package
doc-comment alone.
