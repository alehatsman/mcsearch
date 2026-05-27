#!/usr/bin/env python3
"""Aggregate a benchmark results directory into CSV + markdown summary.

Usage:
  report.py <results-dir>

Writes:
  <results-dir>/runs.csv      — one row per (qid, mode, rep)
  <results-dir>/summary.md    — aggregate cost/quality/effort per mode and band

Reads the wrapper JSON written by run.sh (which embeds the full claude
--output-format json payload under .claude). Pulls token usage from
.claude.usage if present, falls back to total_cost_usd. Tool-call count
comes from counting non-text content blocks in .claude.messages if present,
otherwise 0.
"""
import csv, json, pathlib, statistics, sys, collections

if len(sys.argv) != 2:
    sys.exit("usage: report.py <results-dir>")

RDIR = pathlib.Path(sys.argv[1]).resolve()
RUNS = RDIR / "runs"
if not RUNS.is_dir():
    sys.exit(f"no runs/ in {RDIR}")

rows = []
for f in sorted(RUNS.glob("*.json")):
    if f.name.endswith(".judged.json"):
        continue
    try:
        run = json.loads(f.read_text())
    except Exception as e:
        print(f"skip {f.name}: {e}", file=sys.stderr); continue

    qid  = run.get("qid")
    mode = run.get("mode")
    rep  = run.get("rep")
    wall = run.get("wall_seconds")
    rc   = run.get("rc")
    band = qid.split("-")[0] if qid else ""

    claude = run.get("claude") or {}
    usage  = claude.get("usage") or {}
    in_tok  = usage.get("input_tokens",  0) or 0
    out_tok = usage.get("output_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_create = usage.get("cache_creation_input_tokens", 0) or 0
    cost_usd = claude.get("total_cost_usd", None)
    num_turns = claude.get("num_turns", None)

    # Count tool uses from messages if available.
    tool_calls = 0
    for m in (claude.get("messages") or []):
        for c in (m.get("content") or []):
            if isinstance(c, dict) and c.get("type") == "tool_use":
                tool_calls += 1

    # Judged score (may not exist yet).
    judged_path = f.with_suffix("").with_suffix(".judged.json")
    judged_path = f.parent / (f.stem + ".judged.json")
    score = None; judge_reason = ""
    if judged_path.exists():
        try:
            j = json.loads(judged_path.read_text())
            score = j.get("score")
            judge_reason = (j.get("reason") or "")[:200]
        except Exception:
            pass

    rows.append({
        "qid": qid, "band": band, "mode": mode, "rep": rep,
        "rc": rc, "wall_seconds": wall,
        "input_tokens": in_tok, "output_tokens": out_tok,
        "cache_read_tokens": cache_read, "cache_create_tokens": cache_create,
        "cost_usd": cost_usd, "num_turns": num_turns,
        "tool_calls": tool_calls,
        "score": score, "judge_reason": judge_reason,
    })

# Write CSV.
csv_path = RDIR / "runs.csv"
fields = ["qid","band","mode","rep","rc","wall_seconds",
          "input_tokens","output_tokens","cache_read_tokens","cache_create_tokens",
          "cost_usd","num_turns","tool_calls","score","judge_reason"]
with csv_path.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"wrote {csv_path}  ({len(rows)} rows)")

# Aggregate.
def agg(rs, key):
    vals = [r[key] for r in rs if r.get(key) is not None]
    if not vals: return ("-", "-")
    try:
        return (round(statistics.mean(vals), 3),
                round(statistics.median(vals), 3))
    except statistics.StatisticsError:
        return ("-", "-")

def score_stats(rs):
    scored = [r["score"] for r in rs if r.get("score") is not None]
    if not scored: return ("-", "-", 0)
    avg = round(sum(scored)/len(scored), 2)
    n_correct = sum(1 for s in scored if s == 2)
    return (avg, f"{n_correct}/{len(scored)}", len(scored))

by_mode      = collections.defaultdict(list)
by_mode_band = collections.defaultdict(list)
for r in rows:
    by_mode[r["mode"]].append(r)
    by_mode_band[(r["mode"], r["band"])].append(r)

out = []
out.append("# Benchmark summary\n")
out.append(f"Results dir: `{RDIR}`\n")
manifest = (RDIR / "manifest.json")
if manifest.exists():
    out.append("\n```\n" + manifest.read_text().strip() + "\n```\n")

out.append("\n## Overall (per mode)\n")
out.append("| Mode | N | Avg score | Correct (2/N) | Avg in tok | Avg out tok | Avg cache-read | Avg tool calls | Avg wall (s) | Avg $ |")
out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for mode in sorted(by_mode):
    rs = by_mode[mode]
    avg_s, frac, _ = score_stats(rs)
    in_avg,_   = agg(rs, "input_tokens")
    out_avg,_  = agg(rs, "output_tokens")
    cache_avg,_= agg(rs, "cache_read_tokens")
    tc_avg,_   = agg(rs, "tool_calls")
    wall_avg,_ = agg(rs, "wall_seconds")
    cost_avg,_ = agg(rs, "cost_usd")
    out.append(f"| {mode} | {len(rs)} | {avg_s} | {frac} | {in_avg} | {out_avg} | {cache_avg} | {tc_avg} | {wall_avg} | {cost_avg} |")

out.append("\n## Per band × mode\n")
out.append("| Band | Mode | N | Avg score | Avg in tok | Avg out tok | Avg tool calls | Avg wall (s) |")
out.append("|---|---|---:|---:|---:|---:|---:|---:|")
bands = sorted({b for _,b in by_mode_band})
for band in bands:
    for mode in sorted(by_mode):
        rs = by_mode_band.get((mode, band), [])
        if not rs: continue
        avg_s, frac, _ = score_stats(rs)
        in_avg,_   = agg(rs, "input_tokens")
        out_avg,_  = agg(rs, "output_tokens")
        tc_avg,_   = agg(rs, "tool_calls")
        wall_avg,_ = agg(rs, "wall_seconds")
        out.append(f"| {band} | {mode} | {len(rs)} | {avg_s} | {in_avg} | {out_avg} | {tc_avg} | {wall_avg} |")

# Head-to-head per question (averaged across reps).
out.append("\n## Per-question (dex vs native, averaged across reps)\n")
out.append("| QID | Band | dex score | nat score | dex tok | nat tok | dex calls | nat calls |")
out.append("|---|---|---:|---:|---:|---:|---:|---:|")
qids = sorted({r["qid"] for r in rows})
for qid in qids:
    drs = [r for r in rows if r["qid"]==qid and r["mode"]=="dex"]
    nrs = [r for r in rows if r["qid"]==qid and r["mode"]=="native"]
    band = drs[0]["band"] if drs else (nrs[0]["band"] if nrs else "")
    def avg_score(rs):
        s = [r["score"] for r in rs if r.get("score") is not None]
        return round(sum(s)/len(s),2) if s else "-"
    def avg_tok(rs):
        t = [(r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in rs]
        return round(sum(t)/len(t)) if t else "-"
    def avg_calls(rs):
        c = [r["tool_calls"] for r in rs]
        return round(sum(c)/len(c),1) if c else "-"
    out.append(f"| {qid} | {band} | {avg_score(drs)} | {avg_score(nrs)} | {avg_tok(drs)} | {avg_tok(nrs)} | {avg_calls(drs)} | {avg_calls(nrs)} |")

md_path = RDIR / "summary.md"
md_path.write_text("\n".join(out) + "\n")
print(f"wrote {md_path}")
