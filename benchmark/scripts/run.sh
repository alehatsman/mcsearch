#!/usr/bin/env bash
# Runs the benchmark: every question × every mode × N replicates.
#
# Usage:
#   benchmark/scripts/run.sh [-n N] [-m MODE] [-q ID] [-o OUTDIR]
#
# Defaults:
#   N=1, both modes, all questions, OUTDIR=benchmark/results/<timestamp>
#
# Output: one JSON per (question, mode, rep) in OUTDIR/runs/, plus a manifest.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="$ROOT/benchmark"

# Where each mode's claude process runs and is granted access to.
#   DEX mode    -> the real repo (with LLM_GUIDE.md, PIPELINE.md, etc. — that's the point).
#   NATIVE mode -> a doc-stripped fixture so native cannot read in-repo explanations.
#                  Build it with: benchmark/scripts/make-native-fixture.sh
DEX_ROOT="${DEX_ROOT:-$ROOT}"
NATIVE_ROOT="${NATIVE_ROOT:-/tmp/dex-bench-native}"

N=1
ONLY_MODE=""
ONLY_QID=""
OUTDIR=""
MODEL="${MODEL:-sonnet}"

while getopts "n:m:q:o:" opt; do
  case $opt in
    n) N=$OPTARG ;;
    m) ONLY_MODE=$OPTARG ;;
    q) ONLY_QID=$OPTARG ;;
    o) OUTDIR=$OPTARG ;;
    *) echo "usage: $0 [-n N] [-m dex|native] [-q QID] [-o OUTDIR]" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTDIR" ]]; then
  OUTDIR="$BENCH/results/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUTDIR/runs"
OUTDIR="$(cd "$OUTDIR" && pwd)"  # absolutize — we cd into per-mode workdirs below.

QFILE="$BENCH/questions/questions.yml"
if [[ ! -f "$QFILE" ]]; then
  echo "missing $QFILE" >&2; exit 1
fi

QLOAD="$BENCH/scripts/_qload.py"
command -v claude >/dev/null || { echo "claude CLI required" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 required" >&2; exit 1; }
python3 -c "import yaml" 2>/dev/null || { echo "pyyaml required (pip install pyyaml)" >&2; exit 1; }

# Snapshot manifest
{
  echo "{"
  echo "  \"model\": \"$MODEL\","
  echo "  \"replicates\": $N,"
  echo "  \"started\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"git_sha\": \"$(git -C "$ROOT" rev-parse HEAD)\","
  echo "  \"claude_version\": \"$(claude --version | awk '{print $1}')\","
  echo "  \"dex_root\": \"$DEX_ROOT\","
  echo "  \"native_root\": \"$NATIVE_ROOT\""
  echo "}"
} > "$OUTDIR/manifest.json"

QIDS=$(python3 "$QLOAD" ids)

modes=("dex" "native")
if [[ -n "$ONLY_MODE" ]]; then modes=("$ONLY_MODE"); fi

run_one() {
  local qid="$1" mode="$2" rep="$3"
  local question gt
  question=$(python3 "$QLOAD" get "$qid" question)
  gt=$(python3 "$QLOAD" get "$qid" ground_truth)
  local out="$OUTDIR/runs/${qid}__${mode}__r${rep}.json"

  local mcp_cfg sys_prompt allowed_tools workdir
  case "$mode" in
    dex)
      mcp_cfg="$BENCH/configs/mcp-dex.json"
      sys_prompt="$BENCH/configs/system-dex.md"
      # In dex mode: allow dex MCP tools + Read for verification, no Grep/Glob/Bash.
      allowed_tools="Read mcp__dex__ask mcp__dex__search_semantic mcp__dex__search_symbol mcp__dex__graph_callers mcp__dex__graph_callees mcp__dex__graph_deps mcp__dex__graph_neighbors mcp__dex__view_summarize"
      workdir="$DEX_ROOT"
      ;;
    native)
      mcp_cfg="$BENCH/configs/mcp-none.json"
      sys_prompt="$BENCH/configs/system-native.md"
      # In native mode: Read/Grep/Glob/Bash. No MCP tools.
      allowed_tools="Read Grep Glob Bash"
      workdir="$NATIVE_ROOT"
      if [[ ! -d "$workdir" ]]; then
        echo "native fixture missing: $workdir — run benchmark/scripts/make-native-fixture.sh" >&2
        return 1
      fi
      ;;
    *) echo "bad mode: $mode" >&2; return 1 ;;
  esac

  echo ">> $qid mode=$mode rep=$rep -> $out"

  local started_at finished_at
  started_at=$(date -u +%s)

  # --strict-mcp-config restricts MCP servers to just what we pass.
  # --setting-sources="" ignores user/project/local settings so neither mode
  #   inherits CLAUDE.md, hooks, or extra permissions.
  # --disable-slash-commands kills user/project skills so they can't leak.
  # --output-format json gives a single-object result with usage + cost.
  # --no-session-persistence keeps results clean.
  # --permission-mode bypassPermissions because we're scripted.
  set +e
  ( cd "$workdir" && claude \
      --print \
      --model "$MODEL" \
      --setting-sources "" \
      --disable-slash-commands \
      --append-system-prompt "$(cat "$sys_prompt")" \
      --mcp-config "$mcp_cfg" \
      --strict-mcp-config \
      --allowed-tools $allowed_tools \
      --add-dir "$workdir" \
      --no-session-persistence \
      --permission-mode bypassPermissions \
      --output-format json \
      "$question" \
      > "$out.tmp" 2> "$out.stderr" )
  local rc=$?
  set -e
  finished_at=$(date -u +%s)

  # Wrap the claude JSON result with our metadata.
  jq -n \
    --arg qid "$qid" --arg mode "$mode" --argjson rep "$rep" \
    --arg question "$question" --arg gt "$gt" \
    --argjson started "$started_at" --argjson finished "$finished_at" \
    --argjson rc "$rc" \
    --slurpfile claude "$out.tmp" \
    '{qid:$qid, mode:$mode, rep:$rep, question:$question, ground_truth:$gt,
      started:$started, finished:$finished, wall_seconds:($finished-$started),
      rc:$rc, claude:($claude[0])}' > "$out"
  rm -f "$out.tmp"
}

for qid in $QIDS; do
  if [[ -n "$ONLY_QID" && "$qid" != "$ONLY_QID" ]]; then continue; fi
  for mode in "${modes[@]}"; do
    for rep in $(seq 1 "$N"); do
      run_one "$qid" "$mode" "$rep"
    done
  done
done

echo "done. results in $OUTDIR"
