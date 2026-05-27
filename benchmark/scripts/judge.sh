#!/usr/bin/env bash
# Judge every run JSON in a results dir. Writes a score next to each.
#
# Usage:
#   benchmark/scripts/judge.sh <results-dir>
#
# For each runs/*.json, extracts the agent's final answer, the question and
# the ground truth, sends them to a judge claude session, and writes the
# parsed {score,reason} JSON to runs/<base>.judged.json.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="$ROOT/benchmark"
JUDGE_MODEL="${JUDGE_MODEL:-sonnet}"
RUBRIC="$BENCH/judge/rubric.md"
export BENCH

RDIR="${1:-}"
if [[ -z "$RDIR" || ! -d "$RDIR/runs" ]]; then
  echo "usage: $0 <results-dir>  (must contain runs/)" >&2; exit 2
fi

command -v claude >/dev/null || { echo "claude CLI required" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq required" >&2; exit 1; }

judge_one() {
  local f="$1"
  local base="${f%.json}"
  local judged="${base}.judged.json"
  [[ -f "$judged" ]] && { echo "skip (already judged): $f"; return 0; }

  local qid question gt answer
  qid=$(jq -r '.qid' "$f")
  # Prefer the *current* questions.yml ground truth over the snapshot embedded
  # in the run JSON — so edits to the question set take effect on re-judge
  # without re-running the agent.
  question=$(python3 "$BENCH/scripts/_qload.py" get "$qid" question)
  gt=$(python3 "$BENCH/scripts/_qload.py" get "$qid" ground_truth)
  if [[ -z "$question" || -z "$gt" ]]; then
    question=$(jq -r '.question' "$f")
    gt=$(jq -r '.ground_truth' "$f")
  fi
  answer=$(jq -r '.claude.result // .claude.message // ""' "$f")

  if [[ -z "$answer" ]]; then
    jq -n --arg reason "empty answer" '{score:0,reason:$reason}' > "$judged"
    echo "  empty: $(basename "$f") -> 0"
    return 0
  fi

  # Build the judge prompt.
  local prompt
  prompt=$(jq -n \
    --arg q "$question" --arg gt "$gt" --arg a "$answer" \
    '"QUESTION:\n" + $q + "\n\nGROUND_TRUTH:\n" + $gt + "\n\nANSWER:\n" + $a + "\n\nReply with strict JSON: {\"score\":0|1|2,\"reason\":\"...\"}"')
  # Strip the outer JSON quotes jq added.
  prompt=$(echo "$prompt" | jq -r '.')

  local raw
  raw=$(claude \
    --print \
    --model "$JUDGE_MODEL" \
    --setting-sources "" \
    --disable-slash-commands \
    --append-system-prompt "$(cat "$RUBRIC")" \
    --allowed-tools "" \
    --no-session-persistence \
    --permission-mode bypassPermissions \
    --output-format json \
    "$prompt" 2>/dev/null | jq -r '.result // .message // ""')

  # Parse the strict JSON. Be defensive — strip code fences if present.
  local cleaned
  cleaned=$(echo "$raw" | sed -E 's/^```(json)?//; s/```$//' | tr -d '\r')
  if ! echo "$cleaned" | jq -e '.score' >/dev/null 2>&1; then
    jq -n --arg raw "$raw" '{score:null, reason:"parse_failed", raw:$raw}' > "$judged"
    echo "  parse_failed: $(basename "$f")"
    return 0
  fi
  echo "$cleaned" | jq '.' > "$judged"
  local s
  s=$(jq -r '.score' "$judged")
  echo "  judged: $(basename "$f") -> $s"
}

for f in "$RDIR"/runs/*.json; do
  # Skip already-judged files.
  case "$f" in *.judged.json) continue ;; esac
  judge_one "$f"
done

echo "done. judged files written next to runs."
