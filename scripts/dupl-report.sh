#!/usr/bin/env bash
# Code-duplication report (production code only).
#
# Usage:
#   scripts/dupl-report.sh           # manual mode, threshold 100
#   scripts/dupl-report.sh --ci      # CI mode (graceful skip if dupl missing, 2-space indent)
#
# Override threshold with T env var: T=50 scripts/dupl-report.sh
#
# Pipes `dupl -plumbing` through a Python deduper that pairs each
# clone with its mate and reports them sorted by clone length.
set -euo pipefail

CI_MODE=0
[ "${1:-}" = "--ci" ] && CI_MODE=1

T="${T:-100}"

if ! command -v dupl >/dev/null 2>&1 && ! [ -x "$(go env GOPATH)/bin/dupl" ]; then
  if [ "$CI_MODE" = "1" ]; then
    echo "  dupl not installed — run 'mooncake task install-tools'. Skipping."
    exit 0
  fi
  echo "dupl not installed — run 'mooncake task install-tools'." >&2
  exit 1
fi

INDENT=""
[ "$CI_MODE" = "1" ] && INDENT="  "

dupl -threshold "$T" -plumbing . 2>&1 | grep -v "_test.go" | T="$T" INDENT="$INDENT" python3 -c '
import os, sys
T = os.environ.get("T", "100")
INDENT = os.environ.get("INDENT", "")
pairs = set()
for line in sys.stdin:
    parts = line.strip().split(": duplicate of ")
    if len(parts) != 2: continue
    pairs.add(tuple(sorted(parts)))
def n(rng):
    lo, hi = rng.split(":")[-1].split("-"); return int(hi) - int(lo) + 1
ranked = sorted(pairs, key=lambda p: -n(p[0]))
if not ranked:
    print(f"{INDENT}no production duplication at threshold {T}.")
else:
    print(f"{INDENT}{len(ranked)} production clone pair(s) at threshold {T}:")
    for a, b in ranked:
        print(f"{INDENT}  {n(a):>3}L  {a}  <-->  {b}")
'
