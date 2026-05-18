#!/usr/bin/env bash
# budget-status — print current state of code-quality soft caps.
#
# Soft caps:
#   1. gocyclo > 35 (non-test)  → refactor on next touch
#   2. God files > 500 LOC (non-test) — informational, not a hard fail
#
# Each section prints:  ✗ over cap   ⚠ within 20% of cap   ✓ clean
set -euo pipefail

CAP_GOCYCLO=35
CAP_GOD_LOC=500

cd "$(git rev-parse --show-toplevel)"

if [ -t 1 ]; then
  bold=$(tput bold 2>/dev/null || true)
  red=$(tput setaf 1 2>/dev/null || true)
  yellow=$(tput setaf 3 2>/dev/null || true)
  green=$(tput setaf 2 2>/dev/null || true)
  reset=$(tput sgr0 2>/dev/null || true)
else
  bold= red= yellow= green= reset=
fi

printf '%sCode quality soft caps — current state%s\n' "$bold" "$reset"
echo

# ---- 1. gocyclo --------------------------------------------------------------
printf '%s1. Non-test functions vs gocyclo cap %d%s\n' "$bold" "$CAP_GOCYCLO" "$reset"
if command -v gocyclo >/dev/null 2>&1 || [ -x "$(go env GOPATH)/bin/gocyclo" ]; then
  out=$(gocyclo -over "$CAP_GOCYCLO" . 2>/dev/null | grep -v '_test\.go' || true)
  if [ -n "$out" ]; then
    printf '%s\n' "$out" | awk -v r="$red" -v R="$reset" \
      '{ printf "   %s✗ gocyclo=%-3s %-30s %s%s\n", r, $1, $2"."$3, $4, R }'
  else
    printf '   %s✓ all functions under %d%s\n' "$green" "$CAP_GOCYCLO" "$reset"
  fi
else
  echo "   gocyclo not installed — run 'task install-tools'"
fi
echo

# ---- 2. God files ------------------------------------------------------------
printf '%s2. God files (non-test > %d LOC) — informational%s\n' "$bold" "$CAP_GOD_LOC" "$reset"
gods=$(find . -name '*.go' -not -name '*_test.go' \
       -not -path './vendor/*' -not -path './.git/*' \
       -exec wc -l {} + 2>/dev/null \
  | awk -v cap="$CAP_GOD_LOC" '$1 > cap && $2 != "total" {print}' \
  | sort -nr)
if [ -z "$gods" ]; then
  printf '   %s✓ no files over %d LOC%s\n' "$green" "$CAP_GOD_LOC" "$reset"
else
  printf '%s\n' "$gods" | awk -v y="$yellow" -v R="$reset" \
    '{ printf "   %s⚠ %5d LOC  %s%s\n", y, $1, $2, R }'
fi
