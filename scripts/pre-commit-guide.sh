#!/usr/bin/env bash
# Pre-commit hook: refresh the index and re-render LLM_GUIDE.md so the
# checked-in guide always reflects the current state of the codebase.
#
# Install:
#   ln -s ../../scripts/pre-commit-guide.sh .git/hooks/pre-commit
#
# Skip on a single commit:
#   DEX_SKIP_GUIDE=1 git commit ...
#
# Behaviour:
#   - `dex index` runs incrementally (mtime + content_sha1 fast paths),
#     so unchanged files cost ~nothing. Touched files re-summarize.
#   - `dex guide` is a pure read of the index — no LLM calls — and only
#     re-writes LLM_GUIDE.md when the manifest detects new summaries.
#   - If the guide changes, it is auto-staged into the current commit.

set -euo pipefail

if [[ "${DEX_SKIP_GUIDE:-0}" == "1" ]]; then
    exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"

if ! command -v dex >/dev/null 2>&1; then
    echo "pre-commit-guide: dex not on PATH; skipping" >&2
    exit 0
fi

dex index "$REPO_ROOT" --summarize >/dev/null
dex guide "$REPO_ROOT"

GUIDE="$REPO_ROOT/LLM_GUIDE.md"
MANIFEST="$REPO_ROOT/.dex/llm_guide.manifest.json"

if [[ -f "$GUIDE" ]] && ! git diff --quiet --cached -- "$GUIDE" 2>/dev/null; then
    : # already staged
elif [[ -f "$GUIDE" ]] && ! git diff --quiet -- "$GUIDE" 2>/dev/null; then
    git add "$GUIDE" "$MANIFEST" 2>/dev/null || git add "$GUIDE"
fi
