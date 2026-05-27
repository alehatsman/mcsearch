#!/usr/bin/env bash
# Build a doc-stripped copy of the dex repo for the native baseline.
#
# Native mode must not benefit from in-repo docs that dex itself generated
# (LLM_GUIDE.md) or that constitute hand-written codebase explanation
# (PIPELINE.md, STORAGE.md, README.md, docs/). Stripping them means native
# mode is judged on its ability to derive understanding from *source* alone.
#
# Code-level comments in *.go files are kept — they are part of the code.
#
# Usage:
#   benchmark/scripts/make-native-fixture.sh [dest]
# Default dest: /tmp/dex-bench-native

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-/tmp/dex-bench-native}"

echo "src:  $ROOT"
echo "dest: $DEST"

if [[ -e "$DEST" ]]; then
  read -r -p "destination exists — remove it? [y/N] " ans
  case "$ans" in
    y|Y) rm -rf "$DEST" ;;
    *) echo "aborted"; exit 1 ;;
  esac
fi

mkdir -p "$DEST"

# Copy code only — drop .git (no log/blame leakage), benchmark/ (test fixtures
# and ground truth), and the build cache.
rsync -a \
  --exclude='.git/' \
  --exclude='.claude/' \
  --exclude='.dex/' \
  --exclude='benchmark/' \
  --exclude='/dex' \
  --exclude='*.test' \
  "$ROOT"/ "$DEST"/

# Strip top-level explanation docs.
rm -f "$DEST/README.md"
rm -f "$DEST/LLM_GUIDE.md"
rm -f "$DEST/PIPELINE.md"
rm -f "$DEST/STORAGE.md"

# Strip docs/ directory entirely.
rm -rf "$DEST/docs"

# Keep: cmd/, internal/, scripts/, go.mod, go.sum, Dockerfile, tasks.yml, LICENSE.
echo
echo "Stripped fixture contents:"
ls -la "$DEST"
echo
echo "Any *.md leftover at root (should be empty):"
find "$DEST" -maxdepth 1 -name '*.md' -print
echo "Any docs/ leftover (should be empty):"
find "$DEST" -maxdepth 1 -name 'docs' -print
