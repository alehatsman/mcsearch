#!/usr/bin/env bash
# Setup git hooks for dex development.
#
#   pre-commit → mooncake task ci-fast
#     Fast gate (<5s on warm cache): go vet, gofmt on staged files,
#     ai-lint on staged files. Catches cheap mistakes before they land.
#
#   pre-push → mooncake task ci
#     Full gate: build + test + lint + vuln + arch-snapshot + budget + dupl.
#
# Bypass: `git commit --no-verify` or `git push --no-verify`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$(git -C "$REPO_ROOT" rev-parse --git-path hooks)"
case "$HOOKS_DIR" in
  /*) ;;
  *) HOOKS_DIR="$REPO_ROOT/$HOOKS_DIR" ;;
esac

mkdir -p "$HOOKS_DIR"
echo "Setting up dex development hooks in: $HOOKS_DIR"

# ----- pre-commit ------------------------------------------------------------
cat > "$HOOKS_DIR/pre-commit" << 'HOOK_EOF'
#!/usr/bin/env bash
# Fast gate before the commit lands. Bypass: git commit --no-verify.
set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! command -v mooncake >/dev/null 2>&1; then
  echo "pre-commit: 'mooncake' is required but not installed." >&2
  echo "            See: https://github.com/alehatsman/mooncake" >&2
  exit 1
fi

echo "pre-commit: running 'mooncake task ci-fast' (vet + gofmt + ai-lint)..."
if ! mooncake task ci-fast; then
  echo "" >&2
  echo "pre-commit: ✗ fast gate failed. Fix the issue above and re-commit," >&2
  echo "            or 'git commit --no-verify' to bypass (not recommended)." >&2
  exit 1
fi
HOOK_EOF
chmod +x "$HOOKS_DIR/pre-commit"
echo "  ✓ pre-commit → mooncake task ci-fast"

# ----- pre-push --------------------------------------------------------------
cat > "$HOOKS_DIR/pre-push" << 'HOOK_EOF'
#!/usr/bin/env bash
# Full gate before the push leaves the machine. Bypass: git push --no-verify.
set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! command -v mooncake >/dev/null 2>&1; then
  echo "pre-push: 'mooncake' is required but not installed." >&2
  echo "          See: https://github.com/alehatsman/mooncake" >&2
  exit 1
fi

echo "pre-push: running 'mooncake task ci' (build + test + lint + vuln + arch + budget + dupl)..."
if ! mooncake task ci; then
  echo "" >&2
  echo "pre-push: ✗ full gate failed. Fix the issue above and re-push," >&2
  echo "          or 'git push --no-verify' to bypass (not recommended)." >&2
  exit 1
fi
HOOK_EOF
chmod +x "$HOOKS_DIR/pre-push"
echo "  ✓ pre-push → mooncake task ci"

cat <<EOM

Installed:
  pre-commit  → 'mooncake task ci-fast' (~seconds). Catches stub panics, agent
                TODOs, and unformatted code before they land in a commit.
  pre-push    → 'mooncake task ci'      (~1 min). Full build + test + lint + vuln
                + arch-snapshot + dupl before commits leave the machine.

Bypass either with --no-verify when you really need it.

If 'mooncake' isn't on PATH yet, install it from:
  https://github.com/alehatsman/mooncake
Then:
  mooncake task install-tools
EOM
