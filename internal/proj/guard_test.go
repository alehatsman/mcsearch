package proj

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCheckIndexableForceBypassesEverything(t *testing.T) {
	p := &Project{Root: "/"}
	if err := CheckIndexable(p, true); err != nil {
		t.Fatalf("force=true must bypass guards; got %v", err)
	}
}

func TestCheckIndexableHardDeny(t *testing.T) {
	for _, root := range []string{"/", "/etc", "/usr", "/var", "/home", "/Users", "/proc", "/tmp"} {
		err := CheckIndexable(&Project{Root: root}, false)
		if err == nil {
			t.Errorf("%s should be denied", root)
			continue
		}
		if !strings.Contains(err.Error(), "protected system path") {
			t.Errorf("%s denied for wrong reason: %v", root, err)
		}
	}
}

func TestCheckIndexableRejectsHomeDir(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		t.Skip("no $HOME available")
	}
	err = CheckIndexable(&Project{Root: home}, false)
	if err == nil || !strings.Contains(err.Error(), "home directory") {
		t.Fatalf("home dir should be denied; got %v", err)
	}
}

func TestCheckIndexableAcceptsGitWorkTree(t *testing.T) {
	t.Setenv(EnvAllowPaths, "")
	dir := t.TempDir()
	sub := filepath.Join(dir, "pkg", "x")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := CheckIndexable(&Project{Root: sub}, false); err != nil {
		t.Fatalf("subdir of git work tree should be allowed; got %v", err)
	}
}

func TestCheckIndexableAcceptsGitFileWorktree(t *testing.T) {
	t.Setenv(EnvAllowPaths, "")
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".git"), []byte("gitdir: /tmp/main/.git/worktrees/x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CheckIndexable(&Project{Root: dir}, false); err != nil {
		t.Fatalf("worktree (.git file) should be allowed; got %v", err)
	}
}

func TestCheckIndexableRejectsNonGitNonAllowed(t *testing.T) {
	t.Setenv(EnvAllowPaths, "")
	dir := t.TempDir()
	err := CheckIndexable(&Project{Root: dir}, false)
	if err == nil || !strings.Contains(err.Error(), "not inside a git work tree") {
		t.Fatalf("non-git, non-allowed should be denied; got %v", err)
	}
}

func TestCheckIndexableAllowlistMatch(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvAllowPaths, dir)
	if err := CheckIndexable(&Project{Root: dir}, false); err != nil {
		t.Fatalf("allowlist exact match should pass; got %v", err)
	}
	sub := filepath.Join(dir, "child")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := CheckIndexable(&Project{Root: sub}, false); err != nil {
		t.Fatalf("allowlist subpath should pass; got %v", err)
	}
}

func TestCheckIndexableAllowlistPrefixSeparatorAware(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvAllowPaths, filepath.Join(dir, "foo"))
	sibling := filepath.Join(dir, "foobar")
	if err := os.MkdirAll(sibling, 0o755); err != nil {
		t.Fatal(err)
	}
	err := CheckIndexable(&Project{Root: sibling}, false)
	if err == nil {
		t.Fatal("foobar must not match a 'foo' prefix (separator-aware check)")
	}
}

func TestCheckIndexableAllowlistTildeExpansion(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		t.Skip("no $HOME available")
	}
	parent := filepath.Join(home, ".cache", "dex-guard-test")
	if err := os.MkdirAll(parent, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(parent) })
	t.Setenv(EnvAllowPaths, "~/.cache/dex-guard-test")
	if err := CheckIndexable(&Project{Root: parent}, false); err != nil {
		t.Fatalf("tilde-prefixed allowlist entry should expand; got %v", err)
	}
}

func TestCheckIndexableAllowlistSkipsInvalidEntries(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvAllowPaths, "::relative/path::"+dir)
	if err := CheckIndexable(&Project{Root: dir}, false); err != nil {
		t.Fatalf("invalid entries should be skipped, valid one should pass; got %v", err)
	}
}
