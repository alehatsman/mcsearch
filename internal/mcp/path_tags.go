package mcp

import (
	"path/filepath"
	"strings"
)

// pathTag classifies a path into one or more semantic buckets used by
// the suggested_reads ranker. A path can carry multiple tags — every
// fixture is also a test, every non-empty tag set means "not pure
// implementation code". Callers ask membership via the tag* helpers
// below instead of running a chain of string.Contains checks.
type pathTag uint8

const (
	tagDoc pathTag = 1 << iota
	tagBuild
	tagTest
	tagFixture
)

// pathTags returns the set of tags that apply to p. Pure implementation
// code returns 0. Used by pickSuggestedReads to demote anything
// non-implementation in the ranking tiebreaker — when the rerank stage
// lets a Taskfile.yml or README outscore the .go file they describe,
// the non-impl bit pushes the code back to the top.
func pathTags(p string) pathTag {
	var tags pathTag

	if isFixturePathRaw(p) {
		tags |= tagFixture | tagTest
	}

	base := filepath.Base(p)
	switch {
	case strings.HasSuffix(p, ".md"),
		strings.HasSuffix(p, ".rst"),
		strings.HasSuffix(p, ".txt"),
		strings.HasSuffix(p, ".adoc"),
		strings.HasSuffix(p, ".mdx"):
		tags |= tagDoc
	}
	switch {
	case strings.HasSuffix(p, ".yml"),
		strings.HasSuffix(p, ".yaml"),
		strings.HasSuffix(p, ".toml"):
		tags |= tagBuild
	}
	switch base {
	case "Dockerfile", "Makefile", "Taskfile.yml", "Taskfile.yaml":
		tags |= tagBuild
	}
	switch {
	case strings.HasSuffix(base, "_test.go"),
		strings.HasSuffix(base, ".test.ts"),
		strings.HasSuffix(base, ".test.tsx"),
		strings.HasSuffix(base, ".test.js"),
		strings.HasSuffix(base, ".test.jsx"),
		strings.HasSuffix(base, ".spec.ts"),
		strings.HasSuffix(base, ".spec.tsx"),
		strings.HasSuffix(base, ".spec.js"),
		strings.HasSuffix(base, ".spec.jsx"),
		strings.HasSuffix(base, "_test.py"),
		strings.HasPrefix(base, "test_") && strings.HasSuffix(base, ".py"),
		strings.HasSuffix(base, "_spec.rb"),
		strings.HasSuffix(base, "_test.rs"):
		tags |= tagTest
	}
	return tags
}

// has reports whether the tag set contains every bit in mask.
func (t pathTag) has(mask pathTag) bool { return t&mask == mask }

// isFixturePathRaw reports whether p lives inside a fixture directory.
// Separate from pathTags because the segment scan is the costly part
// of classification — `pathTags` calls it once and lets callers reuse
// the result via the tag set instead of re-scanning.
func isFixturePathRaw(p string) bool {
	p = filepath.ToSlash(p)
	for _, seg := range strings.Split(p, "/") {
		switch seg {
		case "testdata", "__fixtures__":
			return true
		}
	}
	return false
}

// The legacy isXPath helpers are thin wrappers over pathTags. They
// remain because callers thread different demotion rules through the
// same input — and because keeping the names stable means the ranker's
// branching logic still reads naturally.

func isDocPath(p string) bool           { return pathTags(p).has(tagDoc) }
func isBuildOrConfigPath(p string) bool { return pathTags(p).has(tagBuild) }
func isTestPath(p string) bool          { return pathTags(p).has(tagTest) }
func isFixturePath(p string) bool       { return pathTags(p).has(tagFixture) }
func isNonImplPath(p string) bool       { return pathTags(p) != 0 }
