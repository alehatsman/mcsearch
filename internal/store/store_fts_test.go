package store

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestBuildFTSQuery(t *testing.T) {
	cases := []struct {
		name  string
		query string
		mode  FTSMode
		want  string
	}{
		{
			name:  "empty",
			query: "",
			mode:  FTSModeAuto,
			want:  "",
		},
		{
			name:  "single token auto -> AND no-op",
			query: "validate",
			mode:  FTSModeAuto,
			want:  `"validate"`,
		},
		{
			name:  "two tokens auto -> AND",
			query: "validate token",
			mode:  FTSModeAuto,
			want:  `"validate" AND "token"`,
		},
		{
			name:  "three tokens auto -> OR",
			query: "validate token signature",
			mode:  FTSModeAuto,
			want:  `"validate" OR "token" OR "signature"`,
		},
		{
			name:  "explicit OR overrides auto",
			query: "validate token",
			mode:  FTSModeOR,
			want:  `"validate" OR "token"`,
		},
		{
			name:  "explicit AND overrides auto on 3 terms",
			query: "validate token signature",
			mode:  FTSModeAND,
			want:  `"validate" AND "token" AND "signature"`,
		},
		{
			name:  "unicode identifier kept",
			query: "ParseRFC3339Núñez",
			mode:  FTSModeAuto,
			want:  `"ParseRFC3339Núñez"`,
		},
		{
			name:  "non-ASCII script kept",
			query: "ユーザー認証",
			mode:  FTSModeAuto,
			want:  `"ユーザー認証"`,
		},
		{
			name:  "punctuation stripped, internal underscore kept",
			query: "validate_token. signOne!",
			mode:  FTSModeAuto,
			want:  `"validate_token" AND "signOne"`,
		},
		{
			name:  "quoted phrase preserved",
			query: `"package boundary"`,
			mode:  FTSModeAuto,
			want:  `"package boundary"`,
		},
		{
			name:  "phrase plus loose token uses Auto AND",
			query: `"package boundary" check`,
			mode:  FTSModeAuto,
			want:  `"package boundary" AND "check"`,
		},
		{
			name:  "single-char token dropped",
			query: "a validate",
			mode:  FTSModeAuto,
			want:  `"validate"`,
		},
		{
			name:  "all punctuation -> empty",
			query: "!!! ??? ...",
			mode:  FTSModeAuto,
			want:  "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := buildFTSQuery(tc.query, tc.mode)
			if got != tc.want {
				t.Errorf("buildFTSQuery(%q, %v) = %q, want %q", tc.query, tc.mode, got, tc.want)
			}
		})
	}
}

// TestFTSUnicodeMatch indexes a chunk with a non-ASCII identifier and
// verifies the FTS leg returns it for a non-ASCII query — the ASCII-only
// tokenizer used to drop those tokens entirely.
func TestFTSUnicodeMatch(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	ctx := context.Background()
	st, err := Open(ctx, dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	now := time.Now()
	rows := []PendingChunk{
		{Path: "auth.go", Kind: "fn", ContentSHA: "h1", Content: "func ユーザー認証() {}", Vec: []float32{1, 0}},
		{Path: "other.go", Kind: "fn", ContentSHA: "h2", Content: "func unrelated() {}", Vec: []float32{0, 1}},
	}
	if err := st.UpsertMany(ctx, rows, now); err != nil {
		t.Fatal(err)
	}

	hits, err := st.scoreBM25(ctx, "ユーザー認証", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatal("expected at least one BM25 hit for non-ASCII identifier")
	}
}
