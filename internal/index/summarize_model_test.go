package index

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/chat"
	"github.com/alehatsman/dex/internal/chunk"
)

// captureModelServer returns an httptest server that records the
// "model" field of every chat request and replies with a one-line
// stub summary so the caller doesn't block on a real model. The
// captured model name is read via the mu-guarded slice.
func captureModelServer(t *testing.T) (*httptest.Server, *[]string, *sync.Mutex) {
	t.Helper()
	var (
		mu   sync.Mutex
		seen []string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &req)
		mu.Lock()
		seen = append(seen, req.Model)
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "stub summary."}},
			},
		})
	}))
	return srv, &seen, &mu
}

// TestSummarizePropagatesModel ensures each summarize* function passes
// the per-tier model name through to the chat endpoint. Hardcoding the
// JSON wire field guards against future signature drift between
// summarize* and chat.Client.Generate.
func TestSummarizePropagatesModel(t *testing.T) {
	srv, seen, mu := captureModelServer(t)
	defer srv.Close()

	cc := chat.New(srv.URL, "client-default", 5*time.Second)
	ctx := context.Background()

	cases := []struct {
		name  string
		model string
		call  func() error
	}{
		{
			name:  "chunk uses Chunk model",
			model: "tiny-coder:3b",
			call: func() error {
				_, err := summarizeChunk(ctx, cc, "tiny-coder:3b", "x.go",
					chunk.Chunk{Path: "x.go", Kind: "function_declaration", Content: "func A(){}", StartLine: 1, EndLine: 2})
				return err
			},
		},
		{
			name:  "file uses File model",
			model: "balanced:7b",
			call: func() error {
				_, err := summarizeFile(ctx, cc, "balanced:7b", "x.go", []byte("package x"))
				return err
			},
		},
		{
			name:  "package uses Package model",
			model: "heavy:32b",
			call: func() error {
				_, err := summarizePackage(ctx, cc, "heavy:32b", "x", []string{"summary 1", "summary 2"}, pkgGrounding{})
				return err
			},
		},
		{
			name:  "repo uses Repo model",
			model: "heaviest:70b",
			call: func() error {
				_, err := summarizeRepo(ctx, cc, "heaviest:70b", []string{"pkg a", "pkg b"}, repoGrounding{})
				return err
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mu.Lock()
			*seen = (*seen)[:0]
			mu.Unlock()

			if err := tc.call(); err != nil {
				t.Fatalf("call: %v", err)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(*seen) != 1 {
				t.Fatalf("want 1 chat call, got %d", len(*seen))
			}
			if (*seen)[0] != tc.model {
				t.Errorf("wire model=%q, want %q", (*seen)[0], tc.model)
			}
		})
	}
}

// TestSummarizeEmptyModelUsesClientDefault verifies that passing
// model="" falls through to the chat client's configured default —
// the contract that lets SummaryModels.X = "" mean "use chat default".
func TestSummarizeEmptyModelUsesClientDefault(t *testing.T) {
	srv, seen, mu := captureModelServer(t)
	defer srv.Close()

	cc := chat.New(srv.URL, "client-default", 5*time.Second)
	ctx := context.Background()

	if _, err := summarizeFile(ctx, cc, "", "x.go", []byte("package x")); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(*seen) != 1 || (*seen)[0] != "client-default" {
		t.Errorf("empty model should fall back to client default; got %v", *seen)
	}
}
