package rerank

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fakeChatReranker returns a server that responds with the given "yes"/"no"
// logprobs in top_logprobs on the first choice token.
func fakeChatReranker(t *testing.T, yesLP, noLP float64) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{"role": "assistant", "content": "yes"},
					"logprobs": map[string]any{
						"content": []map[string]any{
							{
								"token": "yes", "logprob": yesLP,
								"top_logprobs": []logprobToken{
									{Token: "yes", Logprob: yesLP},
									{Token: "no", Logprob: noLP},
								},
							},
						},
					},
				},
			},
		})
	}))
}

func TestChatRerankerSoftmaxScore(t *testing.T) {
	yesLP := math.Log(0.9)
	noLP := math.Log(0.1)
	srv := fakeChatReranker(t, yesLP, noLP)
	defer srv.Close()

	c := NewChat(srv.URL, "test", 2, 5*time.Second)
	scores, err := c.Rerank(context.Background(), "query", []string{"doc-a", "doc-b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 2 {
		t.Fatalf("got %d scores, want 2", len(scores))
	}
	// Softmax(0.9, 0.1) ≈ 0.9; both docs get the same score from the fake server.
	for _, s := range scores {
		if s.Score < 0.88 || s.Score > 0.92 {
			t.Errorf("score[%d] = %f, want ≈0.90", s.Index, s.Score)
		}
	}
}

func TestChatRerankerTextFallback(t *testing.T) {
	// Server returns no logprobs — should fall back to text parsing.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "yes"}},
			},
		})
	}))
	defer srv.Close()

	c := NewChat(srv.URL, "test", 2, 5*time.Second)
	scores, err := c.Rerank(context.Background(), "q", []string{"doc"})
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 1 || scores[0].Score != 1.0 {
		t.Errorf("text 'yes' fallback: score = %+v, want [{Index:0 Score:1.0}]", scores)
	}
}

func TestChatRerankerNoTextFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "no"}},
			},
		})
	}))
	defer srv.Close()

	c := NewChat(srv.URL, "test", 2, 5*time.Second)
	scores, err := c.Rerank(context.Background(), "q", []string{"doc"})
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 1 || scores[0].Score != 0.0 {
		t.Errorf("text 'no' fallback: score = %+v, want [{Index:0 Score:0.0}]", scores)
	}
}

func TestChatRerankerEmptyDocs(t *testing.T) {
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
	}))
	defer srv.Close()

	c := NewChat(srv.URL, "test", 2, 5*time.Second)
	got, err := c.Rerank(context.Background(), "q", nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != nil {
		t.Errorf("expected nil for empty docs, got %v", got)
	}
	if hit {
		t.Error("server should not be called for empty docs")
	}
}

func TestChatRerankerUnreachable(t *testing.T) {
	c := NewChat("http://127.0.0.1:1", "test", 2, 200*time.Millisecond)
	_, err := c.Rerank(context.Background(), "q", []string{"a"})
	if err == nil {
		t.Fatal("expected error for unreachable server")
	}
	if !errors.Is(err, ErrUnreachable) {
		t.Errorf("expected ErrUnreachable, got %v", err)
	}
}

func TestChatRerankerResultsSortedDescending(t *testing.T) {
	// Return different scores per-call based on request body.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)

		// Find the user message (second to last, before assistant prefill).
		yesLP := math.Log(0.1) // default: low relevance
		noLP := math.Log(0.9)
		for _, m := range body.Messages {
			if strings.Contains(m.Content, "doc-high") {
				yesLP = math.Log(0.9)
				noLP = math.Log(0.1)
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{"content": "yes"},
					"logprobs": map[string]any{
						"content": []map[string]any{
							{
								"token": "yes", "logprob": yesLP,
								"top_logprobs": []map[string]any{
									{"token": "yes", "logprob": yesLP},
									{"token": "no", "logprob": noLP},
								},
							},
						},
					},
				},
			},
		})
	}))
	defer srv.Close()

	c := NewChat(srv.URL, "test", 4, 5*time.Second)
	scores, err := c.Rerank(context.Background(), "q", []string{"doc-low", "doc-high", "doc-low"})
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 3 {
		t.Fatalf("got %d scores, want 3", len(scores))
	}
	// Top score must be for "doc-high" (index 1).
	if scores[0].Index != 1 {
		t.Errorf("top result index = %d, want 1 (doc-high)", scores[0].Index)
	}
	// Results must be in descending order.
	for i := 1; i < len(scores); i++ {
		if scores[i].Score > scores[i-1].Score {
			t.Errorf("results not sorted descending: scores[%d]=%.3f > scores[%d]=%.3f",
				i, scores[i].Score, i-1, scores[i-1].Score)
		}
	}
}

func TestChatRerankerHealthCheck(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "yes"}},
			},
		})
	}))
	defer srv.Close()

	c := NewChat(srv.URL, "test", 2, 5*time.Second)
	if err := c.Health(context.Background()); err != nil {
		t.Errorf("Health on reachable endpoint = %v, want nil", err)
	}
}

func TestChatRerankerImplementsHealthChecker(t *testing.T) {
	c := NewChat("http://example.com", "model", 4, 5*time.Second)
	var _ HealthChecker = c // compile-time interface check
	if c.Endpoint() != "http://example.com" {
		t.Errorf("Endpoint() = %q", c.Endpoint())
	}
	if c.ModelName() != "model" {
		t.Errorf("ModelName() = %q", c.ModelName())
	}
}

func TestYesNoFromLogprobs(t *testing.T) {
	cases := []struct {
		name  string
		tops  []logprobToken
		wantF float32
		wantD float32 // allowed delta
	}{
		{
			name: "both yes and no → softmax",
			tops: []logprobToken{
				{Token: "yes", Logprob: math.Log(0.9)},
				{Token: "no", Logprob: math.Log(0.1)},
			},
			wantF: 0.9, wantD: 0.01,
		},
		{
			name:  "only yes → sigmoid-like",
			tops:  []logprobToken{{Token: "yes", Logprob: math.Log(0.7)}},
			wantF: 0.7, wantD: 0.01,
		},
		{
			name:  "only no → 0",
			tops:  []logprobToken{{Token: "no", Logprob: math.Log(0.8)}},
			wantF: 0, wantD: 0,
		},
		{
			name:  "neither → neutral 0.5",
			tops:  []logprobToken{{Token: "maybe", Logprob: -1}},
			wantF: 0.5, wantD: 0,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := yesNoFromLogprobs(tc.tops)
			diff := float32(math.Abs(float64(got - tc.wantF)))
			if diff > tc.wantD {
				t.Errorf("got %.4f, want ≈%.4f (±%.4f)", got, tc.wantF, tc.wantD)
			}
		})
	}
}
