package rerank

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRerankHappyPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "test",
			"results": []map[string]any{
				{"index": 2, "relevance_score": 0.95},
				{"index": 0, "relevance_score": 0.51},
				{"index": 1, "relevance_score": 0.10},
			},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "test", 5*time.Second)
	got, err := c.Rerank(context.Background(), "q", []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d scores, want 3", len(got))
	}
	if got[0].Index != 2 || got[0].Score < 0.94 {
		t.Errorf("top score = %+v, want Index=2 Score≈0.95", got[0])
	}
	if got[2].Index != 1 || got[2].Score > 0.11 {
		t.Errorf("bottom score = %+v, want Index=1 Score≈0.10", got[2])
	}
}

func TestRerankSendsExpectedFields(t *testing.T) {
	var got rerankRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rerank" {
			t.Errorf("path = %q, want /rerank", r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", ct)
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		_ = json.NewEncoder(w).Encode(map[string]any{"results": []any{}})
	}))
	defer srv.Close()

	c := New(srv.URL, "test-model", 5*time.Second)
	if _, err := c.Rerank(context.Background(), "the query", []string{"doc-a", "doc-b"}); err != nil {
		t.Fatal(err)
	}
	if got.Model != "test-model" {
		t.Errorf("Model = %q, want test-model", got.Model)
	}
	if got.Query != "the query" {
		t.Errorf("Query = %q", got.Query)
	}
	if got.TopN != 2 {
		t.Errorf("TopN = %d, want 2 (len(docs))", got.TopN)
	}
	if len(got.Documents) != 2 || got.Documents[0] != "doc-a" || got.Documents[1] != "doc-b" {
		t.Errorf("Documents = %v", got.Documents)
	}
}

func TestRerankEmptyDocs(t *testing.T) {
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
	}))
	defer srv.Close()

	c := New(srv.URL, "test", 5*time.Second)
	got, err := c.Rerank(context.Background(), "q", nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != nil {
		t.Errorf("got %v, want nil", got)
	}
	if hit {
		t.Error("server should not have been called for empty docs")
	}
}

func TestRerankUnreachable(t *testing.T) {
	// Pointing at a port nothing should be listening on. Short timeout
	// so the test exits quickly even on the unlikely race.
	c := New("http://127.0.0.1:1", "test", 200*time.Millisecond)
	_, err := c.Rerank(context.Background(), "q", []string{"a"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrUnreachable) {
		t.Errorf("expected ErrUnreachable, got %v", err)
	}
}

func TestRerankServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad request"}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "test", 5*time.Second)
	_, err := c.Rerank(context.Background(), "q", []string{"a"})
	if err == nil {
		t.Fatal("expected error")
	}
	if errors.Is(err, ErrUnreachable) {
		t.Error("server error should not be classified as unreachable")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("error should mention status code: %v", err)
	}
}

func TestRerankOutOfRangeIndexFiltered(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]any{
				{"index": 0, "relevance_score": 0.9},
				{"index": 99, "relevance_score": 0.5}, // out of range
				{"index": -1, "relevance_score": 0.4}, // negative
			},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "test", 5*time.Second)
	got, err := c.Rerank(context.Background(), "q", []string{"only-one"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Index != 0 {
		t.Errorf("expected exactly 1 in-range score with Index=0, got %+v", got)
	}
}

func TestRerankServerErrorObjectInJSON(t *testing.T) {
	// Cohere-style error reply that comes back with a 200 but an error
	// object in the JSON body.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"message": "model not loaded"},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "test", 5*time.Second)
	_, err := c.Rerank(context.Background(), "q", []string{"a"})
	if err == nil {
		t.Fatal("expected error from embedded error object")
	}
	if !strings.Contains(err.Error(), "model not loaded") {
		t.Errorf("error should surface server message: %v", err)
	}
}

func TestNewDefaults(t *testing.T) {
	c := New("http://example/", "m", 0)
	if c.HTTP.Timeout != 5*time.Second {
		t.Errorf("default timeout = %s, want 5s", c.HTTP.Timeout)
	}
	if strings.HasSuffix(c.BaseURL, "/") {
		t.Errorf("BaseURL trailing slash should be trimmed: %q", c.BaseURL)
	}
}

func TestHealthSucceedsOnReachable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]any{{"index": 0, "relevance_score": 1.0}},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "test", 5*time.Second)
	if err := c.Health(context.Background()); err != nil {
		t.Errorf("Health on reachable endpoint = %v, want nil", err)
	}
}

func TestHealthReportsUnreachable(t *testing.T) {
	c := New("http://127.0.0.1:1", "test", 200*time.Millisecond)
	err := c.Health(context.Background())
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrUnreachable) {
		t.Errorf("expected ErrUnreachable, got %v", err)
	}
}
