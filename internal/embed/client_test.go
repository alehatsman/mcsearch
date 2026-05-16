package embed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type req struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

func ok(dim int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body req
		_ = json.NewDecoder(r.Body).Decode(&body)
		type item struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}
		out := struct {
			Data  []item `json:"data"`
			Model string `json:"model"`
		}{Model: body.Model}
		for i := range body.Input {
			v := make([]float32, dim)
			for j := range v {
				v[j] = float32(i + j)
			}
			out.Data = append(out.Data, item{Index: i, Embedding: v})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
	})
}

func TestEmbedRoundTrip(t *testing.T) {
	srv := httptest.NewServer(ok(8))
	defer srv.Close()
	c := New(srv.URL, "fake", 4, 5*time.Second)
	vecs, err := c.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 3 {
		t.Fatalf("got %d vectors, want 3", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != 8 {
			t.Errorf("vec[%d] dim=%d, want 8", i, len(v))
		}
	}
}

func TestEmbedBatchingHonored(t *testing.T) {
	var calls atomic.Int32
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		ok(4).ServeHTTP(w, r)
	})
	srv := httptest.NewServer(h)
	defer srv.Close()
	c := New(srv.URL, "fake", 3, 5*time.Second)
	if _, err := c.Embed(context.Background(), []string{"a", "b", "c", "d", "e", "f", "g"}); err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("expected 3 batches for batch=3, len=7; got %d", got)
	}
}

func TestEmbedUnreachable(t *testing.T) {
	c := New("http://127.0.0.1:1", "fake", 4, 200*time.Millisecond)
	_, err := c.Embed(context.Background(), []string{"x"})
	if !errors.Is(err, ErrUnreachable) {
		t.Errorf("expected ErrUnreachable, got %v", err)
	}
}

func TestEmbedServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model overloaded", 503)
	}))
	defer srv.Close()
	c := New(srv.URL, "fake", 4, 2*time.Second)
	_, err := c.Embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "503") {
		t.Errorf("expected http 503 error, got %v", err)
	}
}

func TestNewDefaults(t *testing.T) {
	c := New("http://example/", "m", 0, 0)
	if c.Batch != 32 {
		t.Errorf("Batch default = %d, want 32", c.Batch)
	}
	if c.HTTP.Timeout != 60*time.Second {
		t.Errorf("Timeout default = %s, want 60s", c.HTTP.Timeout)
	}
	if strings.HasSuffix(c.BaseURL, "/") {
		t.Errorf("BaseURL should be trimmed: %q", c.BaseURL)
	}
}
