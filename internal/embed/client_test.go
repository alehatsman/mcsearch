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
	if c.Concurrency != 1 {
		t.Errorf("Concurrency default = %d, want 1 (sequential)", c.Concurrency)
	}
}

// TestEmbedConcurrentDispatch verifies that NewWithConcurrency lets multiple
// batches actually fly in parallel. The handler blocks until it sees the
// configured number of simultaneous requests, then releases them — if
// dispatch were still sequential the test would hang and fail.
func TestEmbedConcurrentDispatch(t *testing.T) {
	const conc = 4
	var (
		inFlight    atomic.Int32
		peak        atomic.Int32
		totalCalls  atomic.Int32
		releaseGate = make(chan struct{})
		readyGate   = make(chan struct{}, conc)
	)
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		now := inFlight.Add(1)
		defer inFlight.Add(-1)
		totalCalls.Add(1)
		for {
			old := peak.Load()
			if now <= old || peak.CompareAndSwap(old, now) {
				break
			}
		}
		select {
		case readyGate <- struct{}{}:
		default:
		}
		<-releaseGate
		ok(4).ServeHTTP(w, r)
	})
	srv := httptest.NewServer(h)
	defer srv.Close()

	c := NewWithConcurrency(srv.URL, "fake", 1, conc, 5*time.Second)
	inputs := []string{"a", "b", "c", "d", "e", "f", "g", "h"}

	done := make(chan error, 1)
	var vecs [][]float32
	go func() {
		var err error
		vecs, err = c.Embed(context.Background(), inputs)
		done <- err
	}()

	deadline := time.After(2 * time.Second)
	for i := 0; i < conc; i++ {
		select {
		case <-readyGate:
		case <-deadline:
			t.Fatalf("only %d concurrent requests reached the server (want %d)", peak.Load(), conc)
		}
	}
	close(releaseGate)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if got := peak.Load(); got < conc {
		t.Errorf("peak in-flight = %d, want >= %d", got, conc)
	}
	if got := totalCalls.Load(); got != int32(len(inputs)) {
		t.Errorf("total calls = %d, want %d", got, len(inputs))
	}
	if len(vecs) != len(inputs) {
		t.Fatalf("got %d vectors, want %d", len(vecs), len(inputs))
	}
	for i := range inputs {
		if len(vecs[i]) != 4 {
			t.Errorf("vec[%d] dim=%d, want 4", i, len(vecs[i]))
		}
	}
}

func TestEmbedConcurrentError(t *testing.T) {
	var calls atomic.Int32
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n == 2 {
			http.Error(w, "boom", 500)
			return
		}
		ok(4).ServeHTTP(w, r)
	})
	srv := httptest.NewServer(h)
	defer srv.Close()
	c := NewWithConcurrency(srv.URL, "fake", 1, 4, 5*time.Second)
	_, err := c.Embed(context.Background(), []string{"a", "b", "c", "d"})
	if err == nil {
		t.Fatal("expected error from failing batch, got nil")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected http 500 error, got %v", err)
	}
}
