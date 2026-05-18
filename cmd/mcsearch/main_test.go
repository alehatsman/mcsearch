package main

import (
	"testing"
	"time"

	"github.com/alehatsman/mcsearch/internal/rerank"
)

func TestNewRerankClientNilWhenURLEmpty(t *testing.T) {
	t.Setenv("MCSEARCH_RERANK_URL", "")
	if c := newRerankClient(); c != nil {
		t.Errorf("newRerankClient() = %+v, want nil when URL unset", c)
	}
}

func TestNewRerankClientReturnsClientWhenURLSet(t *testing.T) {
	t.Setenv("MCSEARCH_RERANK_URL", "http://127.0.0.1:9999")
	t.Setenv("MCSEARCH_RERANK_MODEL", "custom-reranker")
	t.Setenv("MCSEARCH_DISABLE_RERANK", "")
	t.Setenv("MCSEARCH_RERANK_STYLE", "") // ensure Cohere-style client

	c := newRerankClient()
	if c == nil {
		t.Fatal("newRerankClient() = nil, want non-nil when URL is set")
	}
	if c.Endpoint() != "http://127.0.0.1:9999" {
		t.Errorf("Endpoint() = %q, want http://127.0.0.1:9999", c.Endpoint())
	}
	if c.ModelName() != "custom-reranker" {
		t.Errorf("ModelName() = %q, want custom-reranker", c.ModelName())
	}
}

func TestNewRerankClientChatStyleWhenRequested(t *testing.T) {
	t.Setenv("MCSEARCH_RERANK_URL", "http://127.0.0.1:9999")
	t.Setenv("MCSEARCH_RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
	t.Setenv("MCSEARCH_RERANK_STYLE", "chat")
	t.Setenv("MCSEARCH_DISABLE_RERANK", "")

	c := newRerankClient()
	if c == nil {
		t.Fatal("newRerankClient() = nil, want non-nil when URL is set")
	}
	if _, ok := c.(*rerank.ChatReranker); !ok {
		t.Errorf("expected *rerank.ChatReranker, got %T", c)
	}
	if c.Endpoint() != "http://127.0.0.1:9999" {
		t.Errorf("Endpoint() = %q, want http://127.0.0.1:9999", c.Endpoint())
	}
}

func TestNewRerankClientNilWhenDisableSet(t *testing.T) {
	// URL is set, but the kill switch is on. nil should still be returned.
	t.Setenv("MCSEARCH_RERANK_URL", "http://127.0.0.1:9999")
	t.Setenv("MCSEARCH_DISABLE_RERANK", "1")

	if c := newRerankClient(); c != nil {
		t.Errorf("newRerankClient() = %+v, want nil when DISABLE_RERANK=1", c)
	}
}

func TestNewRerankClientDefaultTimeout(t *testing.T) {
	t.Setenv("MCSEARCH_RERANK_URL", "http://127.0.0.1:9999")
	t.Setenv("MCSEARCH_RERANK_TIMEOUT", "")
	t.Setenv("MCSEARCH_DISABLE_RERANK", "")
	t.Setenv("MCSEARCH_RERANK_STYLE", "") // Cohere-style; access concrete type

	c := newRerankClient()
	if c == nil {
		t.Fatal("expected non-nil client")
	}
	cc, ok := c.(*rerank.Client)
	if !ok {
		t.Fatalf("expected *rerank.Client, got %T", c)
	}
	if cc.HTTP.Timeout != 5*time.Second {
		t.Errorf("default timeout = %s, want 5s", cc.HTTP.Timeout)
	}
}

func TestRerankPoolDefault(t *testing.T) {
	t.Setenv("MCSEARCH_RERANK_POOL", "")
	if got := rerankPool(); got != 40 {
		t.Errorf("rerankPool() = %d, want 40 (default)", got)
	}
}

func TestRerankPoolHonoredInRange(t *testing.T) {
	t.Setenv("MCSEARCH_RERANK_POOL", "60")
	if got := rerankPool(); got != 60 {
		t.Errorf("rerankPool() = %d, want 60", got)
	}
}

func TestRerankPoolClampsHigh(t *testing.T) {
	t.Setenv("MCSEARCH_RERANK_POOL", "9999")
	if got := rerankPool(); got != 100 {
		t.Errorf("rerankPool() = %d, want 100 (clamped)", got)
	}
}

func TestRerankPoolFallbackOnInvalid(t *testing.T) {
	t.Setenv("MCSEARCH_RERANK_POOL", "not-an-int")
	if got := rerankPool(); got != 40 {
		t.Errorf("rerankPool() = %d, want 40 (fallback after warning)", got)
	}
}

func TestRerankPoolFallbackOnNonPositive(t *testing.T) {
	t.Setenv("MCSEARCH_RERANK_POOL", "0")
	if got := rerankPool(); got != 40 {
		t.Errorf("rerankPool() = %d, want 40 (zero falls back)", got)
	}
	t.Setenv("MCSEARCH_RERANK_POOL", "-5")
	if got := rerankPool(); got != 40 {
		t.Errorf("rerankPool() = %d, want 40 (negative falls back)", got)
	}
}
