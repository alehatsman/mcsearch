package chat

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

func okHandler(reply string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body chatRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		type choice struct {
			Message      Message `json:"message"`
			FinishReason string  `json:"finish_reason"`
		}
		out := struct {
			Choices []choice `json:"choices"`
			Model   string   `json:"model"`
		}{
			Model: body.Model,
			Choices: []choice{{
				Message:      Message{Role: "assistant", Content: reply},
				FinishReason: "stop",
			}},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
	})
}

func TestGenerateRoundTrip(t *testing.T) {
	srv := httptest.NewServer(okHandler("hello world"))
	defer srv.Close()
	c := New(srv.URL, "fake", 5*time.Second)
	resp, err := c.Generate(context.Background(), []Message{{Role: "user", Content: "hi"}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hello world" {
		t.Errorf("content = %q, want %q", resp.Content, "hello world")
	}
	if resp.Model != "fake" {
		t.Errorf("model = %q, want %q", resp.Model, "fake")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want %q", resp.FinishReason, "stop")
	}
}

func TestGenerateUnreachable(t *testing.T) {
	c := New("http://127.0.0.1:1", "fake", 200*time.Millisecond)
	_, err := c.Generate(context.Background(), []Message{{Role: "user", Content: "x"}}, Options{})
	if !errors.Is(err, ErrUnreachable) {
		t.Errorf("expected ErrUnreachable, got %v", err)
	}
}

func TestGenerateServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model overloaded", 503)
	}))
	defer srv.Close()
	c := New(srv.URL, "fake", 2*time.Second)
	_, err := c.Generate(context.Background(), []Message{{Role: "user", Content: "x"}}, Options{})
	if err == nil || !strings.Contains(err.Error(), "503") {
		t.Errorf("expected http 503 error, got %v", err)
	}
}

func TestGenerateNoMessages(t *testing.T) {
	c := New("http://example/", "m", time.Second)
	_, err := c.Generate(context.Background(), nil, Options{})
	if err == nil {
		t.Fatal("expected error for empty messages")
	}
}

func TestNewDefaults(t *testing.T) {
	c := New("http://example/", "m", 0)
	if c.HTTP.Timeout != 120*time.Second {
		t.Errorf("Timeout default = %s, want 120s", c.HTTP.Timeout)
	}
	if strings.HasSuffix(c.BaseURL, "/") {
		t.Errorf("BaseURL should be trimmed: %q", c.BaseURL)
	}
}
