// Package embed talks to an OpenAI-compatible /v1/embeddings endpoint
// (vLLM, TEI's compat shim, ollama, …). It batches inputs and returns
// packed float32 vectors.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ErrUnreachable is returned when the embed endpoint cannot be reached at
// the network layer. The MCP server translates this into a structured
// "embedding-service-unreachable" result so Claude can fall back to grep.
var ErrUnreachable = errors.New("embedding service unreachable")

type Client struct {
	BaseURL string
	Model   string
	Batch   int
	HTTP    *http.Client
}

// New builds a client. baseURL is the server root (e.g.
// http://127.0.0.1:8082), not the /v1/embeddings path.
func New(baseURL, model string, batch int, timeout time.Duration) *Client {
	if batch <= 0 {
		batch = 32
	}
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Model:   model,
		Batch:   batch,
		HTTP:    &http.Client{Timeout: timeout},
	}
}

type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Embed sends inputs in batches of c.Batch and returns one vector per input.
func (c *Client) Embed(ctx context.Context, inputs []string) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, nil
	}
	out := make([][]float32, len(inputs))
	for start := 0; start < len(inputs); start += c.Batch {
		end := start + c.Batch
		if end > len(inputs) {
			end = len(inputs)
		}
		got, err := c.embedBatch(ctx, inputs[start:end])
		if err != nil {
			return nil, err
		}
		if len(got) != end-start {
			return nil, fmt.Errorf("embed: server returned %d vectors for %d inputs", len(got), end-start)
		}
		copy(out[start:end], got)
	}
	return out, nil
}

func (c *Client) embedBatch(ctx context.Context, inputs []string) ([][]float32, error) {
	body, err := json.Marshal(embedRequest{Model: c.Model, Input: inputs})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		buf, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("embed: http %d: %s", resp.StatusCode, strings.TrimSpace(string(buf)))
	}
	var parsed embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("embed: decode: %w", err)
	}
	if parsed.Error != nil {
		return nil, fmt.Errorf("embed: server error: %s", parsed.Error.Message)
	}
	out := make([][]float32, len(parsed.Data))
	for _, d := range parsed.Data {
		if d.Index < 0 || d.Index >= len(out) {
			return nil, fmt.Errorf("embed: bogus index %d in response", d.Index)
		}
		out[d.Index] = d.Embedding
	}
	return out, nil
}

// Health does a cheap reachability check: a single 1-input embed call.
// Returns nil if the endpoint accepted and answered, ErrUnreachable on
// transport failure, otherwise the server error.
func (c *Client) Health(ctx context.Context) error {
	_, err := c.embedBatch(ctx, []string{"ping"})
	return err
}
