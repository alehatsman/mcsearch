// Package chat talks to an OpenAI-compatible /v1/chat/completions endpoint
// (vLLM, TEI's compat shim, ollama, …). It is the generation-side companion
// to internal/embed: where embed turns text into vectors for retrieval,
// chat turns a prompt + retrieved context into a model completion.
package chat

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

// ErrUnreachable is returned when the chat endpoint cannot be reached at
// the network layer. The MCP server translates this into a structured
// "chat-service-unreachable" result so Claude can surface the failure
// cleanly instead of pretending success.
var ErrUnreachable = errors.New("chat service unreachable")

type Client struct {
	BaseURL string
	Model   string
	HTTP    *http.Client
}

// New builds a client. baseURL is the server root (e.g.
// http://127.0.0.1:8082), not the /v1/chat/completions path.
func New(baseURL, model string, timeout time.Duration) *Client {
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	return &Client{
		BaseURL: strings.TrimSuffix(baseURL, "/"),
		Model:   model,
		HTTP:    &http.Client{Timeout: timeout},
	}
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Options struct {
	// Temperature in [0, 2]. Zero means "use server default" (we omit
	// the field rather than forcing greedy decoding).
	Temperature float32
	// MaxTokens caps the response length. Zero means "use server default".
	MaxTokens int
	// Model overrides the client's default Model for this call. Empty
	// means "use c.Model". Lets a single Client serve multiple tools
	// that each want a different model on the same backend — e.g.
	// generate_code on a coder model, ask_codebase on an instruct model.
	Model string
}

type Response struct {
	Content      string
	Model        string
	FinishReason string
}

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature *float32  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Stream      bool      `json:"stream"`
}

type chatResponse struct {
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Model string `json:"model"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Generate sends messages to the chat endpoint and returns the first
// choice. We don't stream — the MCP tool returns once per call, so the
// extra plumbing wouldn't change anything user-visible.
func (c *Client) Generate(ctx context.Context, messages []Message, opts Options) (Response, error) {
	if len(messages) == 0 {
		return Response{}, fmt.Errorf("chat: no messages")
	}
	model := c.Model
	if opts.Model != "" {
		model = opts.Model
	}
	reqBody := chatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
	}
	if opts.Temperature > 0 {
		t := opts.Temperature
		reqBody.Temperature = &t
	}
	if opts.MaxTokens > 0 {
		m := opts.MaxTokens
		reqBody.MaxTokens = &m
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return Response{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Response{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return Response{}, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		buf, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return Response{}, fmt.Errorf("chat: http %d: %s", resp.StatusCode, strings.TrimSpace(string(buf)))
	}
	var parsed chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return Response{}, fmt.Errorf("chat: decode: %w", err)
	}
	if parsed.Error != nil {
		return Response{}, fmt.Errorf("chat: server error: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return Response{}, fmt.Errorf("chat: server returned no choices")
	}
	return Response{
		Content:      parsed.Choices[0].Message.Content,
		Model:        parsed.Model,
		FinishReason: parsed.Choices[0].FinishReason,
	}, nil
}

// Health does a cheap reachability check: a single 1-token completion.
// Returns nil if the endpoint accepted and answered, ErrUnreachable on
// transport failure, otherwise the server error.
func (c *Client) Health(ctx context.Context) error {
	_, err := c.Generate(ctx, []Message{{Role: "user", Content: "ping"}}, Options{MaxTokens: 1})
	return err
}
