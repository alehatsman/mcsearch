package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

// startTestHTTPServer wraps an *httptest.Server around the same mux
// RunHTTP installs, so endpoint tests don't have to bind a port or
// race the listener.
func startTestHTTPServer(t *testing.T, srv *Server, opts RunHTTPOptions) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(srv.buildHTTPHandler(opts))
	t.Cleanup(ts.Close)
	return ts
}

// stubServer returns a minimally-initialized *Server suitable for
// endpoint tests that just want to exercise routing / auth /
// project-resolution. Underlying tools that would touch the embed
// or chat endpoint will return appropriate status codes; that's
// covered by their own tests.
func stubServer(t *testing.T) *Server {
	t.Helper()
	return &Server{IndexDir: t.TempDir()}
}

// TestProjectIDDeterministic confirms ProjectID is stable across
// calls and matches what BuildProjectRegistry computes for the same
// input. Stability is the load-bearing guarantee that client URLs
// keep working across daemon restarts.
func TestProjectIDDeterministic(t *testing.T) {
	dir := t.TempDir()
	id1, err := ProjectID(dir)
	if err != nil {
		t.Fatalf("ProjectID: %v", err)
	}
	if len(id1) != 64 {
		t.Errorf("ProjectID length: got %d, want 64 hex chars", len(id1))
	}
	id2, err := ProjectID(dir)
	if err != nil {
		t.Fatalf("ProjectID (2nd): %v", err)
	}
	if id1 != id2 {
		t.Errorf("ProjectID drifts between calls: %s != %s", id1, id2)
	}
	reg, err := BuildProjectRegistry([]string{dir})
	if err != nil {
		t.Fatalf("BuildProjectRegistry: %v", err)
	}
	if _, ok := reg[id1]; !ok {
		t.Errorf("BuildProjectRegistry computed a different id than ProjectID: got %v want %s", reg, id1)
	}
}

// TestValidateBindForAuth covers the startup-time check that rejects
// no-token on non-loopback. Both bare `:port` and an explicit public
// IP should fail without a token; loopback bindings succeed.
func TestValidateBindForAuth(t *testing.T) {
	cases := []struct {
		name    string
		addr    string
		token   string
		wantErr bool
	}{
		{"loopback+token", "127.0.0.1:8080", "secret", false},
		{"loopback no token", "127.0.0.1:8080", "", false},
		{"localhost no token", "localhost:8080", "", false},
		{"ipv6 loopback no token", "[::1]:8080", "", false},
		{"all-interfaces+token", ":8080", "secret", false},
		{"all-interfaces no token", ":8080", "", true},
		{"public IP no token", "10.0.0.5:8080", "", true},
		{"public IP+token", "10.0.0.5:8080", "secret", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := validateBindForAuth(c.addr, c.token)
			if (err != nil) != c.wantErr {
				t.Errorf("validateBindForAuth(%q, token=%q): err=%v, wantErr=%v", c.addr, c.token, err, c.wantErr)
			}
		})
	}
}

// TestAuthMiddleware exercises the bearer-token gate. Empty token =
// no-op (auth disabled); non-empty token = strict equality. The
// handler beneath the middleware is observable via a flag the test
// inspects.
func TestAuthMiddleware(t *testing.T) {
	called := false
	inner := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusNoContent)
	})

	cases := []struct {
		name       string
		token      string
		header     string
		wantStatus int
		wantInner  bool
	}{
		{"no token, no header", "", "", http.StatusNoContent, true},
		{"token set, no header", "secret", "", http.StatusUnauthorized, false},
		{"token set, wrong header", "secret", "Bearer wrong", http.StatusUnauthorized, false},
		{"token set, right header", "secret", "Bearer secret", http.StatusNoContent, true},
		{"token set, missing Bearer prefix", "secret", "secret", http.StatusUnauthorized, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			called = false
			h := authMiddleware(c.token, inner)
			rec := httptest.NewRecorder()
			req := httptest.NewRequest("GET", "/v1/anything", nil)
			if c.header != "" {
				req.Header.Set("Authorization", c.header)
			}
			h.ServeHTTP(rec, req)
			if rec.Code != c.wantStatus {
				t.Errorf("status: got %d, want %d", rec.Code, c.wantStatus)
			}
			if called != c.wantInner {
				t.Errorf("inner called: got %v, want %v", called, c.wantInner)
			}
		})
	}
}

// TestConstantTimeEqual is a sanity check that the comparison is
// length-correct. We don't measure timing — that's beyond unit
// scope; we just confirm the equality result for representative
// inputs.
func TestConstantTimeEqual(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"", "", true},
		{"a", "a", true},
		{"abc", "abc", true},
		{"abc", "abd", false},
		{"abc", "abcd", false}, // different lengths
		{"", "x", false},
	}
	for _, c := range cases {
		got := constantTimeEqual(c.a, c.b)
		if got != c.want {
			t.Errorf("constantTimeEqual(%q, %q) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

// TestHTTPUnauthenticated confirms /v1/healthz and /v1/version are
// reachable without a bearer token even when the token gate is
// active for the rest.
func TestHTTPUnauthenticated(t *testing.T) {
	srv := stubServer(t)
	ts := startTestHTTPServer(t, srv, RunHTTPOptions{Token: "secret"})

	for _, path := range []string{"/v1/healthz", "/v1/version"} {
		resp, err := http.Get(ts.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Errorf("%s: status=%d, want 200", path, resp.StatusCode)
		}
		_ = resp.Body.Close()
	}
}

// TestHTTPAuthRequired confirms a protected endpoint (e.g. /v1/projects)
// is gated by the token when one is set.
func TestHTTPAuthRequired(t *testing.T) {
	srv := stubServer(t)
	dir := t.TempDir()
	registry, err := BuildProjectRegistry([]string{dir})
	if err != nil {
		t.Fatalf("BuildProjectRegistry: %v", err)
	}
	ts := startTestHTTPServer(t, srv, RunHTTPOptions{
		Token:    "secret",
		Projects: registry,
	})

	// No header → 401.
	resp, err := http.Get(ts.URL + "/v1/projects")
	if err != nil {
		t.Fatalf("GET /v1/projects: %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("no auth: status=%d, want 401", resp.StatusCode)
	}
	_ = resp.Body.Close()

	// With header → 200 + json list.
	req, _ := http.NewRequest("GET", ts.URL+"/v1/projects", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET /v1/projects authed: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("authed: status=%d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if !strings.Contains(string(body), `"id"`) {
		t.Errorf("body missing id field; got %s", string(body))
	}
}

// TestHTTPProjectUnknownReturns404 confirms a request against an
// unregistered project ID gets a clean 404.
func TestHTTPProjectUnknownReturns404(t *testing.T) {
	srv := stubServer(t)
	ts := startTestHTTPServer(t, srv, RunHTTPOptions{})

	resp, err := http.Post(ts.URL+"/v1/projects/deadbeef/ask",
		"application/json", strings.NewReader(`{"question":"hi"}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status=%d, want 404", resp.StatusCode)
	}
}

// TestHTTPProjectIDOverridesBody confirms the URL's {id} always wins
// over a project field in the JSON body — clients can't smuggle in
// a different project root to bypass the registry.
func TestHTTPProjectIDOverridesBody(t *testing.T) {
	srv := stubServer(t)
	dir := t.TempDir()
	registry, err := BuildProjectRegistry([]string{dir})
	if err != nil {
		t.Fatalf("BuildProjectRegistry: %v", err)
	}
	var id string
	for k := range registry {
		id = k
		break
	}
	ts := startTestHTTPServer(t, srv, RunHTTPOptions{Projects: registry})

	// POST /v1/projects/{id}/ask with a malicious project in the body.
	body, _ := json.Marshal(map[string]string{
		"question": "where is the watcher?",
		"project":  "/etc",
	})
	resp, err := http.Post(ts.URL+"/v1/projects/"+id+"/ask",
		"application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	// The endpoint should reach the handler (200 even though the
	// underlying project has no index — the handler returns
	// status="no-index" in its body, not an HTTP error).
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status=%d, want 200 (handler should reach the registry-overridden project)", resp.StatusCode)
	}
	// And the project the handler resolved should be the registered
	// dir, not /etc. We detect this by inspecting the body: the
	// status response carries "project" with the resolved root.
	var out map[string]any
	respBody, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(respBody, &out); err != nil {
		t.Fatalf("unmarshal: %v body=%s", err, respBody)
	}
	proj, _ := out["project"].(string)
	if proj != "" {
		// The dir we registered has been EvalSymlinks-resolved, same
		// path the handler will canonicalize to internally. Compare
		// the absolute resolved form for portability across symlink
		// layouts (macOS /var → /private/var, etc.).
		dirReal, _ := filepath.EvalSymlinks(dir)
		if proj != dirReal && proj != dir {
			t.Errorf("handler resolved project=%q, want %q (registry root)", proj, dirReal)
		}
	}
}

// TestHTTPBadJSONReturns400 confirms a malformed JSON body produces
// a 400 with a useful error message rather than a 500.
func TestHTTPBadJSONReturns400(t *testing.T) {
	srv := stubServer(t)
	dir := t.TempDir()
	registry, err := BuildProjectRegistry([]string{dir})
	if err != nil {
		t.Fatalf("BuildProjectRegistry: %v", err)
	}
	var id string
	for k := range registry {
		id = k
		break
	}
	ts := startTestHTTPServer(t, srv, RunHTTPOptions{Projects: registry})

	resp, err := http.Post(ts.URL+"/v1/projects/"+id+"/ask",
		"application/json", strings.NewReader(`{not valid json`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status=%d, want 400", resp.StatusCode)
	}
}

// TestRunHTTPRejectsUnsafeBind confirms RunHTTP errors out
// pre-listen when given a non-loopback bind without a token —
// catches misconfiguration before it exposes the daemon to the
// network.
func TestRunHTTPRejectsUnsafeBind(t *testing.T) {
	srv := stubServer(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	err := srv.RunHTTP(ctx, RunHTTPOptions{Addr: ":0", Token: ""})
	if err == nil {
		t.Fatal("expected RunHTTP to reject :0 without a token")
	}
	if !strings.Contains(err.Error(), "all interfaces") && !strings.Contains(err.Error(), "non-loopback") {
		t.Errorf("error doesn't mention bind safety: %v", err)
	}
}

// TestHandleListProjectsSortsByRoot pins the ordering so client
// pagination / display stays stable across daemon restarts when the
// project set is the same.
func TestHandleListProjectsSortsByRoot(t *testing.T) {
	srv := stubServer(t)
	// Two registered projects with sortable root paths.
	a := t.TempDir()
	b := t.TempDir()
	// Use the first two letters of each path to bias the sort; we
	// don't control the temp-dir prefix so we just check ordering
	// matches lexicographic root comparison.
	registry, err := BuildProjectRegistry([]string{a, b})
	if err != nil {
		t.Fatalf("BuildProjectRegistry: %v", err)
	}
	ts := startTestHTTPServer(t, srv, RunHTTPOptions{Projects: registry})

	resp, err := http.Get(ts.URL + "/v1/projects")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	var out struct {
		Projects []struct {
			ID   string `json:"id"`
			Root string `json:"root"`
		} `json:"projects"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("unmarshal: %v body=%s", err, body)
	}
	if len(out.Projects) != 2 {
		t.Fatalf("expected 2 projects, got %d", len(out.Projects))
	}
	if out.Projects[0].Root >= out.Projects[1].Root {
		t.Errorf("projects not sorted by root: %v", out.Projects)
	}
}
