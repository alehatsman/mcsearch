package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/chat"
	"github.com/alehatsman/dex/internal/embed"
	"github.com/alehatsman/dex/internal/ignore"
	"github.com/alehatsman/dex/internal/index"
	"github.com/alehatsman/dex/internal/proj"
	"github.com/alehatsman/dex/internal/rerank"
	"github.com/alehatsman/dex/internal/store"
)

func fakeEmbed(t *testing.T, dim int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		type item struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}
		out := struct {
			Data []item `json:"data"`
		}{}
		for i, s := range body.Input {
			out.Data = append(out.Data, item{Index: i, Embedding: hashVec(s, dim)})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
	}))
}

func hashVec(s string, dim int) []float32 {
	out := make([]float32, dim)
	h := sha256.Sum256([]byte(s))
	for i := range dim {
		u := binary.LittleEndian.Uint32(h[(i*4)%len(h):])
		out[i] = float32(int32(u)) / float32(math.MaxInt32)
	}
	return out
}

// indexProject indexes projDir into cacheDir using srvURL and returns
// the resolved project root (mcp.Server expects absolute paths).
func indexProject(t *testing.T, projDir, cacheDir, srvURL string) string {
	t.Helper()
	ctx := context.Background()
	p, err := proj.Resolve(projDir, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.EnsureCacheDir(); err != nil {
		t.Fatal(err)
	}
	st, err := store.Open(ctx, p.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(srvURL, "fake", 16, 5*time.Second)
	ix := index.New(p, st, em, ig, index.Options{})
	if err := ix.Run(ctx); err != nil {
		t.Fatal(err)
	}
	return p.Root
}

func writeFile(t *testing.T, p, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func newServer(srvURL, cacheDir string) *Server {
	return &Server{
		EmbedClient: embed.New(srvURL, "fake", 16, 5*time.Second),
		IndexDir:    cacheDir,
	}
}

func TestSearchOk(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "main.go"),
		"package main\n\nfunc Greet(name string) string { return \"hi \" + name }\nfunc Bye() {}\n")

	root := indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	_, out, err := s.search(context.Background(), nil, SearchInput{
		Query:       "greeting function",
		ProjectRoot: root,
		K:           5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != "ok" {
		t.Errorf("status = %q, want ok (hint: %q)", out.Status, out.Hint)
	}
	if len(out.Hits) == 0 {
		t.Fatal("expected at least one hit")
	}
}

func TestSearchNoIndex(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	// No index pass.
	s := newServer(srv.URL, cacheDir)
	_, out, _ := s.search(context.Background(), nil, SearchInput{
		Query:       "anything",
		ProjectRoot: projDir,
	})
	if out.Status != "no-index" {
		t.Errorf("status = %q, want no-index", out.Status)
	}
	if out.Hint == "" {
		t.Error("expected a hint for no-index")
	}
}

func TestSearchEmptyQuery(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	s := newServer(srv.URL, t.TempDir())
	for _, q := range []string{"", "   ", "\t\n  "} {
		_, out, _ := s.search(context.Background(), nil, SearchInput{Query: q})
		if out.Status != "error" {
			t.Errorf("query=%q status=%q, want error", q, out.Status)
		}
	}
}

func TestSearchBadProjectRoot(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	s := newServer(srv.URL, t.TempDir())
	_, out, _ := s.search(context.Background(), nil, SearchInput{
		Query:       "x",
		ProjectRoot: "/this/path/does/not/exist",
	})
	if out.Status != "error" {
		t.Errorf("status = %q, want error", out.Status)
	}
}

func TestSearchEmbeddingUnreachable(t *testing.T) {
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	// Need an indexed project first; index against a reachable server,
	// then point the MCP server at a dead one for the actual query.
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	root := indexProject(t, projDir, cacheDir, srv.URL)
	writeFile(t, filepath.Join(projDir, "x.go"), "package main\n")

	s := &Server{
		EmbedClient: embed.New("http://127.0.0.1:1", "fake", 16, 200*time.Millisecond),
		IndexDir:    cacheDir,
	}
	_, out, _ := s.search(context.Background(), nil, SearchInput{
		Query:       "x",
		ProjectRoot: root,
	})
	if out.Status != "embedding-service-unreachable" {
		t.Errorf("status = %q, want embedding-service-unreachable", out.Status)
	}
	if out.Endpoint == "" {
		t.Error("expected Endpoint to be populated on unreachable")
	}
}

func TestSearchKClamping(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	for i := range 40 {
		writeFile(t, filepath.Join(projDir, "f", "g.go"),
			"package main\nfunc F() {}\n") // overwrites — only 1 file needed
		_ = i
	}
	// 40 small Go files so we have enough chunks to test clamping.
	for i := range 40 {
		writeFile(t, filepath.Join(projDir, "f", "f"+itoa(i)+".go"),
			"package main\nfunc F"+itoa(i)+"() {}\n")
	}
	root := indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	_, out, _ := s.search(context.Background(), nil, SearchInput{
		Query: "any", ProjectRoot: root, K: 1000,
	})
	if len(out.Hits) > 30 {
		t.Errorf("got %d hits, want ≤30 (clamp)", len(out.Hits))
	}
	_, out, _ = s.search(context.Background(), nil, SearchInput{
		Query: "any", ProjectRoot: root, K: -1,
	})
	if len(out.Hits) == 0 || len(out.Hits) > 8 {
		t.Errorf("k=-1 → got %d hits, want default 8", len(out.Hits))
	}
}

func TestStatusReachable(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "a.go"), "package main\n")
	indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	_, out, _ := s.status(context.Background(), nil, StatusInput{})
	if !out.Reachable {
		t.Errorf("Reachable = false, want true (error: %s)", out.Error)
	}
	if out.Version == "" {
		t.Error("Version field empty")
	}
	if len(out.Projects) == 0 {
		t.Error("Projects empty after indexing")
	}
}

func TestStatusUnreachable(t *testing.T) {
	s := &Server{
		EmbedClient: embed.New("http://127.0.0.1:1", "fake", 16, 200*time.Millisecond),
		IndexDir:    t.TempDir(),
	}
	_, out, _ := s.status(context.Background(), nil, StatusInput{})
	if out.Reachable {
		t.Error("Reachable = true on a dead endpoint")
	}
	if out.Error == "" {
		t.Error("expected Error to be populated on unreachable")
	}
}

func TestSearchDefaultsToCwd(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "g.go"),
		"package main\nfunc G() {}\n")
	indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	// Chdir into projDir; an empty ProjectRoot should resolve there.
	cwd, _ := os.Getwd()
	defer os.Chdir(cwd)
	_ = os.Chdir(projDir)

	_, out, _ := s.search(context.Background(), nil, SearchInput{Query: "G"})
	if out.Status != "ok" {
		t.Errorf("status = %q, want ok (project=%q)", out.Status, out.Project)
	}
}

func TestBuildSummarizeSystem(t *testing.T) {
	base := buildSummarizeSystem("")
	for _, want := range []string{
		"file summarizer",  // file-kind agnostic, not "code summarizer"
		"Makefiles",        // hint that non-code files have their own framing
		"top-level keys",   // config files
		"section headings", // docs
	} {
		if !strings.Contains(base, want) {
			t.Errorf("base prompt missing %q", want)
		}
	}
	if strings.Contains(base, "Focus specifically on") {
		t.Errorf("empty focus should not inject a focus clause; got: %s", base)
	}

	withFocus := buildSummarizeSystem("  public API surface  ")
	if !strings.Contains(withFocus, "Focus specifically on: public API surface.") {
		t.Errorf("focus clause missing or untrimmed; got: %s", withFocus)
	}
}

// stubReranker returns the docs in input order with descending
// scores; enough to drive a non-zero RerankScore on every Hit.
type stubReranker struct{}

func (stubReranker) Rerank(_ context.Context, _ string, docs []string) ([]rerank.Score, error) {
	out := make([]rerank.Score, len(docs))
	for i := range docs {
		out[i] = rerank.Score{Index: i, Score: 1.0 - float32(i)*0.1}
	}
	return out, nil
}

func TestSearchPopulatesRerankScore(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	// 8 chunks so the fused pool exceeds k=3 and the rerank stage fires.
	for i := range 8 {
		writeFile(t, filepath.Join(projDir, "f"+itoa(i)+".go"),
			"package main\nfunc F"+itoa(i)+"() {}\n")
	}
	root := indexProject(t, projDir, cacheDir, srv.URL)

	s := &Server{
		EmbedClient: embed.New(srv.URL, "fake", 16, 5*time.Second),
		IndexDir:    cacheDir,
		StoreOpts:   store.Options{Reranker: stubReranker{}},
	}
	_, out, err := s.search(context.Background(), nil, SearchInput{
		Query:       "function",
		ProjectRoot: root,
		K:           3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != "ok" {
		t.Fatalf("status = %q, want ok (hint: %q)", out.Status, out.Hint)
	}
	if len(out.Hits) == 0 {
		t.Fatal("expected hits")
	}
	for i, h := range out.Hits {
		if h.RerankScore <= 0 {
			t.Errorf("hit[%d] %q: RerankScore = %v, want > 0", i, h.Path, h.RerankScore)
		}
	}
}

func TestStatusReportsRerankEndpoint(t *testing.T) {
	embedSrv := fakeEmbed(t, 16)
	defer embedSrv.Close()

	rerankSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "fake-reranker",
			"results": []map[string]any{{"index": 0, "relevance_score": 0.99}},
		})
	}))
	defer rerankSrv.Close()

	s := &Server{
		EmbedClient:  embed.New(embedSrv.URL, "fake-embed", 16, 5*time.Second),
		RerankClient: rerank.New(rerankSrv.URL, "fake-reranker", 5*time.Second),
		IndexDir:     t.TempDir(),
	}
	_, out, _ := s.status(context.Background(), nil, StatusInput{})
	if out.RerankEndpoint != rerankSrv.URL {
		t.Errorf("RerankEndpoint = %q, want %q", out.RerankEndpoint, rerankSrv.URL)
	}
	if out.RerankModel != "fake-reranker" {
		t.Errorf("RerankModel = %q, want fake-reranker", out.RerankModel)
	}
	if !out.RerankReachable {
		t.Error("RerankReachable = false, want true (fake server is up)")
	}
}

func TestStatusOmitsRerankWhenUnwired(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	s := newServer(srv.URL, t.TempDir())

	_, out, _ := s.status(context.Background(), nil, StatusInput{})
	if out.RerankEndpoint != "" || out.RerankModel != "" || out.RerankReachable {
		t.Errorf("rerank fields should be zero when RerankClient is nil; got endpoint=%q model=%q reachable=%v",
			out.RerankEndpoint, out.RerankModel, out.RerankReachable)
	}
}

// fakeChat returns a test server that always responds with the given body
// as the assistant completion content.
func fakeChat(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"role": "assistant", "content": body}, "finish_reason": "stop"},
			},
			"model": "fake-chat",
		})
	}))
}

func TestStatusReportsCompressAndDraftEndpoints(t *testing.T) {
	embedSrv := fakeEmbed(t, 16)
	defer embedSrv.Close()

	compressSrv := fakeChat(t, "summary")
	defer compressSrv.Close()
	draftSrv := fakeChat(t, "draft code")
	defer draftSrv.Close()

	s := &Server{
		EmbedClient:    embed.New(embedSrv.URL, "fake-embed", 16, 5*time.Second),
		CompressClient: chat.New(compressSrv.URL, "fake-compress", 5*time.Second),
		DraftClient:    chat.New(draftSrv.URL, "fake-draft", 5*time.Second),
		IndexDir:       t.TempDir(),
	}
	_, out, _ := s.status(context.Background(), nil, StatusInput{})

	if out.CompressEndpoint != compressSrv.URL {
		t.Errorf("CompressEndpoint = %q, want %q", out.CompressEndpoint, compressSrv.URL)
	}
	if out.CompressModel != "fake-compress" {
		t.Errorf("CompressModel = %q, want fake-compress", out.CompressModel)
	}
	if !out.CompressReachable {
		t.Error("CompressReachable = false, want true")
	}
	if out.DraftEndpoint != draftSrv.URL {
		t.Errorf("DraftEndpoint = %q, want %q", out.DraftEndpoint, draftSrv.URL)
	}
	if out.DraftModel != "fake-draft" {
		t.Errorf("DraftModel = %q, want fake-draft", out.DraftModel)
	}
	if !out.DraftReachable {
		t.Error("DraftReachable = false, want true")
	}
}

func TestStatusOmitsCompressAndDraftWhenUnwired(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	s := newServer(srv.URL, t.TempDir())

	_, out, _ := s.status(context.Background(), nil, StatusInput{})
	if out.CompressEndpoint != "" || out.CompressModel != "" || out.CompressReachable {
		t.Errorf("compress fields should be zero when CompressClient is nil; got endpoint=%q model=%q reachable=%v",
			out.CompressEndpoint, out.CompressModel, out.CompressReachable)
	}
	if out.DraftEndpoint != "" || out.DraftModel != "" || out.DraftReachable {
		t.Errorf("draft fields should be zero when DraftClient is nil; got endpoint=%q model=%q reachable=%v",
			out.DraftEndpoint, out.DraftModel, out.DraftReachable)
	}
}

// clamp returns n if n < max, else max. Used in test error messages to
// truncate long strings without importing min from math.
func clamp(max, n int) int {
	if n < max {
		return n
	}
	return max
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	n := len(b)
	for i > 0 {
		n--
		b[n] = byte('0' + i%10)
		i /= 10
	}
	return string(b[n:])
}

func TestFindSymbolNotFoundSurfaceCandidates(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "main.go"),
		"package main\n\nfunc Indexer() {}\nfunc IndexableExt() {}\nfunc cmdIndex() {}\n")
	root := indexProject(t, projDir, cacheDir, srv.URL)

	s := newServer(srv.URL, cacheDir)
	_, out, err := s.findSymbol(context.Background(), nil, FindSymbolInput{
		Name:        "Index", // no exact match, but substring of several
		ProjectRoot: root,
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != "not-found" {
		t.Errorf("status=%q, want not-found", out.Status)
	}
	if !strings.Contains(out.Hint, "Did you mean") {
		t.Errorf("hint should propose near-misses; got %q", out.Hint)
	}
	// At least one real candidate should appear by name.
	if !strings.Contains(out.Hint, "Indexer") && !strings.Contains(out.Hint, "IndexableExt") && !strings.Contains(out.Hint, "cmdIndex") {
		t.Errorf("hint should name a real candidate; got %q", out.Hint)
	}
}
