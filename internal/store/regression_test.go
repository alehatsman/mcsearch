package store

import (
	"bufio"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/rerank"
)

// Retrieval regression harness. Drives a hand-built fixture through the
// semantic, BM25, fused, and (optionally) rerank legs, asserting that
// every expected path lands in fused top-k per query. On failure it
// prints a per-leg rank table so the regressing leg is pinpointed.
// The mock embedder hashes a tag slice rather than the chunk content,
// so the same fixture can exercise cases where one leg succeeds and
// another fails — which is the whole point of a multi-leg harness.

const (
	regressionDim = 64
	// XOR salt used to derive a second basis axis per tag. Two
	// independently-signed projections cut single-axis collisions to
	// roughly zero in a 25-tag vocabulary at dim=64.
	tagHashSalt = 0xa5a5a5a5a5a5a5a5
	// regressionKind is the chunks.kind value used for every fixture
	// row. Must NOT be "window" — scoreBM25 penalizes window-kind
	// rows at 0.7× weight, which would skew the BM25 leg in ways
	// unrelated to retrieval logic.
	regressionKind = "function_declaration"
)

type regressionCorpusEntry struct {
	Path    string   `json:"path"`
	Content string   `json:"content"`
	Tags    []string `json:"tags"`
}

type regressionQueryEntry struct {
	Name          string   `json:"name"`
	Text          string   `json:"text"`
	Tags          []string `json:"tags"`
	K             int      `json:"k"`
	ExpectedPaths []string `json:"expected_paths"`
}

// embedRegressionTags turns a tag slice into a normalized vector by
// projecting each tag onto two basis axes and L2-normalizing.
func embedRegressionTags(tags []string) []float32 {
	v := make([]float32, regressionDim)
	for _, t := range tags {
		projectTag(v, t)
	}
	if !l2Normalize(v) {
		// Store rejects all-zero vectors. An empty tag set has nothing
		// to score against anyway; a sentinel keeps the leg functional
		// and rank deterministic.
		v[0] = 1
	}
	return v
}

func projectTag(v []float32, tag string) {
	h := fnv.New64a()
	_, _ = h.Write([]byte(strings.ToLower(tag)))
	sum := h.Sum64()
	addAxis(v, sum)
	addAxis(v, sum^tagHashSalt)
}

func addAxis(v []float32, h uint64) {
	sign := float32(1)
	if (h>>32)&1 == 1 {
		sign = -1
	}
	v[h%uint64(len(v))] += sign
}

// l2Normalize divides v by its L2 norm in place. Returns false if v is
// all-zero (norm computation would divide by zero).
func l2Normalize(v []float32) bool {
	var n float32
	for _, x := range v {
		n += x * x
	}
	if n == 0 {
		return false
	}
	n = float32(math.Sqrt(float64(n)))
	for i := range v {
		v[i] /= n
	}
	return true
}

// legPool mirrors the candidate-pool floor used by searchRaw in
// store.go. Tests call scoreSemantic / scoreBM25 directly for per-leg
// visibility and need to ask for the same pool size searchRaw would.
func legPool(k int) int {
	pool := k * 5
	if pool < 30 {
		pool = 30
	}
	return pool
}

func loadCorpus(t *testing.T, path string) []regressionCorpusEntry {
	t.Helper()
	var out []regressionCorpusEntry
	loadJSONL(t, path, func(line string) {
		var e regressionCorpusEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("parse %s: %v: %s", path, err, line)
		}
		out = append(out, e)
	})
	return out
}

func loadQueries(t *testing.T, path string) []regressionQueryEntry {
	t.Helper()
	var out []regressionQueryEntry
	loadJSONL(t, path, func(line string) {
		var e regressionQueryEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("parse %s: %v: %s", path, err, line)
		}
		out = append(out, e)
	})
	return out
}

func loadJSONL(t *testing.T, path string, onLine func(string)) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		onLine(line)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan %s: %v", path, err)
	}
}

func indexRegressionCorpus(t *testing.T, st *Store, corpus []regressionCorpusEntry) map[int64]string {
	t.Helper()
	ctx := context.Background()
	rows := make([]PendingChunk, len(corpus))
	for i, c := range corpus {
		sum := sha1.Sum([]byte(c.Content))
		rows[i] = PendingChunk{
			Path:       c.Path,
			Kind:       regressionKind,
			Name:       strings.TrimSuffix(filepath.Base(c.Path), filepath.Ext(c.Path)),
			StartLine:  1,
			EndLine:    1,
			ContentSHA: hex.EncodeToString(sum[:]),
			Content:    c.Content,
			Vec:        embedRegressionTags(c.Tags),
		}
	}
	if err := st.UpsertMany(ctx, rows, time.Now()); err != nil {
		t.Fatalf("upsert corpus: %v", err)
	}
	dbRows, err := st.db.QueryContext(ctx, `SELECT id, path FROM chunks`)
	if err != nil {
		t.Fatalf("load id->path: %v", err)
	}
	defer dbRows.Close()
	out := make(map[int64]string, len(corpus))
	for dbRows.Next() {
		var id int64
		var p string
		if err := dbRows.Scan(&id, &p); err != nil {
			t.Fatalf("scan id->path: %v", err)
		}
		out[id] = p
	}
	return out
}

func pathsFromScored(scs []scored, id2path map[int64]string) []string {
	out := make([]string, 0, len(scs))
	for _, sc := range scs {
		if p, ok := id2path[sc.id]; ok {
			out = append(out, p)
		}
	}
	return out
}

// rankOf returns the 1-based position of path in paths, or 0 if absent.
func rankOf(paths []string, path string) int {
	for i, p := range paths {
		if p == path {
			return i + 1
		}
	}
	return 0
}

func firstN(paths []string, n int) []string {
	if len(paths) <= n {
		return paths
	}
	return paths[:n]
}

func formatLegTable(q regressionQueryEntry, semPaths, bmPaths, fusedPaths []string, rerankOn bool) string {
	var b strings.Builder
	rerankState := "off"
	if rerankOn {
		rerankState = "on (fused == reranked)"
	}
	fmt.Fprintf(&b, "query=%q text=%q k=%d expected=%v rerank=%s\n", q.Name, q.Text, q.K, q.ExpectedPaths, rerankState)
	fmt.Fprintf(&b, "  %-20s | %s\n", "path", "ranks (semantic / bm25 / fused)")
	for _, ep := range q.ExpectedPaths {
		fmt.Fprintf(&b, "  %-20s | %d / %d / %d\n", ep,
			rankOf(semPaths, ep), rankOf(bmPaths, ep), rankOf(fusedPaths, ep))
	}
	fmt.Fprintf(&b, "  fused top:    %v\n", firstN(fusedPaths, q.K))
	fmt.Fprintf(&b, "  semantic top: %v\n", firstN(semPaths, q.K))
	fmt.Fprintf(&b, "  bm25 top:     %v\n", firstN(bmPaths, q.K))
	return b.String()
}

// queryResult captures everything the assertion and failure-formatter
// need from one query run, so the caller can both check correctness
// and report rerank observations without redoing the work.
type queryResult struct {
	hits       []Hit
	semPaths   []string
	bmPaths    []string
	fusedPaths []string
}

func runQuery(t *testing.T, ctx context.Context, st *Store, id2path map[int64]string, q regressionQueryEntry) queryResult {
	t.Helper()
	qvec := embedRegressionTags(q.Tags)
	pool := legPool(q.K)
	semScored, err := st.scoreSemantic(ctx, qvec, pool)
	if err != nil {
		t.Fatalf("scoreSemantic: %v", err)
	}
	bmScored, err := st.scoreBM25(ctx, q.Text, pool)
	if err != nil {
		t.Fatalf("scoreBM25: %v", err)
	}
	hits, err := st.Search(ctx, qvec, q.Text, q.K)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	fusedPaths := make([]string, len(hits))
	for i, h := range hits {
		fusedPaths[i] = h.Path
	}
	return queryResult{
		hits:       hits,
		semPaths:   pathsFromScored(semScored, id2path),
		bmPaths:    pathsFromScored(bmScored, id2path),
		fusedPaths: fusedPaths,
	}
}

// substringReranker scores each doc by counting query tokens that
// appear as substrings of the doc. Enough to verify the rerank wiring
// without depending on a real reranker service.
type substringReranker struct{}

func (substringReranker) Rerank(_ context.Context, query string, docs []string) ([]rerank.Score, error) {
	terms := strings.Fields(strings.ToLower(query))
	scores := make([]rerank.Score, len(docs))
	for i, d := range docs {
		ld := strings.ToLower(d)
		var n int
		for _, t := range terms {
			if t != "" && strings.Contains(ld, t) {
				n++
			}
		}
		scores[i] = rerank.Score{Index: i, Score: float32(n)}
	}
	return scores, nil
}

func TestRetrievalRegression(t *testing.T) {
	corpus := loadCorpus(t, "testdata/regression/corpus.jsonl")
	queries := loadQueries(t, "testdata/regression/queries.jsonl")
	if len(corpus) == 0 || len(queries) == 0 {
		t.Fatalf("empty fixture: %d corpus, %d queries", len(corpus), len(queries))
	}

	cases := []struct {
		name string
		opts Options
	}{
		{name: "plain", opts: Options{}},
		{name: "rerank", opts: Options{Reranker: substringReranker{}, RerankPool: 30}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			dbPath := filepath.Join(t.TempDir(), "regression.db")
			st, err := OpenWith(ctx, dbPath, tc.opts)
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			t.Cleanup(func() { _ = st.Close() })
			id2path := indexRegressionCorpus(t, st, corpus)
			rerankOn := tc.opts.Reranker != nil

			// queriesRan + rerankObservations track suite-wide state for
			// the rerank-wiring check below. Subtests run sequentially
			// (no t.Parallel), so plain ints are enough; adding
			// parallelism would require atomics.
			var queriesRan, rerankObservations int

			for _, q := range queries {
				t.Run(q.Name, func(t *testing.T) {
					queriesRan++
					res := runQuery(t, ctx, st, id2path, q)
					for _, h := range res.hits {
						if h.RerankScore != 0 {
							rerankObservations++
						}
					}
					var missing []string
					for _, ep := range q.ExpectedPaths {
						if rankOf(res.fusedPaths, ep) == 0 {
							missing = append(missing, ep)
						}
					}
					if len(missing) > 0 {
						t.Errorf("missing %v from fused top-%d\n%s",
							missing, q.K,
							formatLegTable(q, res.semPaths, res.bmPaths, res.fusedPaths, rerankOn))
					}
				})
			}

			// Aggregate rerank-wiring check. Some queries legitimately
			// score everything at 0 (no literal term overlap), but
			// across the whole suite at least a few queries DO overlap.
			// If nothing ever records a non-zero RerankScore, the
			// rerank path is wired wrong.
			//
			// Skipped when -run filtered the subtests so the check
			// doesn't false-fire on a narrowed invocation.
			if rerankOn && queriesRan == len(queries) && rerankObservations == 0 {
				t.Errorf("rerank wiring: no Hit.RerankScore set across %d queries despite Reranker configured", len(queries))
			}
		})
	}
}
