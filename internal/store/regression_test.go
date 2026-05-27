package store

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/rerank"
)

// Retrieval regression harness. Builds a fixed synthetic corpus under
// testdata/regression/, runs every queries.jsonl entry through the
// semantic, BM25, fused, and (stub-driven) rerank legs, and asserts the
// expected paths land in fused top-k. Per-leg ranks are printed on
// failure so a regression in any one leg is loud and pinpointed.
//
// The mock embedder is deliberately decoupled from chunk content: it
// maps a `tags` slice to a normalized vector via two-hash projection
// into a 64-dim space. Chunks store one tag set; queries store another;
// content (what BM25 sees) is independent. That decoupling lets the
// fixture exercise cases where one leg succeeds and the other fails
// without coupling the test to any real embedder's behavior.

const regressionDim = 64

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

func embedRegressionTags(tags []string) []float32 {
	v := make([]float32, regressionDim)
	for _, t := range tags {
		h := fnv.New64a()
		_, _ = h.Write([]byte(strings.ToLower(t)))
		sum := h.Sum64()
		// Two-hash projection: each tag contributes to two basis
		// vectors with independent signs. Cuts single-axis collision
		// rate roughly to zero in a 25-tag vocabulary at dim=64.
		idx1 := sum % regressionDim
		sign1 := float32(1)
		if (sum>>32)&1 == 1 {
			sign1 = -1
		}
		v[idx1] += sign1
		sum2 := sum ^ 0xa5a5a5a5a5a5a5a5
		idx2 := sum2 % regressionDim
		sign2 := float32(1)
		if (sum2>>32)&1 == 1 {
			sign2 = -1
		}
		v[idx2] += sign2
	}
	var n float32
	for _, x := range v {
		n += x * x
	}
	if n == 0 {
		// Store rejects all-zero vectors. Empty-tag queries here would
		// have nothing to score against anyway; planting a sentinel
		// keeps the leg functional and rank deterministic.
		v[0] = 1
		return v
	}
	n = float32(math.Sqrt(float64(n)))
	for i := range v {
		v[i] /= n
	}
	return v
}

func loadJSONL[T any](t *testing.T, path string) []T {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var out []T
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var e T
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("parse %s: %v: %s", path, err, line)
		}
		out = append(out, e)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan %s: %v", path, err)
	}
	return out
}

func indexRegressionCorpus(t *testing.T, st *Store, corpus []regressionCorpusEntry) map[int64]string {
	t.Helper()
	ctx := context.Background()
	rows := make([]PendingChunk, len(corpus))
	for i, c := range corpus {
		rows[i] = PendingChunk{
			Path:       c.Path,
			Kind:       "function_declaration",
			Name:       strings.TrimSuffix(filepath.Base(c.Path), filepath.Ext(c.Path)),
			StartLine:  1,
			EndLine:    1,
			ContentSHA: fmt.Sprintf("sha-%d", i),
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

func formatLegTable(q regressionQueryEntry, semPaths, bmPaths, fusedPaths, rerankPaths []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "query=%q text=%q k=%d expected=%v\n", q.Name, q.Text, q.K, q.ExpectedPaths)
	fmt.Fprintf(&b, "  %-12s | %s\n", "path", "ranks (semantic / bm25 / fused / rerank)")
	for _, ep := range q.ExpectedPaths {
		fmt.Fprintf(&b, "  %-12s | %d / %d / %d / %s\n", ep,
			rankOf(semPaths, ep), rankOf(bmPaths, ep), rankOf(fusedPaths, ep),
			optionalRank(rerankPaths, ep))
	}
	fmt.Fprintf(&b, "  fused top:    %v\n", truncate(fusedPaths, q.K))
	fmt.Fprintf(&b, "  semantic top: %v\n", truncate(semPaths, q.K))
	fmt.Fprintf(&b, "  bm25 top:     %v\n", truncate(bmPaths, q.K))
	if rerankPaths != nil {
		fmt.Fprintf(&b, "  rerank top:   %v\n", truncate(rerankPaths, q.K))
	}
	return b.String()
}

func optionalRank(paths []string, p string) string {
	if paths == nil {
		return "-"
	}
	return fmt.Sprintf("%d", rankOf(paths, p))
}

func truncate(paths []string, n int) []string {
	if len(paths) <= n {
		return paths
	}
	return paths[:n]
}

func runRegressionQuery(t *testing.T, ctx context.Context, st *Store, id2path map[int64]string, q regressionQueryEntry, rerankPaths []string) {
	t.Helper()
	qvec := embedRegressionTags(q.Tags)
	pool := q.K * 5
	if pool < 30 {
		pool = 30
	}
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
	semPaths := pathsFromScored(semScored, id2path)
	bmPaths := pathsFromScored(bmScored, id2path)
	fusedPaths := make([]string, len(hits))
	for i, h := range hits {
		fusedPaths[i] = h.Path
	}
	missing := make([]string, 0, len(q.ExpectedPaths))
	for _, ep := range q.ExpectedPaths {
		if rankOf(fusedPaths, ep) == 0 {
			missing = append(missing, ep)
		}
	}
	if len(missing) > 0 {
		t.Errorf("missing %v from fused top-%d\n%s", missing, q.K, formatLegTable(q, semPaths, bmPaths, fusedPaths, rerankPaths))
	}
}

func TestRetrievalRegressionLegs(t *testing.T) {
	corpus := loadJSONL[regressionCorpusEntry](t, "testdata/regression/corpus.jsonl")
	queries := loadJSONL[regressionQueryEntry](t, "testdata/regression/queries.jsonl")
	if len(corpus) == 0 || len(queries) == 0 {
		t.Fatalf("empty fixture: %d corpus, %d queries", len(corpus), len(queries))
	}
	st, ctx := newStore(t)
	id2path := indexRegressionCorpus(t, st, corpus)
	for _, q := range queries {
		q := q
		t.Run(q.Name, func(t *testing.T) {
			runRegressionQuery(t, ctx, st, id2path, q, nil)
		})
	}
}

// substringReranker scores each doc by counting how many whitespace
// tokens from the query appear as substrings of the doc. Deterministic,
// dependency-free, and gives observable reordering so the rerank leg
// covers meaningful retrieval behavior — not just plumbing.
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
	// Sort descending by score so the wrapper sees the rerank ordering
	// when it reads Score[0] through Score[len-1]. The Store's rerank
	// path actually consumes Index, not order, so sorting is decorative
	// — but it matches what real rerankers return.
	sort.SliceStable(scores, func(i, j int) bool { return scores[i].Score > scores[j].Score })
	return scores, nil
}

func TestRetrievalRegressionRerankLeg(t *testing.T) {
	corpus := loadJSONL[regressionCorpusEntry](t, "testdata/regression/corpus.jsonl")
	queries := loadJSONL[regressionQueryEntry](t, "testdata/regression/queries.jsonl")
	if len(corpus) == 0 || len(queries) == 0 {
		t.Fatalf("empty fixture: %d corpus, %d queries", len(corpus), len(queries))
	}
	ctx := context.Background()
	dbPath := filepath.Join(t.TempDir(), "rerank.db")
	st, err := OpenWith(ctx, dbPath, Options{Reranker: substringReranker{}, RerankPool: 30})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	id2path := indexRegressionCorpus(t, st, corpus)
	var rerankObservations int
	for _, q := range queries {
		q := q
		t.Run(q.Name, func(t *testing.T) {
			qvec := embedRegressionTags(q.Tags)
			hits, err := st.Search(ctx, qvec, q.Text, q.K)
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			pool := q.K * 5
			if pool < 30 {
				pool = 30
			}
			semScored, _ := st.scoreSemantic(ctx, qvec, pool)
			bmScored, _ := st.scoreBM25(ctx, q.Text, pool)
			semPaths := pathsFromScored(semScored, id2path)
			bmPaths := pathsFromScored(bmScored, id2path)
			fusedPaths := make([]string, len(hits))
			for i, h := range hits {
				fusedPaths[i] = h.Path
				if h.RerankScore != 0 {
					rerankObservations++
				}
			}
			missing := make([]string, 0, len(q.ExpectedPaths))
			for _, ep := range q.ExpectedPaths {
				if rankOf(fusedPaths, ep) == 0 {
					missing = append(missing, ep)
				}
			}
			if len(missing) > 0 {
				t.Errorf("rerank-leg missing %v from top-%d\n%s",
					missing, q.K, formatLegTable(q, semPaths, bmPaths, fusedPaths, fusedPaths))
			}
		})
	}
	// Aggregate wiring check. The substring reranker returns 0 for docs
	// containing none of the query terms; some queries legitimately
	// score everything at 0 (no literal overlap). But across the suite
	// at least a few queries DO overlap — if every single hit across
	// every query reports RerankScore==0, the rerank path is wired
	// wrong (or the Score map isn't being threaded into Hit.RerankScore)
	// and that should fail loudly.
	if rerankObservations == 0 {
		t.Errorf("rerank wiring: no Hit.RerankScore set across %d queries despite Reranker configured", len(queries))
	}
}
