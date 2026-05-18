// Package store persists per-project chunks + embedding vectors.
//
// One SQLite file per project. Vectors are stored as packed little-endian
// float32 BLOBs. Search is brute-force cosine similarity in Go — fast
// enough for <100 k chunks per project. Swap the search path for an HNSW
// index later without changing the rest of the codebase.
//
// Timestamps (last_seen_at, last_indexed_at) are stored as Unix
// nanoseconds rather than milliseconds, so two index runs that complete
// within the same millisecond produce distinct cutoffs — important
// because PruneUnseen relies on strict-less-than comparison to detect
// stale rows.
package store

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/alehatsman/mcsearch/internal/rerank"
	_ "modernc.org/sqlite"
)

// Options influence the runtime behaviour of an opened Store.
// All fields are optional; the zero value matches the default
// (vector cache enabled, hybrid BM25+semantic search enabled).
type Options struct {
	// DisableVecCache turns off the in-RAM decoded-vector slab.
	// Search falls back to per-row SQL scans (slower; bounded RAM).
	// Useful for very large indexes where dim×chunks×4 bytes of
	// cache exceeds available memory.
	DisableVecCache bool

	// DisableBM25 turns off the lexical (FTS5/BM25) leg of hybrid
	// search. Useful for ablation / debugging the semantic ranking,
	// or for indexes built before the chunks_fts migration on a
	// truly old SQLite without FTS5 (unlikely).
	DisableBM25 bool

	// Reranker, when non-nil, reorders the fused candidate pool via a
	// cross-encoder before truncating to k. Nil = today's behaviour
	// (pure RRF). On rerank.ErrUnreachable the search falls back to
	// the pre-rerank order without surfacing an error to the caller.
	Reranker rerank.Reranker

	// RerankPool caps the fused candidate pool sent to the reranker.
	// Only honored when Reranker is non-nil. Zero = no cap (use the
	// natural pool size, max(5×k, 30)). Typical values 40–100: larger
	// = better recall but slower rerank call.
	RerankPool int

	// MaxHitsPerFile, when > 0, caps results returned per unique file path.
	// Applied after ranking, before final truncation to k. Zero = no cap.
	MaxHitsPerFile int
}

type Store struct {
	db   *sql.DB
	dim  int     // vector dimension; discovered on first upsert
	opts Options // immutable after Open

	// Search-side vector cache. Lazily populated on first Search and
	// invalidated by any mutating call (UpsertMany, DeletePath,
	// DeletePathPrefix, PruneUnseen). Holding decoded vectors in RAM
	// trades up to ~dim*4 bytes per chunk for a 5–10× speedup and a
	// ~30× allocation reduction on Search — the typical MCP server
	// runs many queries against the same Store, so the load cost
	// amortizes immediately.
	//
	// Memory: for a 100k-chunk index at 1024 dim, this is ~400 MB.
	// Acceptable for our target "one project per server" deployment;
	// callers worried about footprint can set Options.DisableVecCache
	// to fall back to the per-row SQL hot path.
	cacheMu        sync.RWMutex
	cacheLoaded    bool
	cacheIndexedAt int64   // last_indexed_at (nanoseconds) when the cache was built
	cacheIDs       []int64
	cacheVecs      []float32 // flat [len(cacheIDs) * dim]
	cacheNorms     []float32 // precomputed |v| per row, zero-norm rows skipped at load time
}

// Open opens or creates the SQLite file at path with default
// Options. Convenience wrapper around OpenWith.
func Open(ctx context.Context, path string) (*Store, error) {
	return OpenWith(ctx, path, Options{})
}

// OpenWith is like Open but lets the caller adjust runtime behaviour
// (e.g. disable the in-RAM vector cache).
//
// `busy_timeout(5000)` lets concurrent writers (e.g. `mcsearch index`
// fired while `mcsearch watch` is also re-indexing) wait up to 5 s for
// the writer lock instead of immediately returning SQLITE_BUSY. Without
// it, racing index runs both crash with a leaked DDL error.
func OpenWith(ctx context.Context, path string, opts Options) (*Store, error) {
	db, err := sql.Open("sqlite",
		"file:"+path+
			"?_pragma=journal_mode(WAL)"+
			"&_pragma=synchronous(NORMAL)"+
			"&_pragma=busy_timeout(5000)"+
			"&_pragma=foreign_keys(1)")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	s := &Store{db: db, opts: opts}
	if err := s.migrate(ctx); err != nil {
		db.Close()
		return nil, err
	}
	// Recover the recorded vector dimension, if any.
	row := db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='dim'`)
	var v string
	switch err := row.Scan(&v); {
	case errors.Is(err, sql.ErrNoRows):
		// fresh db; dim discovered on first Upsert
	case err != nil:
		db.Close()
		return nil, err
	default:
		var dim int
		fmt.Sscanf(v, "%d", &dim)
		s.dim = dim
	}
	return s, nil
}

func (s *Store) Close() error { return s.db.Close() }

func (s *Store) migrate(ctx context.Context) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS meta (
		   key   TEXT PRIMARY KEY,
		   value TEXT NOT NULL
		 )`,
		`CREATE TABLE IF NOT EXISTS chunks (
		   id            INTEGER PRIMARY KEY,
		   path          TEXT NOT NULL,
		   kind          TEXT NOT NULL,
		   name          TEXT NOT NULL DEFAULT '',
		   start_line    INTEGER NOT NULL,
		   end_line      INTEGER NOT NULL,
		   content_sha1  TEXT NOT NULL,
		   content       TEXT NOT NULL,
		   vec           BLOB NOT NULL,
		   last_seen_at  INTEGER NOT NULL,
		   UNIQUE(path, content_sha1)
		 )`,
		`CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)`,
		`CREATE INDEX IF NOT EXISTS idx_chunks_last_seen ON chunks(last_seen_at)`,
		// FTS5 external-content index. Doesn't duplicate chunk text on
		// disk — it references chunks.content by rowid=chunks.id and
		// only persists tokenizer state. We keep it in sync via AFTER
		// triggers on chunks. Hybrid Search fuses cosine ranking with
		// BM25 ranking over this index via RRF.
		`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
		   content, path, kind,
		   content='chunks', content_rowid='id',
		   tokenize='unicode61 remove_diacritics 2'
		 )`,
		`CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
		   INSERT INTO chunks_fts(rowid, content, path, kind)
		   VALUES (new.id, new.content, new.path, new.kind);
		 END`,
		`CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
		   INSERT INTO chunks_fts(chunks_fts, rowid, content, path, kind)
		   VALUES('delete', old.id, old.content, old.path, old.kind);
		 END`,
		`CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
		   INSERT INTO chunks_fts(chunks_fts, rowid, content, path, kind)
		   VALUES('delete', old.id, old.content, old.path, old.kind);
		   INSERT INTO chunks_fts(rowid, content, path, kind)
		   VALUES (new.id, new.content, new.path, new.kind);
		 END`,
	}
	for _, q := range stmts {
		if _, err := s.db.ExecContext(ctx, q); err != nil {
			return fmt.Errorf("migrate: %w (%s)", err, q)
		}
	}
	// Backfill chunks_fts for databases that pre-date the hybrid search
	// migration. Cheap on first-run (one INSERT-from-SELECT batch);
	// guarded by a meta flag so we don't pay it on every Open.
	var built string
	_ = s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='fts_built'`).Scan(&built)
	if built != "1" {
		if _, err := s.db.ExecContext(ctx,
			`INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')`); err != nil {
			return fmt.Errorf("migrate: fts rebuild: %w", err)
		}
		if _, err := s.db.ExecContext(ctx,
			`INSERT INTO meta(key, value) VALUES('fts_built', '1')
			 ON CONFLICT(key) DO UPDATE SET value=excluded.value`); err != nil {
			return fmt.Errorf("migrate: fts flag: %w", err)
		}
	}
	// Add name column to existing databases that pre-date this migration.
	// On fresh databases the column already exists from CREATE TABLE, so
	// we silently ignore "duplicate column name" errors from ALTER TABLE.
	var nameColAdded string
	_ = s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='name_col_added'`).Scan(&nameColAdded)
	if nameColAdded != "1" {
		if _, err := s.db.ExecContext(ctx,
			`ALTER TABLE chunks ADD COLUMN name TEXT NOT NULL DEFAULT ''`); err != nil &&
			!strings.Contains(err.Error(), "duplicate column name") {
			return fmt.Errorf("migrate: add name column: %w", err)
		}
		if _, err := s.db.ExecContext(ctx,
			`INSERT INTO meta(key, value) VALUES('name_col_added', '1')
			 ON CONFLICT(key) DO UPDATE SET value=excluded.value`); err != nil {
			return fmt.Errorf("migrate: name_col flag: %w", err)
		}
	}
	return nil
}

// Stats reports the current state of an index.
type Stats struct {
	Chunks    int
	Files     int
	Dim       int
	LastIndex time.Time
}

func (s *Store) Stats(ctx context.Context) (Stats, error) {
	var st Stats
	st.Dim = s.dim
	row := s.db.QueryRowContext(ctx, `SELECT COUNT(*), COUNT(DISTINCT path) FROM chunks`)
	if err := row.Scan(&st.Chunks, &st.Files); err != nil {
		return st, err
	}
	row = s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='last_indexed_at'`)
	var v string
	if err := row.Scan(&v); err == nil {
		var ts int64
		fmt.Sscanf(v, "%d", &ts)
		if ts > 0 {
			st.LastIndex = time.Unix(0, ts)
		}
	}
	return st, nil
}

// SetLastIndexedAt records the wall-clock time of the most recent
// successful (full or incremental) re-index.
func (s *Store) SetLastIndexedAt(ctx context.Context, t time.Time) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO meta(key,value) VALUES('last_indexed_at', ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		fmt.Sprintf("%d", t.UnixNano()))
	return err
}

// SetProjectRoot records the absolute project path this index belongs
// to. Needed by `reindex --all`, which walks the sha256(path)-keyed
// cache dirs and has to recover each project's original on-disk root.
func (s *Store) SetProjectRoot(ctx context.Context, root string) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO meta(key,value) VALUES('project_root', ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`, root)
	return err
}

// ProjectRoot returns the path previously recorded by SetProjectRoot.
// Returns "" (not an error) if the row is missing — that's the
// pre-migration case for indexes built before this metadata existed.
func (s *Store) ProjectRoot(ctx context.Context) (string, error) {
	var v string
	row := s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='project_root'`)
	if err := row.Scan(&v); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", nil
		}
		return "", err
	}
	return v, nil
}

// ExistingSHAs returns the set of content_sha1 already present for path,
// so the indexer can skip re-embedding unchanged chunks.
func (s *Store) ExistingSHAs(ctx context.Context, path string) (map[string]bool, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT content_sha1 FROM chunks WHERE path=?`, path)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]bool{}
	for rows.Next() {
		var sha string
		if err := rows.Scan(&sha); err != nil {
			return nil, err
		}
		out[sha] = true
	}
	return out, rows.Err()
}

// PendingChunk is one row destined for an UpsertMany batch.
type PendingChunk struct {
	Path       string
	Kind       string
	Name       string
	StartLine  int
	EndLine    int
	ContentSHA string
	Content    string
	Vec        []float32
}

// UpsertMany inserts a batch of chunks in a single transaction. One
// commit per batch instead of one commit per chunk drops the no-op
// fsync count by ~32× on a typical run and is well worth the slight
// API duplication.
func (s *Store) UpsertMany(ctx context.Context, rows []PendingChunk, now time.Time) error {
	if len(rows) == 0 {
		return nil
	}
	if s.dim == 0 {
		s.dim = len(rows[0].Vec)
		if _, err := s.db.ExecContext(ctx,
			`INSERT INTO meta(key,value) VALUES('dim', ?)
			 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
			fmt.Sprintf("%d", s.dim)); err != nil {
			return err
		}
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO chunks(path, kind, name, start_line, end_line, content_sha1, content, vec, last_seen_at)
		 VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(path, content_sha1) DO UPDATE SET
		   kind=excluded.kind,
		   name=excluded.name,
		   start_line=excluded.start_line,
		   end_line=excluded.end_line,
		   content=excluded.content,
		   vec=excluded.vec,
		   last_seen_at=excluded.last_seen_at`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	defer stmt.Close()
	for _, r := range rows {
		if len(r.Vec) != s.dim {
			_ = tx.Rollback()
			return fmt.Errorf("vector dim mismatch: index has dim=%d, got %d (did the embedding model change?)", s.dim, len(r.Vec))
		}
		if _, err := stmt.ExecContext(ctx,
			r.Path, r.Kind, r.Name, r.StartLine, r.EndLine, r.ContentSHA, r.Content,
			encodeVec(r.Vec), now.UnixNano()); err != nil {
			_ = tx.Rollback()
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	s.invalidateCache()
	return nil
}

// TouchSeen bumps last_seen_at for an already-present (path, sha) pair.
// Used when we re-walk a project and encounter unchanged content.
func (s *Store) TouchSeen(ctx context.Context, path, contentSHA string, now time.Time) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE chunks SET last_seen_at=? WHERE path=? AND content_sha1=?`,
		now.UnixNano(), path, contentSHA)
	return err
}

// TouchPath bumps last_seen_at for every chunk of a single file in one
// statement. Used by the mtime fast-path: when a file hasn't changed
// since the previous successful index, we don't need to read it or
// re-chunk it — we just have to mark its chunks live so PruneUnseen
// doesn't drop them. Returns the number of rows touched (0 means the
// file has no chunks yet — caller must fall back to the slow path).
func (s *Store) TouchPath(ctx context.Context, path string, now time.Time) (int64, error) {
	res, err := s.db.ExecContext(ctx,
		`UPDATE chunks SET last_seen_at=? WHERE path=?`,
		now.UnixNano(), path)
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

// PruneUnseen deletes chunks last seen before `cutoff`. Call at the end
// of a re-index to remove stale rows for files that disappeared.
func (s *Store) PruneUnseen(ctx context.Context, cutoff time.Time) (int64, error) {
	res, err := s.db.ExecContext(ctx,
		`DELETE FROM chunks WHERE last_seen_at < ?`, cutoff.UnixNano())
	if err != nil {
		return 0, err
	}
	n, err := res.RowsAffected()
	if err == nil && n > 0 {
		s.invalidateCache()
	}
	return n, err
}

// DeletePath drops all chunks for a single relative path.
func (s *Store) DeletePath(ctx context.Context, path string) error {
	if _, err := s.db.ExecContext(ctx, `DELETE FROM chunks WHERE path=?`, path); err != nil {
		return err
	}
	s.invalidateCache()
	return nil
}

// DeletePathPrefix drops all chunks whose path starts with prefix.
// Used by the indexer to evict chunks under a directory that has
// become ignored between runs (e.g. a fresh `node_modules/` entry).
func (s *Store) DeletePathPrefix(ctx context.Context, prefix string) error {
	if prefix == "" {
		return nil
	}
	if _, err := s.db.ExecContext(ctx,
		`DELETE FROM chunks WHERE path LIKE ? ESCAPE '\'`,
		escapeLike(prefix)+`%`); err != nil {
		return err
	}
	s.invalidateCache()
	return nil
}

// escapeLike escapes the LIKE-pattern metacharacters in s.
func escapeLike(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `%`, `\%`)
	s = strings.ReplaceAll(s, `_`, `\_`)
	return s
}

// Hit is one search result.
type Hit struct {
	Path      string
	Kind      string
	Name      string
	StartLine int
	EndLine   int
	Content   string

	// Score is the cosine similarity in [-1, 1] (1.0 == identical
	// direction). Always populated, even for hits that surfaced via
	// the BM25 path — useful as a familiar "is this close?" number
	// for humans and for downstream filtering.
	Score float32

	// BM25Score is the FTS5 bm25() rank when the hit surfaced through
	// the lexical path. SQLite returns these as small negative
	// numbers (more negative = better); we negate so larger = better.
	// Zero when the hit didn't match the BM25 query at all.
	BM25Score float32

	// RRFScore is the fused rank used for ordering when hybrid search
	// is active: 1/(60+sem_rank) + 1/(60+bm25_rank). Zero when search
	// ran semantic-only (empty query text or MCSEARCH_DISABLE_BM25=1).
	RRFScore float32

	// RerankScore is the cross-encoder relevance score in [0, 1] for
	// the (query, chunk) pair. Zero when rerank didn't run (no client
	// wired, pool ≤ k, or endpoint unreachable). Larger = more relevant.
	RerankScore float32
}

// FormatHits renders a slice of hits as a fenced CONTEXT block for
// injection into a chat completion message. Each chunk gets a header
// with path:line coordinates so the model can cite real locations.
func FormatHits(hits []Hit) string {
	var b strings.Builder
	b.WriteString("CONTEXT — relevant chunks from the project's mcsearch index:\n\n")
	for i, h := range hits {
		fmt.Fprintf(&b, "--- chunk %d: %s:%d-%d (%s, score=%.4f) ---\n",
			i+1, h.Path, h.StartLine, h.EndLine, h.Kind, h.Score)
		b.WriteString(h.Content)
		if !strings.HasSuffix(h.Content, "\n") {
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}

// dedupChunkSummaries drops chunk_summary hits whose source chunk already
// appears at the same path:start_line in the result set. When --summarize
// was used during indexing both a code chunk and its prose summary land in
// the index; without dedup they consume two top-k slots for the same
// function. Ordering is preserved.
func dedupChunkSummaries(hits []Hit) []Hit {
	type locKey struct {
		path string
		line int
	}
	codeAt := make(map[locKey]bool, len(hits))
	for _, h := range hits {
		if h.Kind != "chunk_summary" {
			codeAt[locKey{h.Path, h.StartLine}] = true
		}
	}
	out := make([]Hit, 0, len(hits))
	for _, h := range hits {
		if h.Kind == "chunk_summary" && codeAt[locKey{h.Path, h.StartLine}] {
			continue
		}
		out = append(out, h)
	}
	return out
}

// scored holds one chunk's score during ranking. Used internally by
// both the semantic and BM25 legs; the RRF fuser then walks both lists.
type scored struct {
	id    int64
	score float32 // cosine for semantic; -bm25() for BM25 (larger = better)
}

// rrfK is the RRF dampening constant. 60 is the canonical default from
// Cormack et al. (2009); behavior is robust to values in [10, 100].
const rrfK = 60

// Search returns the top-k chunks ranked by hybrid scoring with optional
// per-file diversity via Options.MaxHitsPerFile.
func (s *Store) Search(ctx context.Context, queryVec []float32, queryText string, k int) ([]Hit, error) {
	hits, err := s.searchRaw(ctx, queryVec, queryText, k)
	if err != nil || len(hits) == 0 || s.opts.MaxHitsPerFile <= 0 {
		return hits, err
	}
	return diversify(hits, s.opts.MaxHitsPerFile), nil
}

// diversify caps the number of hits per unique file path, preserving
// the existing score-based ordering. Hits beyond the cap are dropped.
func diversify(hits []Hit, maxPerFile int) []Hit {
	counts := make(map[string]int, len(hits)/2)
	out := make([]Hit, 0, len(hits))
	for _, h := range hits {
		if counts[h.Path] >= maxPerFile {
			continue
		}
		counts[h.Path]++
		out = append(out, h)
	}
	return out
}

// searchRaw is the internal search implementation. See Search for the
// public API. When `queryText` is non-empty AND BM25 isn't disabled,
// results from the cosine path and the FTS5/BM25 path are fused via
// Reciprocal Rank Fusion: rrf_score(id) = Σ 1/(60+rank_in_list). RRF
// is scale-free, so the two heterogenous scoring schemes compose without
// per-corpus tuning. When `queryText` is empty (or BM25 disabled),
// search degrades to semantic-only.
func (s *Store) searchRaw(ctx context.Context, queryVec []float32, queryText string, k int) ([]Hit, error) {
	if k <= 0 {
		k = 8
	}
	semScores, err := s.scoreSemantic(ctx, queryVec)
	if err != nil {
		return nil, err
	}
	useBM25 := !s.opts.DisableBM25 && strings.TrimSpace(queryText) != ""

	if !useBM25 {
		// Semantic-only path. Sort + cap before fetching content.
		if len(semScores) == 0 {
			return nil, nil
		}
		sort.Slice(semScores, func(i, j int) bool { return semScores[i].score > semScores[j].score })
		if len(semScores) > k {
			semScores = semScores[:k]
		}
		return s.fetchHits(ctx, semScores, nil, nil, nil, nil)
	}

	// Pull more candidates per leg than the final k so fusion has
	// headroom to surface lexical-only or semantic-only hits.
	pool := k * 5
	if pool < 30 {
		pool = 30
	}
	// When a reranker is wired, cap the pool so we don't pay
	// cross-encoder cost on more docs than the operator chose.
	if s.opts.Reranker != nil && s.opts.RerankPool > 0 && pool > s.opts.RerankPool {
		pool = s.opts.RerankPool
	}

	// Semantic top-pool.
	semSorted := semScores
	sort.Slice(semSorted, func(i, j int) bool { return semSorted[i].score > semSorted[j].score })
	if len(semSorted) > pool {
		semSorted = semSorted[:pool]
	}
	semCosine := make(map[int64]float32, len(semSorted))
	semRank := make(map[int64]int, len(semSorted))
	for i, sc := range semSorted {
		semCosine[sc.id] = sc.score
		semRank[sc.id] = i + 1
	}

	// BM25 top-pool.
	bm25Scores, err := s.scoreBM25(ctx, queryText, pool)
	if err != nil {
		// If FTS5 chokes on the query (e.g. unbalanced quotes), fall
		// back to semantic-only rather than failing the user's search.
		bm25Scores = nil
	}
	bm25Rank := make(map[int64]int, len(bm25Scores))
	bm25Score := make(map[int64]float32, len(bm25Scores))
	for i, sc := range bm25Scores {
		bm25Rank[sc.id] = i + 1
		bm25Score[sc.id] = sc.score
	}

	// Fuse via RRF.
	rrf := make(map[int64]float32, len(semRank)+len(bm25Rank))
	for id, r := range semRank {
		rrf[id] += 1.0 / float32(rrfK+r)
	}
	for id, r := range bm25Rank {
		rrf[id] += 1.0 / float32(rrfK+r)
	}
	fused := make([]scored, 0, len(rrf))
	for id, r := range rrf {
		fused = append(fused, scored{id, r})
	}
	sort.Slice(fused, func(i, j int) bool { return fused[i].score > fused[j].score })

	// Rerank stage: only fires if a client is wired and we actually
	// have more candidates than k (otherwise reordering is a no-op).
	// On ErrUnreachable, fall through to the pre-rerank truncation so
	// reranker outages never surface as search failures.
	if s.opts.Reranker != nil && len(fused) > k {
		reranked, rerankScore, err := s.rerank(ctx, queryText, fused, k)
		switch {
		case err == nil:
			return s.fetchHits(ctx, reranked, semCosine, bm25Score, rrf, rerankScore)
		case errors.Is(err, rerank.ErrUnreachable):
			// fall through to non-reranked path
		default:
			return nil, err
		}
	}

	if len(fused) > k {
		fused = fused[:k]
	}
	return s.fetchHits(ctx, fused, semCosine, bm25Score, nil, nil)
}

// rerank fetches `Content` for the fused pool, sends (query, docs) to
// the reranker, maps the returned indices back to chunk IDs, and
// returns the top-k slice together with a per-id rerank score map.
func (s *Store) rerank(ctx context.Context, queryText string, fused []scored, k int) ([]scored, map[int64]float32, error) {
	if len(fused) == 0 {
		return nil, nil, nil
	}
	idArgs := make([]any, len(fused))
	for i, sc := range fused {
		idArgs[i] = sc.id
	}
	placeholders := strings.Repeat("?,", len(idArgs))
	placeholders = placeholders[:len(placeholders)-1]
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, content FROM chunks WHERE id IN (`+placeholders+`)`,
		idArgs...)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	contentByID := make(map[int64]string, len(fused))
	for rows.Next() {
		var id int64
		var content string
		if err := rows.Scan(&id, &content); err != nil {
			return nil, nil, err
		}
		contentByID[id] = content
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	// Build docs in fused order so rerank.Score.Index maps cleanly back.
	docs := make([]string, 0, len(fused))
	docIDs := make([]int64, 0, len(fused))
	for _, sc := range fused {
		c, ok := contentByID[sc.id]
		if !ok {
			continue // chunk vanished between fusion and content fetch
		}
		docs = append(docs, c)
		docIDs = append(docIDs, sc.id)
	}
	scores, err := s.opts.Reranker.Rerank(ctx, queryText, docs)
	if err != nil {
		return nil, nil, err
	}
	reranked := make([]scored, 0, len(scores))
	rerankScore := make(map[int64]float32, len(scores))
	for _, sc := range scores {
		if sc.Index < 0 || sc.Index >= len(docIDs) {
			continue
		}
		id := docIDs[sc.Index]
		reranked = append(reranked, scored{id: id, score: sc.Score})
		rerankScore[id] = sc.Score
	}
	if len(reranked) > k {
		reranked = reranked[:k]
	}
	return reranked, rerankScore, nil
}

// scoreSemantic computes cosine similarity for every chunk against
// queryVec. Returns one scored row per chunk (unsorted). Uses the
// in-RAM cache when enabled, else streams from SQL.
func (s *Store) scoreSemantic(ctx context.Context, queryVec []float32) ([]scored, error) {
	if s.dim != 0 && len(queryVec) != s.dim {
		return nil, fmt.Errorf("query dim %d != index dim %d", len(queryVec), s.dim)
	}
	qNorm := norm(queryVec)
	if qNorm == 0 {
		return nil, fmt.Errorf("query vector is zero")
	}

	if s.opts.DisableVecCache {
		rows, err := s.db.QueryContext(ctx, `SELECT id, vec FROM chunks`)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		out := make([]scored, 0, 1024)
		var vbuf []float32
		for rows.Next() {
			var id int64
			var blob []byte
			if err := rows.Scan(&id, &blob); err != nil {
				return nil, err
			}
			if len(blob)%4 != 0 {
				return nil, fmt.Errorf("vec blob length %d not divisible by 4", len(blob))
			}
			need := len(blob) / 4
			if cap(vbuf) < need {
				vbuf = make([]float32, need)
			} else {
				vbuf = vbuf[:need]
			}
			for i := range vbuf {
				vbuf[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
			}
			dot := float32(0)
			var vNormSq float32
			for i, qv := range queryVec {
				vi := vbuf[i]
				dot += qv * vi
				vNormSq += vi * vi
			}
			if vNormSq == 0 {
				continue
			}
			out = append(out, scored{id, dot / (qNorm * float32(math.Sqrt(float64(vNormSq))))})
		}
		return out, rows.Err()
	}

	if err := s.ensureCache(ctx); err != nil {
		return nil, err
	}
	s.cacheMu.RLock()
	ids := s.cacheIDs
	vecs := s.cacheVecs
	norms := s.cacheNorms
	dim := s.dim
	s.cacheMu.RUnlock()
	if len(ids) == 0 || dim == 0 {
		return nil, nil
	}
	out := make([]scored, len(ids))
	for i, id := range ids {
		off := i * dim
		dot := float32(0)
		for j, qv := range queryVec {
			dot += qv * vecs[off+j]
		}
		out[i] = scored{id, dot / (qNorm * norms[i])}
	}
	return out, nil
}

// scoreBM25 runs the FTS5 / BM25 leg of hybrid search. Returns the
// top-`limit` chunk IDs ordered by BM25 rank (best first), with the
// score field set to -bm25() (so larger = better, consistent with the
// cosine path's convention).
//
// Kind weighting: bm25() returns negative numbers (more negative =
// better). Multiplying by 0.7 for `window` chunks (free-form line
// slices, dominated by Markdown/README content) pushes them toward
// zero — i.e. worse rank — so a README that happens to list every
// identifier the codebase exposes can't crowd out the actual
// definition site. Structural chunks (function_declaration etc.) and
// `orphan` chunks (top-level const/var/import we'd lose otherwise)
// keep their full BM25 weight.
func (s *Store) scoreBM25(ctx context.Context, queryText string, limit int) ([]scored, error) {
	matchExpr := buildFTSQuery(queryText)
	if matchExpr == "" {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT chunks_fts.rowid,
		        bm25(chunks_fts) * CASE chunks.kind
		            WHEN 'window' THEN 0.7
		            ELSE 1.0
		          END AS weighted_rank
		   FROM chunks_fts
		   JOIN chunks ON chunks.id = chunks_fts.rowid
		   WHERE chunks_fts MATCH ?
		   ORDER BY weighted_rank
		   LIMIT ?`,
		matchExpr, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]scored, 0, limit)
	for rows.Next() {
		var id int64
		var bm float64
		if err := rows.Scan(&id, &bm); err != nil {
			return nil, err
		}
		// bm25() returns negative rank by convention (smaller = better).
		// Flip the sign so larger = better, matching cosine.
		out = append(out, scored{id, float32(-bm)})
	}
	return out, rows.Err()
}

// buildFTSQuery turns a natural-language query into an FTS5 MATCH
// expression. We split on whitespace, lower-case each token, drop
// anything that isn't safe to embed as a quoted FTS5 string, and OR
// them together. The OR (vs FTS5's default AND) trades precision for
// recall — bad lexical matches are sunk by their BM25 rank anyway,
// while AND would too often return zero hits on natural-language
// queries like "function that validates a token".
func buildFTSQuery(q string) string {
	var toks []string
	for _, f := range strings.Fields(q) {
		// Strip surrounding punctuation so "validateToken." behaves
		// like "validateToken". Keep internal alphanumerics + `_`.
		var b strings.Builder
		for _, r := range f {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
				(r >= '0' && r <= '9') || r == '_' {
				b.WriteRune(r)
			}
		}
		t := b.String()
		if len(t) < 2 {
			continue
		}
		toks = append(toks, `"`+t+`"`)
	}
	if len(toks) == 0 {
		return ""
	}
	return strings.Join(toks, " OR ")
}

// fetchHits issues one SELECT to get content for the supplied
// (ordered) scored IDs, then returns them as Hits with their scores
// joined back in.
//
//   - semCosine / bm25Score: nil for semantic-only mode.
//   - rrfScore: non-nil only on the reranked path, where `ranked[i].score`
//     carries the rerank score instead of the RRF score. When nil and
//     semCosine is non-nil, fetchHits falls back to using ranked[i].score
//     as the RRF score (today's pre-rerank hybrid behaviour).
//   - rerankScore: non-nil only on the reranked path; populates
//     Hit.RerankScore.
func (s *Store) fetchHits(ctx context.Context, ranked []scored, semCosine, bm25Score, rrfScore, rerankScore map[int64]float32) ([]Hit, error) {
	if len(ranked) == 0 {
		return nil, nil
	}
	idArgs := make([]any, len(ranked))
	for i, sc := range ranked {
		idArgs[i] = sc.id
	}
	placeholders := strings.Repeat("?,", len(idArgs))
	placeholders = placeholders[:len(placeholders)-1]
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, path, kind, name, start_line, end_line, content FROM chunks WHERE id IN (`+placeholders+`)`,
		idArgs...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	byID := make(map[int64]Hit, len(ranked))
	for rows.Next() {
		var id int64
		var h Hit
		if err := rows.Scan(&id, &h.Path, &h.Kind, &h.Name, &h.StartLine, &h.EndLine, &h.Content); err != nil {
			return nil, err
		}
		byID[id] = h
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	out := make([]Hit, 0, len(ranked))
	for _, sc := range ranked {
		h, ok := byID[sc.id]
		if !ok {
			continue
		}
		// Score is always the cosine (what humans read). RRFScore /
		// BM25Score / RerankScore are filled in when hybrid (and
		// possibly reranked) mode produced them.
		if semCosine != nil {
			if c, ok := semCosine[sc.id]; ok {
				h.Score = c
			}
			if rrfScore != nil {
				// Reranked path: ranked[i].score is the rerank score, so
				// RRFScore must come from the explicit map.
				h.RRFScore = rrfScore[sc.id]
			} else {
				h.RRFScore = sc.score
			}
		} else {
			h.Score = sc.score
		}
		if bm25Score != nil {
			h.BM25Score = bm25Score[sc.id]
		}
		if rerankScore != nil {
			h.RerankScore = rerankScore[sc.id]
		}
		out = append(out, h)
	}
	return dedupChunkSummaries(out), nil
}

// FindSymbol returns chunks whose `name` column exactly matches the
// given identifier. Results are ordered by (path, start_line). Uses a
// SQL index scan — no embedding required — so it is fast regardless of
// index size.
func (s *Store) FindSymbol(ctx context.Context, name string, k int) ([]Hit, error) {
	if k <= 0 {
		k = 10
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, path, kind, name, start_line, end_line, content
		 FROM chunks WHERE name = ?
		 ORDER BY path, start_line LIMIT ?`,
		name, k)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Hit
	for rows.Next() {
		var id int64
		var h Hit
		if err := rows.Scan(&id, &h.Path, &h.Kind, &h.Name, &h.StartLine, &h.EndLine, &h.Content); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// RelatedChunks returns the top-k chunks most similar to the chunk at
// (path, startLine), excluding the source chunk itself. Uses the in-RAM
// vector cache for speed. Returns an error if no chunk is found at the
// given location.
func (s *Store) RelatedChunks(ctx context.Context, path string, startLine int, k int) ([]Hit, error) {
	if k <= 0 {
		k = 8
	}
	var blob []byte
	var sourceID int64
	row := s.db.QueryRowContext(ctx,
		`SELECT id, vec FROM chunks WHERE path = ? AND start_line = ?`,
		path, startLine)
	if err := row.Scan(&sourceID, &blob); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("no chunk at %s:%d", path, startLine)
		}
		return nil, err
	}
	if len(blob)%4 != 0 {
		return nil, fmt.Errorf("vec blob length %d not divisible by 4", len(blob))
	}
	dim := len(blob) / 4
	queryVec := make([]float32, dim)
	for i := range dim {
		queryVec[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
	}
	if err := s.ensureCache(ctx); err != nil {
		return nil, err
	}
	s.cacheMu.RLock()
	ids := s.cacheIDs
	vecs := s.cacheVecs
	norms := s.cacheNorms
	cDim := s.dim
	s.cacheMu.RUnlock()
	qNorm := norm(queryVec)
	if qNorm == 0 {
		return nil, fmt.Errorf("source chunk has zero-norm vector")
	}
	scores := make([]scored, 0, len(ids))
	for i, id := range ids {
		if id == sourceID {
			continue
		}
		off := i * cDim
		var dot float32
		for j, qv := range queryVec {
			dot += qv * vecs[off+j]
		}
		scores = append(scores, scored{id, dot / (qNorm * norms[i])})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	if len(scores) > k {
		scores = scores[:k]
	}
	return s.fetchHits(ctx, scores, nil, nil, nil, nil)
}

// ensureCache lazily loads (id, vec) for every chunk into a flat
// in-RAM slab plus a parallel slice of precomputed norms. Subsequent
// Search calls work entirely off this slab — no SQL on the hot path.
//
// Cross-process staleness: if another process (e.g. `mcsearch index`)
// wrote new chunks since the cache was built, last_indexed_at in the meta
// table will have advanced. We detect this with a cheap scalar query and
// invalidate before rebuilding, so long-lived MCP server processes always
// serve fresh results without a restart.
func (s *Store) ensureCache(ctx context.Context) error {
	// Read last_indexed_at from meta to detect out-of-process mutations.
	var currentIndexedAt int64
	row := s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='last_indexed_at'`)
	var raw string
	if err := row.Scan(&raw); err == nil {
		fmt.Sscanf(raw, "%d", &currentIndexedAt)
	}

	s.cacheMu.RLock()
	loaded := s.cacheLoaded && s.cacheIndexedAt == currentIndexedAt
	s.cacheMu.RUnlock()
	if loaded {
		return nil
	}
	s.cacheMu.Lock()
	defer s.cacheMu.Unlock()
	if s.cacheLoaded && s.cacheIndexedAt == currentIndexedAt {
		return nil
	}
	// Invalidate stale cache before rebuild.
	s.cacheLoaded = false
	s.cacheIDs = nil
	s.cacheVecs = nil
	s.cacheNorms = nil
	rows, err := s.db.QueryContext(ctx, `SELECT id, vec FROM chunks`)
	if err != nil {
		return err
	}
	defer rows.Close()
	ids := make([]int64, 0, 1024)
	var vecs []float32
	norms := make([]float32, 0, 1024)
	dim := s.dim
	for rows.Next() {
		var id int64
		var blob []byte
		if err := rows.Scan(&id, &blob); err != nil {
			return err
		}
		if len(blob)%4 != 0 {
			return fmt.Errorf("vec blob length %d not divisible by 4", len(blob))
		}
		n := len(blob) / 4
		if dim == 0 {
			dim = n
		} else if n != dim {
			return fmt.Errorf("vec dim mismatch in cache: got %d, want %d", n, dim)
		}
		// Decode in place; skip zero-norm vectors (they can't score).
		row := make([]float32, dim)
		var sq float32
		for i := range dim {
			row[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
			sq += row[i] * row[i]
		}
		if sq == 0 {
			continue
		}
		ids = append(ids, id)
		vecs = append(vecs, row...)
		norms = append(norms, float32(math.Sqrt(float64(sq))))
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if dim != 0 && s.dim == 0 {
		s.dim = dim
	}
	s.cacheIDs = ids
	s.cacheVecs = vecs
	s.cacheNorms = norms
	s.cacheIndexedAt = currentIndexedAt
	s.cacheLoaded = true
	return nil
}

// invalidateCache marks the in-RAM slab stale. Cheap: we just drop the
// references and let the next Search rebuild. Called by every mutator.
func (s *Store) invalidateCache() {
	s.cacheMu.Lock()
	s.cacheLoaded = false
	s.cacheIDs = nil
	s.cacheVecs = nil
	s.cacheNorms = nil
	s.cacheMu.Unlock()
}

func norm(v []float32) float32 {
	var s float32
	for _, x := range v {
		s += x * x
	}
	return float32(math.Sqrt(float64(s)))
}

func encodeVec(v []float32) []byte {
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(x))
	}
	return buf
}
