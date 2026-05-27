// Package store persists per-project chunks + embedding vectors.
//
// One SQLite file per project. Vectors live in a sqlite-vec `vec0`
// virtual table (`chunk_vecs`) and KNN runs natively in SQL — cosine
// distance with a serialized float32 BLOB query. The chunks table
// still holds the raw vec BLOB so chunk_vecs can be rebuilt and so
// vec_distance_cosine() can score BM25-only hits cheaply.
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
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/alehatsman/dex/internal/rerank"
	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3" // register sqlite3 driver
)

func init() {
	// Register the sqlite-vec extension so every new connection opened
	// by mattn/go-sqlite3 has vec0 / vec_distance_cosine available.
	sqlite_vec.Auto()
}

const (
	metaDim              = "dim"
	metaLastIndexedAt    = "last_indexed_at"
	metaProjectRoot      = "project_root"
	metaLastSummarizedAt = "last_summarized_at"
)

// Options influence the runtime behaviour of an opened Store.
// All fields are optional; the zero value matches the default
// (hybrid BM25+semantic search enabled).
type Options struct {
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
	dim  atomic.Int64 // vector dimension; set once on first upsert, read concurrently
	opts Options      // immutable after Open
}

// Open opens or creates the SQLite file at path with default
// Options. Convenience wrapper around OpenWith.
func Open(ctx context.Context, path string) (*Store, error) {
	return OpenWith(ctx, path, Options{})
}

// OpenWith is like Open but lets the caller adjust runtime behaviour
// (e.g. disable the BM25 leg of hybrid search).
//
// `_busy_timeout=5000` lets concurrent writers (e.g. `dex index`
// fired while `dex watch` is also re-indexing) wait up to 5 s for
// the writer lock instead of immediately returning SQLITE_BUSY. Without
// it, racing index runs both crash with a leaked DDL error.
func OpenWith(ctx context.Context, path string, opts Options) (*Store, error) {
	db, err := sql.Open("sqlite3",
		"file:"+path+
			"?_journal_mode=WAL"+
			"&_synchronous=NORMAL"+
			"&_busy_timeout=5000"+
			"&_foreign_keys=1")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	s := &Store{db: db, opts: opts}
	if err := s.migrate(ctx); err != nil {
		db.Close()
		return nil, err
	}
	// Recover the recorded vector dimension, if any.
	row := db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='`+metaDim+`'`)
	var v string
	switch err := row.Scan(&v); {
	case errors.Is(err, sql.ErrNoRows):
		// fresh db; dim discovered on first Upsert
	case err != nil:
		db.Close()
		return nil, err
	default:
		dim, _ := strconv.ParseInt(v, 10, 64)
		s.dim.Store(dim)
	}
	// Materialize the vec0 table now if we know the dim — covers both
	// brand-new opens (no chunks yet, dim known from a prior run) and
	// pre-vec0 indexes that need a one-shot backfill from chunks.vec.
	if err := s.ensureVecTable(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("ensure vec table: %w", err)
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
		// graph_nodes / graph_edges hold the structural index produced by
		// the graph phase of `dex index`. The schema is intentionally string-keyed
		// (id TEXT) so node identities are stable across re-extraction and
		// independent of SQLite's rowid. chunk_id links back to chunks.id
		// for callers that want code text plus structural neighborhood;
		// no FK constraint — chunks can be re-upserted with new rowids and
		// we re-resolve the linkage on the next graph index pass.
		`CREATE TABLE IF NOT EXISTS graph_nodes (
		   id              TEXT PRIMARY KEY,
		   kind            TEXT NOT NULL,
		   name            TEXT NOT NULL,
		   qualified_name  TEXT NOT NULL,
		   package_path    TEXT NOT NULL DEFAULT '',
		   file_path       TEXT NOT NULL DEFAULT '',
		   start_line      INTEGER NOT NULL DEFAULT 0,
		   end_line        INTEGER NOT NULL DEFAULT 0,
		   chunk_id        INTEGER,
		   metadata_json   TEXT NOT NULL DEFAULT '{}',
		   content_hash    TEXT NOT NULL,
		   last_seen_at    INTEGER NOT NULL
		 )`,
		`CREATE INDEX IF NOT EXISTS idx_graph_nodes_kind      ON graph_nodes(kind)`,
		`CREATE INDEX IF NOT EXISTS idx_graph_nodes_name      ON graph_nodes(name)`,
		`CREATE INDEX IF NOT EXISTS idx_graph_nodes_package   ON graph_nodes(package_path)`,
		`CREATE INDEX IF NOT EXISTS idx_graph_nodes_file      ON graph_nodes(file_path)`,
		`CREATE INDEX IF NOT EXISTS idx_graph_nodes_last_seen ON graph_nodes(last_seen_at)`,
		`CREATE TABLE IF NOT EXISTS graph_edges (
		   id              TEXT PRIMARY KEY,
		   kind            TEXT NOT NULL,
		   src_id          TEXT NOT NULL,
		   dst_id          TEXT NOT NULL,
		   file_path       TEXT NOT NULL DEFAULT '',
		   start_line      INTEGER NOT NULL DEFAULT 0,
		   end_line        INTEGER NOT NULL DEFAULT 0,
		   metadata_json   TEXT NOT NULL DEFAULT '{}',
		   content_hash    TEXT NOT NULL,
		   last_seen_at    INTEGER NOT NULL
		 )`,
		`CREATE INDEX IF NOT EXISTS idx_graph_edges_src       ON graph_edges(src_id, kind)`,
		`CREATE INDEX IF NOT EXISTS idx_graph_edges_dst       ON graph_edges(dst_id, kind)`,
		`CREATE INDEX IF NOT EXISTS idx_graph_edges_last_seen ON graph_edges(last_seen_at)`,
		// pending_summaries holds work queued by `dex index --summarize`
		// when running in deferred mode. Each row is one summarization job
		// that the drainer (`dex index summarize` or watch idle ticks) will
		// pick up later. UNIQUE(path,kind,content_sha1) makes Enqueue
		// idempotent — repeating an index run on the same source content
		// doesn't multiply queue entries.
		//
		// content_sha1 is the SHA of the *target* summary chunk (what
		// ultimately lands in chunks.content_sha1 once drained), so a
		// pending row that already has a matching chunks row can be
		// deduped at enqueue time. source_sha1 is set on chunk_summary
		// rows to let the drainer look up the source chunk's content in
		// chunks (path, content_sha1=source_sha1) without re-parsing.
		`CREATE TABLE IF NOT EXISTS pending_summaries (
		   id            INTEGER PRIMARY KEY AUTOINCREMENT,
		   path          TEXT NOT NULL,
		   kind          TEXT NOT NULL,
		   content_sha1  TEXT NOT NULL,
		   start_line    INTEGER NOT NULL DEFAULT 0,
		   end_line      INTEGER NOT NULL DEFAULT 0,
		   chunk_kind    TEXT NOT NULL DEFAULT '',
		   chunk_name    TEXT NOT NULL DEFAULT '',
		   source_sha1   TEXT NOT NULL DEFAULT '',
		   queued_at     INTEGER NOT NULL,
		   attempts      INTEGER NOT NULL DEFAULT 0,
		   last_error    TEXT NOT NULL DEFAULT '',
		   UNIQUE(path, kind, content_sha1)
		 )`,
		`CREATE INDEX IF NOT EXISTS idx_pending_summaries_queued ON pending_summaries(queued_at)`,
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
	// Centrality columns on graph_nodes — populated post-extract from
	// the `calls` edges. All four default to 0 so untouched rows behave
	// as "unknown" (sort_by_centrality just deprioritises them). Added
	// idempotently per the name_col pattern.
	var centralityColsAdded string
	_ = s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='centrality_cols_added'`).Scan(&centralityColsAdded)
	if centralityColsAdded != "1" {
		alters := []string{
			`ALTER TABLE graph_nodes ADD COLUMN in_degree INTEGER NOT NULL DEFAULT 0`,
			`ALTER TABLE graph_nodes ADD COLUMN out_degree INTEGER NOT NULL DEFAULT 0`,
			`ALTER TABLE graph_nodes ADD COLUMN cross_pkg_callers INTEGER NOT NULL DEFAULT 0`,
			`ALTER TABLE graph_nodes ADD COLUMN pagerank REAL NOT NULL DEFAULT 0`,
		}
		for _, q := range alters {
			if _, err := s.db.ExecContext(ctx, q); err != nil &&
				!strings.Contains(err.Error(), "duplicate column name") {
				return fmt.Errorf("migrate: %s: %w", q, err)
			}
		}
		if _, err := s.db.ExecContext(ctx,
			`INSERT INTO meta(key, value) VALUES('centrality_cols_added', '1')
			 ON CONFLICT(key) DO UPDATE SET value=excluded.value`); err != nil {
			return fmt.Errorf("migrate: centrality flag: %w", err)
		}
	}
	return nil
}

// ensureVecTable materializes the sqlite-vec vec0 virtual table and the
// triggers that keep it in sync with `chunks`. The dim is fixed at CREATE
// time, so this is a no-op until s.dim is known (either recovered from
// meta.dim on Open or set on the first UpsertMany).
//
// On first creation against a pre-vec0 index (chunks already populated),
// it backfills chunk_vecs from chunks.vec in one INSERT...SELECT. Cheap
// because vec0 takes the BLOB format we already store on disk (packed
// little-endian float32).
func (s *Store) ensureVecTable(ctx context.Context) error {
	dim := s.dim.Load()
	if dim <= 0 {
		return nil
	}
	stmts := []string{
		// vec0 keeps the embedding in its own storage; cosine distance
		// lets the rest of the search code keep treating "larger score =
		// better" (we return 1 - distance for callers).
		fmt.Sprintf(`CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vecs USING vec0(
		   embedding FLOAT[%d] distance_metric=cosine
		 )`, dim),
		// Mirror chunks.vec into chunk_vecs.embedding. We piggyback on
		// chunks.id (== rowid) as the join key, matching the FTS5 pattern
		// already in use for chunks_fts.
		`CREATE TRIGGER IF NOT EXISTS chunks_vec_ai AFTER INSERT ON chunks BEGIN
		   INSERT INTO chunk_vecs(rowid, embedding) VALUES (new.id, new.vec);
		 END`,
		`CREATE TRIGGER IF NOT EXISTS chunks_vec_ad AFTER DELETE ON chunks BEGIN
		   DELETE FROM chunk_vecs WHERE rowid = old.id;
		 END`,
		`CREATE TRIGGER IF NOT EXISTS chunks_vec_au AFTER UPDATE OF vec ON chunks BEGIN
		   DELETE FROM chunk_vecs WHERE rowid = new.id;
		   INSERT INTO chunk_vecs(rowid, embedding) VALUES (new.id, new.vec);
		 END`,
	}
	for _, q := range stmts {
		if _, err := s.db.ExecContext(ctx, q); err != nil {
			return fmt.Errorf("ensure vec table: %w (%s)", err, q)
		}
	}
	// One-shot backfill for indexes that pre-date sqlite-vec. Triggers
	// only fire on future writes, so any chunks already on disk need to
	// be pushed into chunk_vecs explicitly. Cheap and idempotent: if
	// chunk_vecs is already populated, the SELECT yields zero new rows.
	var vecRows, chunkRows int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunk_vecs`).Scan(&vecRows); err != nil {
		return fmt.Errorf("ensure vec table: count chunk_vecs: %w", err)
	}
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks`).Scan(&chunkRows); err != nil {
		return fmt.Errorf("ensure vec table: count chunks: %w", err)
	}
	if vecRows == 0 && chunkRows > 0 {
		if _, err := s.db.ExecContext(ctx,
			`INSERT INTO chunk_vecs(rowid, embedding) SELECT id, vec FROM chunks`); err != nil {
			return fmt.Errorf("ensure vec table: backfill: %w", err)
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
	// PendingSummaries is the current depth of the pending_summaries
	// queue. Non-zero on a fresh `dex index --summarize-defer` run;
	// drained by `dex index summarize` or by `dex watch`'s idle hook.
	PendingSummaries int
	// LastSummarized is the wall-clock time of the most recent
	// successful summary generation (per-chunk, file, package, or
	// repo). Zero when no summaries have ever been produced.
	LastSummarized time.Time
}

func (s *Store) Stats(ctx context.Context) (Stats, error) {
	var st Stats
	st.Dim = int(s.dim.Load())
	row := s.db.QueryRowContext(ctx, `SELECT COUNT(*), COUNT(DISTINCT path) FROM chunks`)
	if err := row.Scan(&st.Chunks, &st.Files); err != nil {
		return st, err
	}
	row = s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='`+metaLastIndexedAt+`'`)
	var v string
	if err := row.Scan(&v); err == nil {
		ts, _ := strconv.ParseInt(v, 10, 64)
		if ts > 0 {
			st.LastIndex = time.Unix(0, ts)
		}
	}
	// pending_summaries queue depth. Treat the table-missing case as 0
	// so old indexes that pre-date the table don't break Stats.
	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM pending_summaries`)
	_ = row.Scan(&st.PendingSummaries)

	row = s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='`+metaLastSummarizedAt+`'`)
	var sv string
	if err := row.Scan(&sv); err == nil {
		ts, _ := strconv.ParseInt(sv, 10, 64)
		if ts > 0 {
			st.LastSummarized = time.Unix(0, ts)
		}
	}
	return st, nil
}

// SetLastIndexedAt records the wall-clock time of the most recent
// successful (full or incremental) re-index.
func (s *Store) SetLastIndexedAt(ctx context.Context, t time.Time) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO meta(key,value) VALUES('`+metaLastIndexedAt+`', ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		strconv.FormatInt(t.UnixNano(), 10))
	return err
}

// SetLastSummarizedAt records the wall-clock time of the most recent
// successful summary generation. Bumped by the drainer whenever a
// batch produces new summary chunks (chunk / file / package / repo).
// Cache-hit batches that only TouchSeen existing summaries do not bump.
func (s *Store) SetLastSummarizedAt(ctx context.Context, t time.Time) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO meta(key,value) VALUES('`+metaLastSummarizedAt+`', ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		strconv.FormatInt(t.UnixNano(), 10))
	return err
}

// SetProjectRoot records the absolute project path this index belongs
// to. Needed by `reindex --all`, which walks the sha256(path)-keyed
// cache dirs and has to recover each project's original on-disk root.
func (s *Store) SetProjectRoot(ctx context.Context, root string) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO meta(key,value) VALUES('`+metaProjectRoot+`', ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`, root)
	return err
}

// ProjectRoot returns the path previously recorded by SetProjectRoot.
// Returns "" (not an error) if the row is missing — that's the
// pre-migration case for indexes built before this metadata existed.
func (s *Store) ProjectRoot(ctx context.Context) (string, error) {
	var v string
	row := s.db.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='`+metaProjectRoot+`'`)
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

// ExistingSHAsBatch returns existing content_sha1 sets for multiple paths in a
// single round-trip. The outer map is keyed by path; missing paths map to nil.
// Batched in groups of 500 to stay within SQLite's default parameter limit.
func (s *Store) ExistingSHAsBatch(ctx context.Context, paths []string) (map[string]map[string]bool, error) {
	out := make(map[string]map[string]bool, len(paths))
	const batchSize = 500
	for i := 0; i < len(paths); i += batchSize {
		end := i + batchSize
		if end > len(paths) {
			end = len(paths)
		}
		slice := paths[i:end]
		args := make([]any, len(slice))
		for j, p := range slice {
			args[j] = p
		}
		rows, err := s.db.QueryContext(ctx,
			`SELECT path, content_sha1 FROM chunks WHERE path IN (`+inPlaceholders(len(slice))+`)`,
			args...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var path, sha string
			if err := rows.Scan(&path, &sha); err != nil {
				rows.Close()
				return nil, err
			}
			if out[path] == nil {
				out[path] = make(map[string]bool)
			}
			out[path][sha] = true
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, err
		}
		rows.Close()
	}
	return out, nil
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
	if s.dim.Load() == 0 {
		s.dim.Store(int64(len(rows[0].Vec)))
		if _, err := s.db.ExecContext(ctx,
			`INSERT INTO meta(key,value) VALUES('`+metaDim+`', ?)
			 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
			strconv.FormatInt(s.dim.Load(), 10)); err != nil {
			return err
		}
		// First write to a fresh index — now we know the dim, so create
		// the vec0 table + triggers. After this, every INSERT/UPDATE on
		// chunks mirrors into chunk_vecs via the triggers.
		if err := s.ensureVecTable(ctx); err != nil {
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
		if int64(len(r.Vec)) != s.dim.Load() {
			_ = tx.Rollback()
			return fmt.Errorf("vector dim mismatch: index has dim=%d, got %d (did the embedding model change?)", s.dim.Load(), len(r.Vec))
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
	return nil
}

// TouchSeen bumps last_seen_at for an already-present (path, sha) pair and
// backfills the name column — so chunks indexed before the name column was
// added get their names populated on the next walk without re-embedding.
//
// When startLine > 0, also refreshes start_line/end_line. Required for
// the chunker fast-path: a chunk's content can stay byte-identical
// (same SHA) while its position in the file shifts because some earlier
// chunk in the same file grew or shrank. Without this update, search_symbol
// returns the chunk's ORIGINAL line range even after the file was edited
// above it. Callers that don't have line info (file/package/repo summary
// touches) pass 0 to skip the position update.
func (s *Store) TouchSeen(ctx context.Context, path, contentSHA, name string, startLine, endLine int, now time.Time) error {
	if startLine > 0 {
		_, err := s.db.ExecContext(ctx,
			`UPDATE chunks SET last_seen_at=?, name=?, start_line=?, end_line=? WHERE path=? AND content_sha1=?`,
			now.UnixNano(), name, startLine, endLine, path, contentSHA)
		return err
	}
	_, err := s.db.ExecContext(ctx,
		`UPDATE chunks SET last_seen_at=?, name=? WHERE path=? AND content_sha1=?`,
		now.UnixNano(), name, path, contentSHA)
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
//
// Excludes `package_summary` and `repo_summary` kinds. Defer-mode index
// passes (used by `dex watch` and the MCP auto-watcher) skip Pass 5/6
// entirely, which means they never bump last_seen_at on those rows —
// pruning would then destroy good summaries every time the watcher
// fires, and the next `dex guide` run would error with "no summaries".
// The drainer's cascadePackageAndRepo regenerates them when their
// content_sha1 cache key drifts; stale rows are unlikely (package
// content rarely changes file-set composition) and a `dex reindex`
// clears them. Better to keep a slightly-stale 32b-generated overview
// than to drop the only one we have.
func (s *Store) PruneUnseen(ctx context.Context, cutoff time.Time) (int64, error) {
	res, err := s.db.ExecContext(ctx,
		`DELETE FROM chunks
		   WHERE last_seen_at < ?
		     AND kind NOT IN ('package_summary','repo_summary')`,
		cutoff.UnixNano())
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

// DeletePath drops all chunks for a single relative path.
func (s *Store) DeletePath(ctx context.Context, path string) error {
	if _, err := s.db.ExecContext(ctx, `DELETE FROM chunks WHERE path=?`, path); err != nil {
		return err
	}
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
	// ran semantic-only (empty query text or DEX_DISABLE_BM25=1).
	RRFScore float32

	// RerankScore is the cross-encoder relevance score in [0, 1] for
	// the (query, chunk) pair. Zero when rerank didn't run (no client
	// wired, pool ≤ k, or endpoint unreachable). Larger = more relevant.
	RerankScore float32

	// Centrality fields — populated from graph_nodes via the
	// chunk_id join when the symbol has a corresponding graph node.
	// Zero when no graph node exists (the file is in an unindexed
	// language, the chunk isn't a function/method, or the graph hasn't
	// been built yet). Callers use these to sort and to compose the
	// role-hint shown to agents.
	InDegree        int
	OutDegree       int
	CrossPkgCallers int
	PageRank        float64
}

// FormatHits renders a slice of hits as a fenced CONTEXT block for
// injection into a chat completion message. Each chunk gets a header
// with path:line coordinates so the model can cite real locations.
func FormatHits(hits []Hit) string {
	var b strings.Builder
	b.WriteString("CONTEXT — relevant chunks from the project's dex index:\n\n")
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

// FileSummariesForPaths returns the content of file_summary chunks for the
// given relative file paths, ordered by path. Used by the indexer to collect
// per-file prose when generating a package-level summary.
func (s *Store) FileSummariesForPaths(ctx context.Context, paths []string) ([]string, error) {
	if len(paths) == 0 {
		return nil, nil
	}
	args := make([]any, len(paths)+1)
	args[0] = "file_summary"
	for i, p := range paths {
		args[i+1] = p
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT content FROM chunks WHERE kind = ? AND path IN (`+inPlaceholders(len(paths))+`) ORDER BY path`,
		args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []string
	for rows.Next() {
		var content string
		if err := rows.Scan(&content); err != nil {
			return nil, err
		}
		out = append(out, content)
	}
	return out, rows.Err()
}

// FileSummarySHAs returns path → content_sha1 for every file_summary chunk
// in the store. Used by the indexer's mtime fast-path under --summarize:
// fetching all SHAs once up front lets the walker decide fast-path
// eligibility synchronously without N round-trips, and the recovered SHA
// feeds Pass 5's package_summary cache key so dirs whose files all took
// the fast-path don't regenerate.
func (s *Store) FileSummarySHAs(ctx context.Context) (map[string]string, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT path, content_sha1 FROM chunks WHERE kind = 'file_summary'`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	out := make(map[string]string)
	for rows.Next() {
		var path, sha string
		if err := rows.Scan(&path, &sha); err != nil {
			return nil, err
		}
		out[path] = sha
	}
	return out, rows.Err()
}

// AllSummariesByKind returns the content of every chunk with the given kind,
// ordered by path. Used by the indexer to aggregate lower-level summaries
// into a higher-level one (e.g. package summaries → repo summary).
func (s *Store) AllSummariesByKind(ctx context.Context, kind string) ([]string, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT content FROM chunks WHERE kind = ? ORDER BY path`, kind)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []string
	for rows.Next() {
		var content string
		if err := rows.Scan(&content); err != nil {
			return nil, err
		}
		out = append(out, content)
	}
	return out, rows.Err()
}

// SummaryRow carries the columns the guide renderer needs.
// last_seen_at is unix-nanoseconds; callers compare against
// the guide's stored render timestamp to detect dirtiness.
type SummaryRow struct {
	Path       string
	Content    string
	LastSeenAt int64
}

// SummariesByKindWithMeta returns path + content + last_seen_at for every
// chunk of the given kind, ordered by path. Drives the guide renderer.
func (s *Store) SummariesByKindWithMeta(ctx context.Context, kind string) ([]SummaryRow, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT path, content, last_seen_at FROM chunks WHERE kind = ? ORDER BY path`, kind)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var out []SummaryRow
	for rows.Next() {
		var r SummaryRow
		if err := rows.Scan(&r.Path, &r.Content, &r.LastSeenAt); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// SearchSummaries runs a semantic search restricted to summary-kind chunks
// (file_summary, package_summary, repo_summary). Fetches a larger candidate
// pool than k to compensate for summaries being sparse relative to code
// chunks, then filters and trims to k.
func (s *Store) SearchSummaries(ctx context.Context, queryVec []float32, queryText string, k int) ([]Hit, error) {
	candidates := k * 20
	if candidates < 50 {
		candidates = 50
	}
	hits, err := s.searchRaw(ctx, queryVec, queryText, candidates)
	if err != nil || len(hits) == 0 {
		return hits, err
	}
	out := hits[:0]
	for _, h := range hits {
		if h.Kind == "file_summary" || h.Kind == "package_summary" || h.Kind == "repo_summary" {
			out = append(out, h)
			if len(out) == k {
				break
			}
		}
	}
	return out, nil
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
	useBM25 := !s.opts.DisableBM25 && strings.TrimSpace(queryText) != ""

	if !useBM25 {
		// Semantic-only path. vec0 already returns rows sorted by
		// similarity desc, so no client-side sort needed.
		semScores, err := s.scoreSemantic(ctx, queryVec, k)
		if err != nil {
			return nil, err
		}
		if len(semScores) == 0 {
			return nil, nil
		}
		return s.fetchHits(ctx, semScores, scoreContext{})
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

	// Semantic top-pool — sqlite-vec KNN, already sorted desc by similarity.
	semSorted, err := s.scoreSemantic(ctx, queryVec, pool)
	if err != nil {
		return nil, err
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

	// Fill cosine for BM25-only fused IDs so Hit.Score stays populated
	// for every result, not just semantic-leg ones. The set is bounded
	// by `pool` and usually small in practice (high lexical/semantic
	// overlap), so the extra round-trip is cheap.
	var missing []int64
	for id := range bm25Rank {
		if _, ok := semCosine[id]; !ok {
			missing = append(missing, id)
		}
	}
	if filled, err := s.scoreSemanticForIDs(ctx, queryVec, missing); err == nil {
		for id, sim := range filled {
			semCosine[id] = sim
		}
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
			return s.fetchHits(ctx, reranked, scoreContext{semCosine: semCosine, bm25Score: bm25Score, rrfScore: rrf, rerankScore: rerankScore})
		case errors.Is(err, rerank.ErrUnreachable):
			// fall through to non-reranked path
		default:
			return nil, err
		}
	}

	if len(fused) > k {
		fused = fused[:k]
	}
	return s.fetchHits(ctx, fused, scoreContext{semCosine: semCosine, bm25Score: bm25Score})
}

// inPlaceholders returns a comma-separated list of n SQL "?" bind vars,
// e.g. inPlaceholders(3) == "?,?,?".
func inPlaceholders(n int) string {
	s := strings.Repeat("?,", n)
	return s[:len(s)-1]
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
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, content FROM chunks WHERE id IN (`+inPlaceholders(len(idArgs))+`)`,
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

// scoreSemantic returns up to `limit` chunks ranked by cosine similarity
// to queryVec, best first. Runs as a single KNN query against the
// sqlite-vec `chunk_vecs` virtual table; vec0 returns rows sorted by
// distance ascending, which is similarity descending — no client-side
// sort needed.
func (s *Store) scoreSemantic(ctx context.Context, queryVec []float32, limit int) ([]scored, error) {
	if d := s.dim.Load(); d != 0 && int64(len(queryVec)) != d {
		return nil, fmt.Errorf("query dim %d != index dim %d", len(queryVec), d)
	}
	// Reject all-zero queries up front. vec0's cosine path would otherwise
	// produce NaN distances on a zero vector and surface nonsense rankings.
	// Done before the empty-index early-return so callers get a clear error
	// even when there's nothing to search yet.
	allZero := true
	for _, x := range queryVec {
		if x != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return nil, fmt.Errorf("query vector is zero")
	}
	if limit <= 0 || s.dim.Load() == 0 {
		return nil, nil
	}
	qBlob := encodeVec(queryVec)
	rows, err := s.db.QueryContext(ctx,
		`SELECT rowid, distance FROM chunk_vecs
		 WHERE embedding MATCH ? AND k = ?
		 ORDER BY distance`,
		qBlob, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]scored, 0, limit)
	for rows.Next() {
		var id int64
		var dist float64
		if err := rows.Scan(&id, &dist); err != nil {
			return nil, err
		}
		// Cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1] so
		// callers can keep the "larger = better" convention shared with
		// the BM25 leg (which flips bm25() sign for the same reason).
		out = append(out, scored{id, float32(1 - dist)})
	}
	return out, rows.Err()
}

// scoreSemanticForIDs fills in cosine similarity for a specific set of
// chunk IDs that the vec0 top-K query missed (BM25-only fused hits).
// Uses sqlite-vec's scalar vec_distance_cosine() so we can keep Hit.Score
// populated even for hits that surfaced purely through the lexical leg.
// Returns a partial map; callers must tolerate missing entries.
func (s *Store) scoreSemanticForIDs(ctx context.Context, queryVec []float32, ids []int64) (map[int64]float32, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	args := make([]any, 0, len(ids)+1)
	args = append(args, encodeVec(queryVec))
	for _, id := range ids {
		args = append(args, id)
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, vec_distance_cosine(?, vec) FROM chunks WHERE id IN (`+inPlaceholders(len(ids))+`)`,
		args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make(map[int64]float32, len(ids))
	for rows.Next() {
		var id int64
		var dist float64
		if err := rows.Scan(&id, &dist); err != nil {
			return nil, err
		}
		out[id] = float32(1 - dist)
	}
	return out, rows.Err()
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
	fields := strings.Fields(q)
	toks := make([]string, 0, len(fields))
	for _, f := range fields {
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

// scoreContext carries the per-id score maps produced by the hybrid /
// reranked search pipeline. All fields are optional (nil = not available).
type scoreContext struct {
	semCosine   map[int64]float32 // raw cosine scores from the semantic leg
	bm25Score   map[int64]float32 // BM25 scores from the FTS leg
	rrfScore    map[int64]float32 // RRF fusion scores (non-nil only on reranked path)
	rerankScore map[int64]float32 // cross-encoder scores (non-nil only on reranked path)
}

// fetchHits issues one SELECT to get content for the ranked IDs, then
// assembles Hit values with scores from sc.
//   - sc.semCosine / sc.bm25Score: nil in semantic-only mode.
//   - sc.rrfScore: non-nil on the reranked path; ranked[i].score is the
//     rerank score in that case, so RRFScore must come from the map.
//   - sc.rerankScore: non-nil on the reranked path; populates Hit.RerankScore.
func (s *Store) fetchHits(ctx context.Context, ranked []scored, sc scoreContext) ([]Hit, error) {
	if len(ranked) == 0 {
		return nil, nil
	}
	idArgs := make([]any, len(ranked))
	for i, r := range ranked {
		idArgs[i] = r.id
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, path, kind, name, start_line, end_line, content FROM chunks WHERE id IN (`+inPlaceholders(len(idArgs))+`)`,
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
	for _, r := range ranked {
		h, ok := byID[r.id]
		if !ok {
			continue
		}
		if sc.semCosine != nil {
			h.Score = sc.semCosine[r.id]
			if sc.rrfScore != nil {
				h.RRFScore = sc.rrfScore[r.id]
			} else {
				h.RRFScore = r.score
			}
		} else {
			h.Score = r.score
		}
		if sc.bm25Score != nil {
			h.BM25Score = sc.bm25Score[r.id]
		}
		if sc.rerankScore != nil {
			h.RerankScore = sc.rerankScore[r.id]
		}
		out = append(out, h)
	}
	return dedupChunkSummaries(out), nil
}

// FindSymbol returns chunks whose `name` column exactly matches the
// given identifier. Results are ordered by (path, start_line). Uses a
// SQL index scan — no embedding required — so it is fast regardless of
// index size.
//
// When the chunks table yields zero hits, falls back to a graph_nodes
// scan. The Go-graph layer indexes types and struct fields that don't
// produce standalone chunks (the chunker emits chunks per function/
// method/class, not per field), so a query like `MaxFileSize` finds
// the field via the graph even though chunks has nothing. Graph-fallback
// hits carry path + line range but empty Content, since graph nodes
// only point at offsets — agents can Read the range for the body.
func (s *Store) FindSymbol(ctx context.Context, name string, k int) ([]Hit, error) {
	if k <= 0 {
		k = 10
	}
	// LEFT JOIN graph_nodes on chunk_id surfaces centrality columns for
	// the (typically single) graph node bound to each chunk. When the
	// graph hasn't been built — or the chunk isn't a function/method —
	// the COALESCEd zeros sink the row to the natural path-order tail,
	// preserving the pre-centrality default.
	//
	// Sort key: pagerank DESC, in_degree DESC, then path/line for
	// determinism on ties. Centrality is per-symbol, so two callers
	// asking "search_symbol Indexer" land on the SAME top result every
	// run, instead of whichever chunk happens to come first in path
	// order.
	rows, err := s.db.QueryContext(ctx,
		`SELECT c.id, c.path, c.kind, c.name, c.start_line, c.end_line, c.content,
		        COALESCE(g.in_degree, 0), COALESCE(g.out_degree, 0),
		        COALESCE(g.cross_pkg_callers, 0), COALESCE(g.pagerank, 0)
		 FROM chunks c
		 LEFT JOIN graph_nodes g ON g.chunk_id = c.id
		 WHERE c.name = ?
		 ORDER BY COALESCE(g.pagerank, 0) DESC,
		          COALESCE(g.in_degree, 0) DESC,
		          c.path, c.start_line
		 LIMIT ?`,
		name, k)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Hit
	for rows.Next() {
		var id int64
		var h Hit
		if err := rows.Scan(&id, &h.Path, &h.Kind, &h.Name, &h.StartLine, &h.EndLine, &h.Content,
			&h.InDegree, &h.OutDegree, &h.CrossPkgCallers, &h.PageRank); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(out) > 0 {
		return out, nil
	}
	return s.findSymbolInGraph(ctx, name, k)
}

// findSymbolInGraph queries the Go-graph layer for nodes whose `name`
// column matches exactly. Used as a fallback by FindSymbol when the
// chunks table has nothing — covers types, struct fields, and other
// entities that don't produce standalone chunks. Returns nil on
// missing graph table (older index versions) rather than failing the
// surrounding lookup.
func (s *Store) findSymbolInGraph(ctx context.Context, name string, k int) ([]Hit, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT kind, name, file_path, start_line, end_line,
		        in_degree, out_degree, cross_pkg_callers, pagerank
		 FROM graph_nodes
		 WHERE name = ? AND file_path != '' AND start_line > 0
		 ORDER BY pagerank DESC, in_degree DESC, file_path, start_line LIMIT ?`,
		name, k)
	if err != nil {
		// graph_nodes may not exist on older indexes — degrade silently.
		return nil, nil //nolint:nilerr
	}
	defer rows.Close()
	var out []Hit
	for rows.Next() {
		var h Hit
		if err := rows.Scan(&h.Kind, &h.Name, &h.Path, &h.StartLine, &h.EndLine,
			&h.InDegree, &h.OutDegree, &h.CrossPkgCallers, &h.PageRank); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// FindSymbolCandidates returns up to k distinct chunk names whose
// `name` column contains `query` as a substring. Ordered by length
// (shorter ≈ closer-in-spirit) then alphabetically. Intended as a
// "did you mean" surface for search_symbol misses — callers should pass
// the exact-name lookup query and surface the results in a hint so
// the agent can retry with a real identifier instead of guessing.
func (s *Store) FindSymbolCandidates(ctx context.Context, query string, k int) ([]string, error) {
	if k <= 0 {
		k = 5
	}
	if query == "" {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT DISTINCT name FROM chunks
		 WHERE name LIKE '%' || ? || '%' AND name != '' AND name != ?
		 ORDER BY length(name), name LIMIT ?`,
		query, query, k)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return nil, err
		}
		out = append(out, name)
	}
	return out, rows.Err()
}

// RelatedChunks returns the top-k chunks most similar to the chunk at
// (path, startLine), excluding the source chunk itself. Issues one vec0
// KNN query with k+1 candidates so we can drop the source (which always
// ranks first at distance 0). Returns an error if no chunk is found at
// the given location.
func (s *Store) RelatedChunks(ctx context.Context, path string, startLine int, k int) ([]Hit, error) {
	if k <= 0 {
		k = 8
	}
	var blob []byte
	var sourceID int64
	err := s.db.QueryRowContext(ctx,
		`SELECT id, vec FROM chunks WHERE path = ? AND start_line = ?`,
		path, startLine).Scan(&sourceID, &blob)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("no chunk at %s:%d", path, startLine)
		}
		return nil, err
	}
	if len(blob)%4 != 0 {
		return nil, fmt.Errorf("vec blob length %d not divisible by 4", len(blob))
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT rowid, distance FROM chunk_vecs
		 WHERE embedding MATCH ? AND k = ?
		 ORDER BY distance`,
		blob, k+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	scores := make([]scored, 0, k)
	for rows.Next() {
		var id int64
		var dist float64
		if err := rows.Scan(&id, &dist); err != nil {
			return nil, err
		}
		if id == sourceID {
			continue
		}
		scores = append(scores, scored{id, float32(1 - dist)})
		if len(scores) >= k {
			break
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return s.fetchHits(ctx, scores, scoreContext{})
}

func encodeVec(v []float32) []byte {
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(x))
	}
	return buf
}
