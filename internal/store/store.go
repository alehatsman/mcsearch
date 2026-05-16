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
	"time"

	_ "modernc.org/sqlite"
)

type Store struct {
	db  *sql.DB
	dim int // vector dimension; discovered on first upsert
}

// Open or create the SQLite file at path and run migrations.
//
// `busy_timeout(5000)` lets concurrent writers (e.g. `mcsearch index`
// fired while `mcsearch watch` is also re-indexing) wait up to 5 s for
// the writer lock instead of immediately returning SQLITE_BUSY. Without
// it, racing index runs both crash with a leaked DDL error.
func Open(ctx context.Context, path string) (*Store, error) {
	db, err := sql.Open("sqlite",
		"file:"+path+
			"?_pragma=journal_mode(WAL)"+
			"&_pragma=synchronous(NORMAL)"+
			"&_pragma=busy_timeout(5000)"+
			"&_pragma=foreign_keys(1)")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	s := &Store{db: db}
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
	}
	for _, q := range stmts {
		if _, err := s.db.ExecContext(ctx, q); err != nil {
			return fmt.Errorf("migrate: %w (%s)", err, q)
		}
	}
	return nil
}

// Stats reports the current state of an index.
type Stats struct {
	Chunks     int
	Files      int
	Dim        int
	LastIndex  time.Time
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

// Upsert inserts a chunk (or refreshes its last_seen_at if (path, sha) is
// already present). Vector dimension must be consistent across the index;
// the first call seeds it.
func (s *Store) Upsert(ctx context.Context, path, kind string, startLine, endLine int, contentSHA, content string, vec []float32, now time.Time) error {
	if s.dim == 0 {
		s.dim = len(vec)
		if _, err := s.db.ExecContext(ctx,
			`INSERT INTO meta(key,value) VALUES('dim', ?)
			 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
			fmt.Sprintf("%d", s.dim)); err != nil {
			return err
		}
	} else if s.dim != len(vec) {
		return fmt.Errorf("vector dim mismatch: index has dim=%d, got %d (did the embedding model change?)", s.dim, len(vec))
	}
	blob := encodeVec(vec)
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO chunks(path, kind, start_line, end_line, content_sha1, content, vec, last_seen_at)
		 VALUES(?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(path, content_sha1) DO UPDATE SET
		   kind=excluded.kind,
		   start_line=excluded.start_line,
		   end_line=excluded.end_line,
		   content=excluded.content,
		   vec=excluded.vec,
		   last_seen_at=excluded.last_seen_at`,
		path, kind, startLine, endLine, contentSHA, content, blob, now.UnixNano())
	return err
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
	return res.RowsAffected()
}

// DeletePath drops all chunks for a single relative path.
func (s *Store) DeletePath(ctx context.Context, path string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM chunks WHERE path=?`, path)
	return err
}

// Hit is one search result.
type Hit struct {
	Path      string
	Kind      string
	StartLine int
	EndLine   int
	Content   string
	Score     float32 // cosine similarity, 1.0 == identical direction
}

// Search returns the top-k chunks by cosine similarity to query.
//
// Two-pass: first scan loads only (id, vec) to compute scores, picks
// the top-k IDs, and the second scan fetches path/kind/line/content
// for those IDs only. The naive single-pass version loads every
// chunk's content text (often KB per row) just to discard most of it —
// at 100 k chunks × 1 KB that's ~100 MB of throwaway I/O per query.
func (s *Store) Search(ctx context.Context, query []float32, k int) ([]Hit, error) {
	if k <= 0 {
		k = 8
	}
	if s.dim != 0 && len(query) != s.dim {
		return nil, fmt.Errorf("query dim %d != index dim %d", len(query), s.dim)
	}
	qNorm := norm(query)
	if qNorm == 0 {
		return nil, fmt.Errorf("query vector is zero")
	}

	type scored struct {
		id    int64
		score float32
	}
	rows, err := s.db.QueryContext(ctx, `SELECT id, vec FROM chunks`)
	if err != nil {
		return nil, err
	}
	scores := make([]scored, 0, 1024)
	// Reuse one decode buffer across rows so we don't allocate a new
	// []float32 per chunk in the hot loop.
	var vbuf []float32
	for rows.Next() {
		var id int64
		var blob []byte
		if err := rows.Scan(&id, &blob); err != nil {
			rows.Close()
			return nil, err
		}
		if len(blob)%4 != 0 {
			rows.Close()
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
		for i, q := range query {
			vi := vbuf[i]
			dot += q * vi
			vNormSq += vi * vi
		}
		if vNormSq == 0 {
			continue
		}
		score := dot / (qNorm * float32(math.Sqrt(float64(vNormSq))))
		scores = append(scores, scored{id, score})
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(scores) == 0 {
		return nil, nil
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	if len(scores) > k {
		scores = scores[:k]
	}

	// Second pass: fetch content for the top-k IDs only.
	ids := make([]any, len(scores))
	for i, s := range scores {
		ids[i] = s.id
	}
	placeholders := strings.Repeat("?,", len(ids))
	placeholders = placeholders[:len(placeholders)-1]
	q := `SELECT id, path, kind, start_line, end_line, content FROM chunks WHERE id IN (` + placeholders + `)`
	rows2, err := s.db.QueryContext(ctx, q, ids...)
	if err != nil {
		return nil, err
	}
	defer rows2.Close()
	byID := make(map[int64]Hit, len(scores))
	for rows2.Next() {
		var id int64
		var h Hit
		if err := rows2.Scan(&id, &h.Path, &h.Kind, &h.StartLine, &h.EndLine, &h.Content); err != nil {
			return nil, err
		}
		byID[id] = h
	}
	if err := rows2.Err(); err != nil {
		return nil, err
	}
	out := make([]Hit, 0, len(scores))
	for _, s := range scores {
		h, ok := byID[s.id]
		if !ok {
			continue
		}
		h.Score = s.score
		out = append(out, h)
	}
	return out, nil
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

func decodeVec(blob []byte) ([]float32, error) {
	if len(blob)%4 != 0 {
		return nil, fmt.Errorf("vec blob length %d not divisible by 4", len(blob))
	}
	out := make([]float32, len(blob)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
	}
	return out, nil
}
