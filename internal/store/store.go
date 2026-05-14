// Package store persists per-project chunks + embedding vectors.
//
// One SQLite file per project. Vectors are stored as packed little-endian
// float32 BLOBs. Search is brute-force cosine similarity in Go — fast
// enough for <100 k chunks per project. Swap the search path for an HNSW
// index later without changing the rest of the codebase.
package store

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	_ "modernc.org/sqlite"
)

type Store struct {
	db  *sql.DB
	dim int // vector dimension; discovered on first upsert
}

// Open or create the SQLite file at path and run migrations.
func Open(ctx context.Context, path string) (*Store, error) {
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=foreign_keys(1)")
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
			st.LastIndex = time.UnixMilli(ts)
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
		fmt.Sprintf("%d", t.UnixMilli()))
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
		path, kind, startLine, endLine, contentSHA, content, blob, now.UnixMilli())
	return err
}

// TouchSeen bumps last_seen_at for an already-present (path, sha) pair.
// Used when we re-walk a project and encounter unchanged content.
func (s *Store) TouchSeen(ctx context.Context, path, contentSHA string, now time.Time) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE chunks SET last_seen_at=? WHERE path=? AND content_sha1=?`,
		now.UnixMilli(), path, contentSHA)
	return err
}

// PruneUnseen deletes chunks last seen before `cutoff`. Call at the end
// of a re-index to remove stale rows for files that disappeared.
func (s *Store) PruneUnseen(ctx context.Context, cutoff time.Time) (int64, error) {
	res, err := s.db.ExecContext(ctx,
		`DELETE FROM chunks WHERE last_seen_at < ?`, cutoff.UnixMilli())
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

// Search returns the top-k chunks by cosine similarity to query. Brute
// force across the whole table; cheap at <100 k rows.
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
	rows, err := s.db.QueryContext(ctx,
		`SELECT path, kind, start_line, end_line, content, vec FROM chunks`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	hits := make([]Hit, 0, 1024)
	for rows.Next() {
		var h Hit
		var blob []byte
		if err := rows.Scan(&h.Path, &h.Kind, &h.StartLine, &h.EndLine, &h.Content, &blob); err != nil {
			return nil, err
		}
		v, err := decodeVec(blob)
		if err != nil {
			return nil, err
		}
		dot := float32(0)
		for i, q := range query {
			dot += q * v[i]
		}
		vNorm := norm(v)
		if vNorm == 0 {
			continue
		}
		h.Score = dot / (qNorm * vNorm)
		hits = append(hits, h)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits, nil
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
