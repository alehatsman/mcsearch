package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

// PendingSummary is one queued summarization job. Kind tells the drainer
// which summarize* helper to invoke; the other fields hold the inputs.
// ContentSHA is the SHA of the *target* summary chunk (i.e. what its
// row in chunks.content_sha1 will be when the drainer completes), which
// makes UNIQUE(path,kind,content_sha1) a natural idempotency key.
//
// SourceSHA is only set on chunk_summary jobs: it points at the source
// chunk in chunks (path, content_sha1=source_sha1) so the drainer can
// recover the chunk's content without re-parsing the file. For
// file_summary jobs the drainer reads the file from disk; for
// package_summary the drainer queries existing file_summary chunks.
type PendingSummary struct {
	ID         int64
	Path       string
	Kind       string
	ContentSHA string
	StartLine  int
	EndLine    int
	ChunkKind  string
	ChunkName  string
	SourceSHA  string
	QueuedAt   time.Time
	Attempts   int
	LastError  string
}

// EnqueuePendingSummary inserts a queue row. Idempotent on
// (path, kind, content_sha1): a re-enqueue of the same logical job is a
// no-op, leaving the existing row's attempts / last_error untouched.
// `now` provides the queued_at timestamp (Unix nanos) — callers usually
// pass the run's startTime so a single index pass's queue rows share
// timestamps.
func (s *Store) EnqueuePendingSummary(ctx context.Context, p PendingSummary, now time.Time) error {
	if p.Path == "" || p.Kind == "" || p.ContentSHA == "" {
		return errors.New("EnqueuePendingSummary: path, kind, content_sha1 are required")
	}
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO pending_summaries (
		   path, kind, content_sha1, start_line, end_line,
		   chunk_kind, chunk_name, source_sha1, queued_at
		 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(path, kind, content_sha1) DO NOTHING`,
		p.Path, p.Kind, p.ContentSHA, p.StartLine, p.EndLine,
		p.ChunkKind, p.ChunkName, p.SourceSHA, now.UnixNano())
	if err != nil {
		return fmt.Errorf("EnqueuePendingSummary: %w", err)
	}
	return nil
}

// ListPendingSummaries returns pending rows ordered by queued_at asc,
// then id asc for stable iteration within the same nanosecond. limit
// caps the result; pass 0 for all rows.
func (s *Store) ListPendingSummaries(ctx context.Context, limit int) ([]PendingSummary, error) {
	q := `SELECT id, path, kind, content_sha1, start_line, end_line,
	             chunk_kind, chunk_name, source_sha1, queued_at, attempts, last_error
	      FROM pending_summaries ORDER BY queued_at ASC, id ASC`
	args := []any{}
	if limit > 0 {
		q += ` LIMIT ?`
		args = append(args, limit)
	}
	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("ListPendingSummaries: %w", err)
	}
	defer func() { _ = rows.Close() }()
	var out []PendingSummary
	for rows.Next() {
		var p PendingSummary
		var queuedNanos int64
		if err := rows.Scan(&p.ID, &p.Path, &p.Kind, &p.ContentSHA,
			&p.StartLine, &p.EndLine, &p.ChunkKind, &p.ChunkName,
			&p.SourceSHA, &queuedNanos, &p.Attempts, &p.LastError); err != nil {
			return nil, fmt.Errorf("ListPendingSummaries scan: %w", err)
		}
		p.QueuedAt = time.Unix(0, queuedNanos)
		out = append(out, p)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("ListPendingSummaries: %w", err)
	}
	return out, nil
}

// DeletePendingSummary removes one row by ID. Called by the drainer
// once a job has been successfully processed (summary embedded and
// upserted into chunks). Returns sql.ErrNoRows-style silently — a
// double-delete from concurrent drainers is not an error.
func (s *Store) DeletePendingSummary(ctx context.Context, id int64) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM pending_summaries WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("DeletePendingSummary: %w", err)
	}
	return nil
}

// BumpPendingAttempts increments attempts and records the most recent
// error message. Called by the drainer when a job fails (chat error,
// stale source content, etc.) so the row can be inspected later or
// retried with backoff. The row stays in the queue.
func (s *Store) BumpPendingAttempts(ctx context.Context, id int64, errMsg string) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE pending_summaries SET attempts = attempts + 1, last_error = ? WHERE id = ?`,
		errMsg, id)
	if err != nil {
		return fmt.Errorf("BumpPendingAttempts: %w", err)
	}
	return nil
}

// CountPendingSummaries returns the current queue depth. Cheap COUNT(*)
// — used by status reporting and by the drainer to know when to stop.
func (s *Store) CountPendingSummaries(ctx context.Context) (int, error) {
	var n int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM pending_summaries`).Scan(&n)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return 0, fmt.Errorf("CountPendingSummaries: %w", err)
	}
	return n, nil
}
