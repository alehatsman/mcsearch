package store

import (
	"context"
	"testing"
	"time"
)

// helper to enqueue a minimal valid PendingSummary
func enqueueFileSummary(t *testing.T, ctx context.Context, st *Store, path, sha string, now time.Time) {
	t.Helper()
	if err := st.EnqueuePendingSummary(ctx, PendingSummary{
		Path:       path,
		Kind:       "file_summary",
		ContentSHA: sha,
		StartLine:  1,
		EndLine:    10,
	}, now); err != nil {
		t.Fatal(err)
	}
}

func TestEnqueuePendingSummaryIdempotent(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()

	enqueueFileSummary(t, ctx, st, "a.go", "sha1", now)
	enqueueFileSummary(t, ctx, st, "a.go", "sha1", now) // duplicate
	enqueueFileSummary(t, ctx, st, "a.go", "sha1", now.Add(time.Second))

	got, err := st.CountPendingSummaries(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Errorf("ON CONFLICT DO NOTHING should keep count at 1; got %d", got)
	}
}

func TestEnqueuePendingSummaryRequiresFields(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	cases := []PendingSummary{
		{Kind: "file_summary", ContentSHA: "sha"},     // missing path
		{Path: "a.go", ContentSHA: "sha"},             // missing kind
		{Path: "a.go", Kind: "file_summary"},          // missing content_sha1
	}
	for i, c := range cases {
		if err := st.EnqueuePendingSummary(ctx, c, now); err == nil {
			t.Errorf("case %d: expected error for missing field; got nil", i)
		}
	}
}

func TestListPendingSummariesOrderingAndLimit(t *testing.T) {
	st, ctx := newStore(t)
	base := time.Unix(0, 1_000_000_000) // arbitrary nanos
	enqueueFileSummary(t, ctx, st, "c.go", "shaC", base.Add(2*time.Second))
	enqueueFileSummary(t, ctx, st, "a.go", "shaA", base)
	enqueueFileSummary(t, ctx, st, "b.go", "shaB", base.Add(time.Second))

	all, err := st.ListPendingSummaries(ctx, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 3 {
		t.Fatalf("expected 3 rows; got %d", len(all))
	}
	wantOrder := []string{"a.go", "b.go", "c.go"} // by queued_at asc
	for i, p := range all {
		if p.Path != wantOrder[i] {
			t.Errorf("position %d: want path=%q got %q (queued_at=%v)",
				i, wantOrder[i], p.Path, p.QueuedAt)
		}
	}

	// limit caps result
	top1, err := st.ListPendingSummaries(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(top1) != 1 || top1[0].Path != "a.go" {
		t.Errorf("limit=1 should return only a.go; got %+v", top1)
	}
}

func TestListPendingSummariesRoundTripsAllFields(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now().Truncate(time.Nanosecond)
	want := PendingSummary{
		Path:       "pkg/foo.go",
		Kind:       "chunk_summary",
		ContentSHA: "sumsha",
		StartLine:  10,
		EndLine:    42,
		ChunkKind:  "function_declaration",
		ChunkName:  "DoThing",
		SourceSHA:  "srcsha",
	}
	if err := st.EnqueuePendingSummary(ctx, want, now); err != nil {
		t.Fatal(err)
	}
	got, err := st.ListPendingSummaries(ctx, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 row; got %d", len(got))
	}
	p := got[0]
	if p.ID == 0 {
		t.Errorf("ID should be auto-assigned; got 0")
	}
	if p.Path != want.Path || p.Kind != want.Kind || p.ContentSHA != want.ContentSHA ||
		p.StartLine != want.StartLine || p.EndLine != want.EndLine ||
		p.ChunkKind != want.ChunkKind || p.ChunkName != want.ChunkName ||
		p.SourceSHA != want.SourceSHA {
		t.Errorf("round-trip mismatch:\n  want %+v\n  got  %+v", want, p)
	}
	if !p.QueuedAt.Equal(now) {
		t.Errorf("QueuedAt: want %v, got %v", now, p.QueuedAt)
	}
	if p.Attempts != 0 || p.LastError != "" {
		t.Errorf("fresh row should have Attempts=0 LastError=''; got %d / %q", p.Attempts, p.LastError)
	}
}

func TestDeletePendingSummary(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	enqueueFileSummary(t, ctx, st, "a.go", "sha1", now)
	enqueueFileSummary(t, ctx, st, "b.go", "sha2", now)

	rows, _ := st.ListPendingSummaries(ctx, 0)
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows; got %d", len(rows))
	}
	if err := st.DeletePendingSummary(ctx, rows[0].ID); err != nil {
		t.Fatal(err)
	}
	remaining, _ := st.ListPendingSummaries(ctx, 0)
	if len(remaining) != 1 || remaining[0].Path != "b.go" {
		t.Errorf("expected only b.go left; got %+v", remaining)
	}

	// Deleting a non-existent ID is a no-op, not an error.
	if err := st.DeletePendingSummary(ctx, 99999); err != nil {
		t.Errorf("delete of unknown id should be no-op; got error %v", err)
	}
}

func TestBumpPendingAttempts(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	enqueueFileSummary(t, ctx, st, "a.go", "sha1", now)
	rows, _ := st.ListPendingSummaries(ctx, 0)
	id := rows[0].ID

	if err := st.BumpPendingAttempts(ctx, id, "chat timeout"); err != nil {
		t.Fatal(err)
	}
	if err := st.BumpPendingAttempts(ctx, id, "chat 500"); err != nil {
		t.Fatal(err)
	}
	after, _ := st.ListPendingSummaries(ctx, 0)
	if len(after) != 1 {
		t.Fatalf("expected 1 row; got %d", len(after))
	}
	if after[0].Attempts != 2 {
		t.Errorf("Attempts should be 2 after two bumps; got %d", after[0].Attempts)
	}
	if after[0].LastError != "chat 500" {
		t.Errorf("LastError should reflect most recent message; got %q", after[0].LastError)
	}
}

func TestCountPendingSummariesEmpty(t *testing.T) {
	st, ctx := newStore(t)
	n, err := st.CountPendingSummaries(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("fresh DB should have 0 pending; got %d", n)
	}
}

func TestOldestPendingSummaryAge(t *testing.T) {
	st, ctx := newStore(t)

	// Empty queue → zero.
	if age, err := st.OldestPendingSummaryAge(ctx); err != nil || age != 0 {
		t.Fatalf("empty queue: age=%v err=%v, want 0 and nil", age, err)
	}

	// Enqueue with a known queued_at in the past.
	past := time.Now().Add(-5 * time.Minute)
	enqueueFileSummary(t, ctx, st, "old.go", "sha-old", past)
	// And one fresh row.
	enqueueFileSummary(t, ctx, st, "new.go", "sha-new", time.Now())

	age, err := st.OldestPendingSummaryAge(ctx)
	if err != nil {
		t.Fatal(err)
	}
	// The oldest queued_at is `past`; age should be roughly 5 min.
	// Wide tolerance so the test isn't flaky under load.
	if age < 4*time.Minute || age > 6*time.Minute {
		t.Errorf("age = %v, want roughly 5m", age)
	}

	// Stats() should expose the same field.
	stats, _ := st.Stats(ctx)
	if stats.PendingSummariesOldestAge < 4*time.Minute || stats.PendingSummariesOldestAge > 6*time.Minute {
		t.Errorf("Stats.PendingSummariesOldestAge = %v, want roughly 5m", stats.PendingSummariesOldestAge)
	}
}

// TestStatsExposesPendingAndLastSummarized verifies the new Stats
// fields surface what `dex index status` and the MCP status tool need
// to report background-summary progress to the user.
func TestStatsExposesPendingAndLastSummarized(t *testing.T) {
	st, ctx := newStore(t)

	// Fresh DB: both fields zero.
	stats, err := st.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.PendingSummaries != 0 {
		t.Errorf("fresh DB: PendingSummaries = %d, want 0", stats.PendingSummaries)
	}
	if !stats.LastSummarized.IsZero() {
		t.Errorf("fresh DB: LastSummarized = %v, want zero", stats.LastSummarized)
	}

	// Enqueue rows and bump the timestamp; Stats should report both.
	now := time.Now().Truncate(time.Microsecond) // SQLite stores ns, but ns→s round-trip preserves to μs in tests
	enqueueFileSummary(t, ctx, st, "a.go", "sha1", now)
	enqueueFileSummary(t, ctx, st, "b.go", "sha2", now)
	enqueueFileSummary(t, ctx, st, "c.go", "sha3", now)

	if err := st.SetLastSummarizedAt(ctx, now); err != nil {
		t.Fatalf("SetLastSummarizedAt: %v", err)
	}

	stats, err = st.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.PendingSummaries != 3 {
		t.Errorf("PendingSummaries = %d, want 3", stats.PendingSummaries)
	}
	if !stats.LastSummarized.Equal(now) {
		t.Errorf("LastSummarized = %v, want %v", stats.LastSummarized, now)
	}
}
