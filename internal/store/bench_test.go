package store

import (
	"context"
	"math/rand"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func benchStore(b *testing.B, n, dim int) *Store {
	b.Helper()
	db := filepath.Join(b.TempDir(), "bench.db")
	ctx := context.Background()
	st, err := Open(ctx, db)
	if err != nil {
		b.Fatal(err)
	}
	r := rand.New(rand.NewSource(42))
	rows := make([]PendingChunk, n)
	for i := range n {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		rows[i] = PendingChunk{
			Path:       "f" + strconv.Itoa(i) + ".go",
			Kind:       "function_declaration",
			ContentSHA: "sha" + strconv.Itoa(i),
			Content:    "func F" + strconv.Itoa(i) + "() {}",
			Vec:        v,
		}
	}
	if err := st.UpsertMany(ctx, rows, time.Now()); err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = st.Close() })
	return st
}

func BenchmarkSearch_1k_16(b *testing.B)   { benchSearch(b, 1000, 16) }
func BenchmarkSearch_5k_1024(b *testing.B) { benchSearch(b, 5000, 1024) }
func BenchmarkSearch_20k_1024(b *testing.B) {
	if testing.Short() {
		b.Skip()
	}
	benchSearch(b, 20000, 1024)
}
func BenchmarkSearch_50k_1024(b *testing.B) {
	if testing.Short() {
		b.Skip()
	}
	benchSearch(b, 50000, 1024)
}
func BenchmarkSearch_100k_1024(b *testing.B) {
	if testing.Short() {
		b.Skip()
	}
	benchSearch(b, 100000, 1024)
}
func BenchmarkSearch_50k_2560(b *testing.B) {
	if testing.Short() {
		b.Skip()
	}
	benchSearch(b, 50000, 2560) // Qwen3-Embedding-4B-shaped vectors
}

func benchSearch(b *testing.B, n, dim int) {
	st := benchStore(b, n, dim)
	r := rand.New(rand.NewSource(0))
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(r.NormFloat64())
	}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := st.Search(ctx, q, "", 8); err != nil {
			b.Fatal(err)
		}
	}
}
