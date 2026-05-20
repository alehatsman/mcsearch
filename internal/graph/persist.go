package graph

import (
	"context"
	"sort"
	"time"

	"github.com/alehatsman/mcsearch/internal/store"
)

// GraphStore is the persistence surface the Indexer needs. Production
// uses StoreAdapter wrapping *store.Store; tests can supply an in-memory
// fake without depending on SQLite.
type GraphStore interface {
	GraphUpsertNodes(ctx context.Context, ns []Node, now time.Time) error
	GraphUpsertEdges(ctx context.Context, es []Edge, now time.Time) error
	GraphPruneUnseen(ctx context.Context, cutoff time.Time) (nodes, edges int64, err error)
	GraphStats(ctx context.Context) (nodes, edges int64, err error)
	GraphAllNodes(ctx context.Context) ([]Node, error)
	GraphAllEdges(ctx context.Context) ([]Edge, error)
	// GraphSetCentrality batch-updates centrality columns (in/out
	// degree, cross-package fanin, PageRank) on already-upserted nodes.
	// Run as the post-edge pass in Indexer.Run.
	GraphSetCentrality(ctx context.Context, rows []CentralityRow) error
	// ChunksByPaths returns (chunk_id, start_line, end_line) for every
	// chunk under any of paths. Used only by the chunk-linkage pass.
	ChunksByPaths(ctx context.Context, paths []string) (map[string][]ChunkLocation, error)
}

// CentralityRow is the slice of a node's columns that the centrality
// pass writes back to the store. Mirrors store.GraphCentralityRow so
// callers don't have to import internal/store transitively.
type CentralityRow struct {
	ID              string
	InDegree        int
	OutDegree       int
	CrossPkgCallers int
	PageRank        float64
}

// ChunkLocation is the minimal chunk projection the chunk-linkage pass
// needs. Mirrors store.ChunkLocation but lives in this package so
// callers don't have to import internal/store transitively.
type ChunkLocation struct {
	ID        int64
	StartLine int
	EndLine   int
}

// NewStoreAdapter wraps a *store.Store so it satisfies GraphStore.
func NewStoreAdapter(s *store.Store) GraphStore { return &storeAdapter{s: s} }

type storeAdapter struct{ s *store.Store }

func (a *storeAdapter) GraphUpsertNodes(ctx context.Context, ns []Node, now time.Time) error {
	rows := make([]store.GraphNodeRow, 0, len(ns))
	for _, n := range ns {
		rows = append(rows, nodeToRow(n))
	}
	return a.s.GraphUpsertNodes(ctx, rows, now)
}

func (a *storeAdapter) GraphUpsertEdges(ctx context.Context, es []Edge, now time.Time) error {
	rows := make([]store.GraphEdgeRow, 0, len(es))
	for _, e := range es {
		rows = append(rows, edgeToRow(e))
	}
	return a.s.GraphUpsertEdges(ctx, rows, now)
}

func (a *storeAdapter) GraphPruneUnseen(ctx context.Context, cutoff time.Time) (int64, int64, error) {
	return a.s.GraphPruneUnseen(ctx, cutoff)
}

func (a *storeAdapter) GraphStats(ctx context.Context) (int64, int64, error) {
	return a.s.GraphStats(ctx)
}

func (a *storeAdapter) GraphAllNodes(ctx context.Context) ([]Node, error) {
	rows, err := a.s.GraphAllNodes(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]Node, 0, len(rows))
	for _, r := range rows {
		out = append(out, rowToNode(r))
	}
	return out, nil
}

func (a *storeAdapter) GraphAllEdges(ctx context.Context) ([]Edge, error) {
	rows, err := a.s.GraphAllEdges(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]Edge, 0, len(rows))
	for _, r := range rows {
		out = append(out, rowToEdge(r))
	}
	return out, nil
}

func (a *storeAdapter) GraphSetCentrality(ctx context.Context, rows []CentralityRow) error {
	storeRows := make([]store.GraphCentralityRow, 0, len(rows))
	for _, r := range rows {
		storeRows = append(storeRows, store.GraphCentralityRow{
			ID:              r.ID,
			InDegree:        r.InDegree,
			OutDegree:       r.OutDegree,
			CrossPkgCallers: r.CrossPkgCallers,
			PageRank:        r.PageRank,
		})
	}
	return a.s.GraphSetCentrality(ctx, storeRows)
}

func (a *storeAdapter) ChunksByPaths(ctx context.Context, paths []string) (map[string][]ChunkLocation, error) {
	raw, err := a.s.ChunksByPaths(ctx, paths)
	if err != nil {
		return nil, err
	}
	out := make(map[string][]ChunkLocation, len(raw))
	for p, list := range raw {
		conv := make([]ChunkLocation, 0, len(list))
		for _, c := range list {
			conv = append(conv, ChunkLocation{ID: c.ID, StartLine: c.StartLine, EndLine: c.EndLine})
		}
		out[p] = conv
	}
	return out, nil
}

// nodeToRow / rowToNode / edgeToRow / rowToEdge translate between the
// in-memory model and the store row shape. Metadata is JSON-marshalled
// through graph.MarshalMetadata so encoding stays byte-identical to
// what ContentHash hashed.

func nodeToRow(n Node) store.GraphNodeRow {
	return store.GraphNodeRow{
		ID:            n.ID,
		Kind:          string(n.Kind),
		Name:          n.Name,
		QualifiedName: n.QualifiedName,
		PackagePath:   n.PackagePath,
		FilePath:      n.FilePath,
		StartLine:     n.StartLine,
		EndLine:       n.EndLine,
		ChunkID:       n.ChunkID,
		MetadataJSON:  MarshalMetadata(n.Metadata),
		ContentHash:   n.ContentHash(),
	}
}

func edgeToRow(e Edge) store.GraphEdgeRow {
	return store.GraphEdgeRow{
		ID:           e.ID,
		Kind:         string(e.Kind),
		SrcID:        e.SrcID,
		DstID:        e.DstID,
		FilePath:     e.FilePath,
		StartLine:    e.StartLine,
		EndLine:      e.EndLine,
		MetadataJSON: MarshalMetadata(e.Metadata),
		ContentHash:  e.ContentHash(),
	}
}

func rowToNode(r store.GraphNodeRow) Node {
	return Node{
		ID:              r.ID,
		Kind:            NodeKind(r.Kind),
		Name:            r.Name,
		QualifiedName:   r.QualifiedName,
		PackagePath:     r.PackagePath,
		FilePath:        r.FilePath,
		StartLine:       r.StartLine,
		EndLine:         r.EndLine,
		ChunkID:         r.ChunkID,
		Metadata:        unmarshalMetadata(r.MetadataJSON),
		InDegree:        r.InDegree,
		OutDegree:       r.OutDegree,
		CrossPkgCallers: r.CrossPkgCallers,
		PageRank:        r.PageRank,
	}
}

func rowToEdge(r store.GraphEdgeRow) Edge {
	return Edge{
		ID:        r.ID,
		Kind:      EdgeKind(r.Kind),
		SrcID:     r.SrcID,
		DstID:     r.DstID,
		FilePath:  r.FilePath,
		StartLine: r.StartLine,
		EndLine:   r.EndLine,
		Metadata:  unmarshalMetadata(r.MetadataJSON),
	}
}

// linkChunks resolves Node.ChunkID for function/method nodes whose
// line range fits inside an existing chunk. Mutates nodes in place.
// Returns the count of nodes that were successfully linked.
//
// Coverage is "chunk strictly covers node": chunk.StartLine <= node.StartLine
// AND chunk.EndLine >= node.EndLine. When multiple chunks cover the same
// node, the smallest (tightest) is preferred so a struct method links
// to its own chunk rather than the surrounding file-window chunk.
func linkChunks(ctx context.Context, s GraphStore, nodes []Node) (int, error) {
	// Collect unique file paths for nodes that can plausibly link.
	pathSet := map[string]struct{}{}
	for _, n := range nodes {
		if n.FilePath == "" {
			continue
		}
		if n.Kind != NodeFunction && n.Kind != NodeMethod {
			continue
		}
		pathSet[n.FilePath] = struct{}{}
	}
	if len(pathSet) == 0 {
		return 0, nil
	}
	paths := make([]string, 0, len(pathSet))
	for p := range pathSet {
		paths = append(paths, p)
	}
	sort.Strings(paths) // deterministic for reproducible test output

	chunksByPath, err := s.ChunksByPaths(ctx, paths)
	if err != nil {
		return 0, err
	}
	linked := 0
	for i := range nodes {
		n := &nodes[i]
		if n.FilePath == "" || (n.Kind != NodeFunction && n.Kind != NodeMethod) {
			continue
		}
		candidates, ok := chunksByPath[n.FilePath]
		if !ok {
			continue
		}
		bestID := int64(0)
		bestSpan := int(^uint(0) >> 1) // max int
		for _, c := range candidates {
			if c.StartLine > n.StartLine || c.EndLine < n.EndLine {
				continue
			}
			span := c.EndLine - c.StartLine
			if span < bestSpan {
				bestSpan = span
				bestID = c.ID
			}
		}
		if bestID > 0 {
			n.ChunkID = bestID
			linked++
		}
	}
	return linked, nil
}
