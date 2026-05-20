// Package graph extracts a Go-specific static code graph (packages,
// files, functions, methods, types, fields, imports + the structural
// edges between them) and persists it next to the chunk/vector index.
//
// Where the chunk index answers "what code is *about* X", the graph
// answers structural questions like "what methods belong to (*Store)",
// "what does package P import", "where does this type embed", "who
// calls this function". Edges emitted today: contains, imports,
// has_method, has_field, embeds, implements, calls (Go-only via
// go/types). `references` lands with the LSP-as-consumer work.
package graph

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"strconv"
	"time"

	"github.com/alehatsman/mcsearch/internal/proj"
)

// NodeKind enumerates the kinds of structural symbols the extractor
// emits. Stored as a TEXT column so adding new kinds in later layers
// (e.g. constant, variable) is a no-op migration.
type NodeKind string

const (
	NodePackage   NodeKind = "package"
	NodeFile      NodeKind = "file"
	NodeFunction  NodeKind = "function"
	NodeMethod    NodeKind = "method"
	NodeType      NodeKind = "type"
	NodeInterface NodeKind = "interface"
	NodeStruct    NodeKind = "struct"
	NodeField     NodeKind = "field"
	NodeImport    NodeKind = "import"
)

// EdgeKind enumerates the structural relationships emitted by the
// extractor. `references`, `returns`, and `parameter` are reserved
// for follow-up work but not emitted yet.
type EdgeKind string

const (
	EdgeContains   EdgeKind = "contains"
	EdgeImports    EdgeKind = "imports"
	EdgeHasMethod  EdgeKind = "has_method"
	EdgeHasField   EdgeKind = "has_field"
	EdgeEmbeds     EdgeKind = "embeds"
	EdgeImplements EdgeKind = "implements"
	// EdgeCalls is emitted per *ast.CallExpr in extractCalls. Src is
	// the enclosing function/method node; dst is the resolved callee
	// (function, method, or interface method). Cross-package edges
	// are emitted only when both endpoints are in the loaded set —
	// std lib and unindexed dependencies are skipped to keep the
	// graph navigable.
	EdgeCalls EdgeKind = "calls"
)

// Node is a structural symbol persisted in graph_nodes.
//
// ID is stable across runs: <module>::<pkg>::<kind>::<qualified-name>.
// Two runs that emit the "same" symbol therefore produce the same row
// (upsert), and renaming a symbol naturally drops the old row via the
// prune-by-last-seen-at pass.
type Node struct {
	ID            string
	Kind          NodeKind
	Name          string
	QualifiedName string
	PackagePath   string
	FilePath      string // relative to project root; "" for package-level nodes
	StartLine     int
	EndLine       int
	ChunkID       int64 // 0 = unlinked
	Metadata      map[string]any
}

// Edge is a directed relationship persisted in graph_edges.
type Edge struct {
	ID        string
	Kind      EdgeKind
	SrcID     string
	DstID     string
	FilePath  string
	StartLine int
	EndLine   int
	Metadata  map[string]any
}

// NodeID is the canonical PK for a graph node. Anything that can be
// referenced from another node or edge goes through here.
func NodeID(modulePath, pkgPath string, kind NodeKind, qualifiedName string) string {
	return modulePath + "::" + pkgPath + "::" + string(kind) + "::" + qualifiedName
}

// EdgeID derives the edge's primary key from its endpoints + location.
// Same edge re-emitted from re-extracting the same file collides on PK
// and upserts in place; an edge moving to a different line counts as
// a different edge (the prune pass cleans up the old row).
func EdgeID(srcID string, kind EdgeKind, dstID, filePath string, startLine int) string {
	h := sha1.New()
	h.Write([]byte(srcID))
	h.Write([]byte{0})
	h.Write([]byte(kind))
	h.Write([]byte{0})
	h.Write([]byte(dstID))
	h.Write([]byte{0})
	h.Write([]byte(filePath))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(startLine)))
	return hex.EncodeToString(h.Sum(nil))
}

// ContentHash returns a SHA-1 of the mutable fields. The upsert path
// uses it to skip touching rows whose payload is unchanged (only
// last_seen_at advances).
func (n Node) ContentHash() string {
	h := sha1.New()
	h.Write([]byte(n.Kind))
	h.Write([]byte{0})
	h.Write([]byte(n.QualifiedName))
	h.Write([]byte{0})
	h.Write([]byte(n.PackagePath))
	h.Write([]byte{0})
	h.Write([]byte(n.FilePath))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(n.StartLine)))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(n.EndLine)))
	h.Write([]byte{0})
	h.Write([]byte(strconv.FormatInt(n.ChunkID, 10)))
	h.Write([]byte{0})
	h.Write(marshalMetadata(n.Metadata))
	return hex.EncodeToString(h.Sum(nil))
}

func (e Edge) ContentHash() string {
	h := sha1.New()
	h.Write([]byte(e.Kind))
	h.Write([]byte{0})
	h.Write([]byte(e.SrcID))
	h.Write([]byte{0})
	h.Write([]byte(e.DstID))
	h.Write([]byte{0})
	h.Write([]byte(e.FilePath))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(e.StartLine)))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(e.EndLine)))
	h.Write([]byte{0})
	h.Write(marshalMetadata(e.Metadata))
	return hex.EncodeToString(h.Sum(nil))
}

// marshalMetadata returns a deterministic JSON encoding of m. Empty
// map and nil both encode to "{}" so they hash identically.
func marshalMetadata(m map[string]any) []byte {
	if len(m) == 0 {
		return []byte("{}")
	}
	b, err := json.Marshal(m)
	if err != nil {
		// Metadata values originate inside this package; an encode
		// failure is a programming error, not user input. Surface a
		// stable placeholder so the hash remains computable.
		return []byte(`{"error":"marshal"}`)
	}
	return b
}

// MarshalMetadata exports the deterministic JSON encoding used for
// persistence (so persist.go and export.go agree byte-for-byte).
func MarshalMetadata(m map[string]any) []byte { return marshalMetadata(m) }

// unmarshalMetadata is the inverse of marshalMetadata. Returns nil for
// empty objects so round-tripping an unset Metadata stays comparable.
func unmarshalMetadata(b []byte) map[string]any {
	if len(b) == 0 {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// nilLogger returns a discarding logger when Options.Logger is unset.
func nilLogger() *slog.Logger { return slog.New(slog.NewTextHandler(discardWriter{}, nil)) }

type discardWriter struct{}

func (discardWriter) Write(p []byte) (int, error) { return len(p), nil }

// Options influence the runtime behaviour of an Indexer.
type Options struct {
	// Verbose toggles per-package extraction logs.
	Verbose bool
	// Logger receives structured logs. Nil = discard.
	Logger *slog.Logger
}

// Stats summarises one Indexer.Run.
type Stats struct {
	Packages       int
	NodesUpserted  int64
	EdgesUpserted  int64
	NodesPruned    int64
	EdgesPruned    int64
	Warnings       []string
	Elapsed        time.Duration
	LinkedToChunks int // count of nodes whose ChunkID resolved
}

// Indexer orchestrates extraction → chunk-linkage → persist → prune.
type Indexer struct {
	project *proj.Project
	store   GraphStore
	opts    Options
	log     *slog.Logger
}

// New wires an Indexer. Caller retains ownership of the store.
func New(p *proj.Project, s GraphStore, opts Options) *Indexer {
	log := opts.Logger
	if log == nil {
		log = nilLogger()
	}
	return &Indexer{project: p, store: s, opts: opts, log: log}
}

// Run extracts the Go graph for the project's source tree, persists it,
// and prunes rows from files that no longer exist. Safe to invoke
// repeatedly; idempotent on unchanged sources.
func (g *Indexer) Run(ctx context.Context) (*Stats, error) {
	t0 := time.Now()
	result, err := ExtractGo(ctx, g.project.Root)
	if err != nil {
		return nil, fmt.Errorf("extract: %w", err)
	}
	yamlRes, err := ExtractYAML(ctx, g.project.Root)
	if err != nil {
		return nil, fmt.Errorf("extract yaml: %w", err)
	}
	result.Nodes = append(result.Nodes, yamlRes.Nodes...)
	result.Edges = append(result.Edges, yamlRes.Edges...)
	result.Warnings = append(result.Warnings, yamlRes.Warnings...)
	if g.opts.Verbose {
		g.log.Info("graph extracted",
			"packages", result.Packages,
			"nodes", len(result.Nodes),
			"edges", len(result.Edges),
			"warnings", len(result.Warnings))
	}

	linked, err := linkChunks(ctx, g.store, result.Nodes)
	if err != nil {
		return nil, fmt.Errorf("link chunks: %w", err)
	}

	if err := g.store.GraphUpsertNodes(ctx, result.Nodes, t0); err != nil {
		return nil, fmt.Errorf("upsert nodes: %w", err)
	}
	if err := g.store.GraphUpsertEdges(ctx, result.Edges, t0); err != nil {
		return nil, fmt.Errorf("upsert edges: %w", err)
	}
	prunedNodes, prunedEdges, err := g.store.GraphPruneUnseen(ctx, t0)
	if err != nil {
		return nil, fmt.Errorf("prune: %w", err)
	}

	return &Stats{
		Packages:       result.Packages,
		NodesUpserted:  int64(len(result.Nodes)),
		EdgesUpserted:  int64(len(result.Edges)),
		NodesPruned:    prunedNodes,
		EdgesPruned:    prunedEdges,
		Warnings:       result.Warnings,
		Elapsed:        time.Since(t0),
		LinkedToChunks: linked,
	}, nil
}
