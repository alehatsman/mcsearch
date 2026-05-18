package graph

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// nodeJSON / edgeJSON are the on-the-wire shapes for export. They match
// the format the issue documents so callers can pipe `nodes.jsonl` /
// `edges.jsonl` into other tools without a custom parser.
type nodeJSON struct {
	ID            string         `json:"id"`
	Kind          string         `json:"kind"`
	Name          string         `json:"name"`
	QualifiedName string         `json:"qualified_name"`
	PackagePath   string         `json:"package_path,omitempty"`
	FilePath      string         `json:"file_path,omitempty"`
	StartLine     int            `json:"start_line,omitempty"`
	EndLine       int            `json:"end_line,omitempty"`
	ChunkID       int64          `json:"chunk_id,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

type edgeJSON struct {
	ID        string         `json:"id"`
	Kind      string         `json:"kind"`
	SrcID     string         `json:"src_id"`
	DstID     string         `json:"dst_id"`
	FilePath  string         `json:"file_path,omitempty"`
	StartLine int            `json:"start_line,omitempty"`
	EndLine   int            `json:"end_line,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// ExportJSONL writes nodes.jsonl and edges.jsonl into outDir (created
// 0o755 if missing). One JSON object per line; no trailing newline on
// the last line.
func ExportJSONL(ctx context.Context, s GraphStore, outDir string) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", outDir, err)
	}

	nodes, err := s.GraphAllNodes(ctx)
	if err != nil {
		return fmt.Errorf("read nodes: %w", err)
	}
	if err := writeNodesJSONL(filepath.Join(outDir, "nodes.jsonl"), nodes); err != nil {
		return err
	}

	edges, err := s.GraphAllEdges(ctx)
	if err != nil {
		return fmt.Errorf("read edges: %w", err)
	}
	return writeEdgesJSONL(filepath.Join(outDir, "edges.jsonl"), edges)
}

func writeNodesJSONL(path string, nodes []Node) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for _, n := range nodes {
		if err := enc.Encode(nodeJSON{
			ID:            n.ID,
			Kind:          string(n.Kind),
			Name:          n.Name,
			QualifiedName: n.QualifiedName,
			PackagePath:   n.PackagePath,
			FilePath:      n.FilePath,
			StartLine:     n.StartLine,
			EndLine:       n.EndLine,
			ChunkID:       n.ChunkID,
			Metadata:      n.Metadata,
		}); err != nil {
			return fmt.Errorf("encode node %s: %w", n.ID, err)
		}
	}
	return w.Flush()
}

func writeEdgesJSONL(path string, edges []Edge) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for _, e := range edges {
		if err := enc.Encode(edgeJSON{
			ID:        e.ID,
			Kind:      string(e.Kind),
			SrcID:     e.SrcID,
			DstID:     e.DstID,
			FilePath:  e.FilePath,
			StartLine: e.StartLine,
			EndLine:   e.EndLine,
			Metadata:  e.Metadata,
		}); err != nil {
			return fmt.Errorf("encode edge %s: %w", e.ID, err)
		}
	}
	return w.Flush()
}
