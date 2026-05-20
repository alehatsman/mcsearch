package store

// Graph-side persistence (graph_nodes / graph_edges). Lives in its own
// file purely for navigability; the methods are on *Store and share the
// same migrations, *sql.DB, and transactional discipline as the chunk
// side. The package-level doc lives in store.go.

import (
	"context"
	"database/sql"
	"fmt"
	"time"
)

// GraphNodeRow mirrors one graph_nodes row. It exists to keep this
// package free of an import on internal/graph (which would otherwise
// be a cycle, since internal/graph depends on internal/store via the
// GraphStore interface). The graph package converts between its own
// graph.Node type and this row shape.
type GraphNodeRow struct {
	ID            string
	Kind          string
	Name          string
	QualifiedName string
	PackagePath   string
	FilePath      string
	StartLine     int
	EndLine       int
	ChunkID       int64 // 0 = NULL
	MetadataJSON  []byte
	ContentHash   string
	// Centrality columns. Populated by graph.ComputeCentrality after the
	// upsert pass. Zero on freshly-upserted rows; stays zero for nodes
	// the centrality computation skips (non-calls-graph nodes).
	InDegree        int
	OutDegree       int
	CrossPkgCallers int
	PageRank        float64
}

// GraphCentralityRow is the minimal slice of GraphNodeRow needed to
// update a node's centrality columns. Used by GraphSetCentrality so
// the centrality pass doesn't have to rewrite every other field.
type GraphCentralityRow struct {
	ID              string
	InDegree        int
	OutDegree       int
	CrossPkgCallers int
	PageRank        float64
}

// GraphEdgeRow mirrors one graph_edges row. Same rationale as
// GraphNodeRow.
type GraphEdgeRow struct {
	ID           string
	Kind         string
	SrcID        string
	DstID        string
	FilePath     string
	StartLine    int
	EndLine      int
	MetadataJSON []byte
	ContentHash  string
}

// ChunkLocation is the slice of chunks each graph node needs to
// resolve `chunk_id`. Returned by ChunksByPaths, consumed by the
// chunk-linkage pass in internal/graph.
type ChunkLocation struct {
	ID        int64
	Path      string
	StartLine int
	EndLine   int
}

// GraphUpsertNodes batch-upserts nodes in one transaction. Rows whose
// content_hash matches the existing row only bump last_seen_at, keeping
// the prune-by-cutoff pass correct without rewriting unchanged rows.
func (s *Store) GraphUpsertNodes(ctx context.Context, rows []GraphNodeRow, now time.Time) error {
	if len(rows) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO graph_nodes(
		   id, kind, name, qualified_name, package_path, file_path,
		   start_line, end_line, chunk_id, metadata_json, content_hash, last_seen_at
		 ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(id) DO UPDATE SET
		   kind          = excluded.kind,
		   name          = excluded.name,
		   qualified_name= excluded.qualified_name,
		   package_path  = excluded.package_path,
		   file_path     = excluded.file_path,
		   start_line    = excluded.start_line,
		   end_line      = excluded.end_line,
		   chunk_id      = excluded.chunk_id,
		   metadata_json = excluded.metadata_json,
		   content_hash  = excluded.content_hash,
		   last_seen_at  = excluded.last_seen_at`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	defer stmt.Close()
	ts := now.UnixNano()
	for _, r := range rows {
		var chunkID any
		if r.ChunkID > 0 {
			chunkID = r.ChunkID
		} else {
			chunkID = nil
		}
		if _, err := stmt.ExecContext(ctx,
			r.ID, r.Kind, r.Name, r.QualifiedName, r.PackagePath, r.FilePath,
			r.StartLine, r.EndLine, chunkID, string(r.MetadataJSON), r.ContentHash, ts); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("upsert node %s: %w", r.ID, err)
		}
	}
	return tx.Commit()
}

// GraphUpsertEdges batch-upserts edges in one transaction. Same
// content_hash/last_seen_at discipline as nodes.
func (s *Store) GraphUpsertEdges(ctx context.Context, rows []GraphEdgeRow, now time.Time) error {
	if len(rows) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO graph_edges(
		   id, kind, src_id, dst_id, file_path,
		   start_line, end_line, metadata_json, content_hash, last_seen_at
		 ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(id) DO UPDATE SET
		   kind          = excluded.kind,
		   src_id        = excluded.src_id,
		   dst_id        = excluded.dst_id,
		   file_path     = excluded.file_path,
		   start_line    = excluded.start_line,
		   end_line      = excluded.end_line,
		   metadata_json = excluded.metadata_json,
		   content_hash  = excluded.content_hash,
		   last_seen_at  = excluded.last_seen_at`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	defer stmt.Close()
	ts := now.UnixNano()
	for _, r := range rows {
		if _, err := stmt.ExecContext(ctx,
			r.ID, r.Kind, r.SrcID, r.DstID, r.FilePath,
			r.StartLine, r.EndLine, string(r.MetadataJSON), r.ContentHash, ts); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("upsert edge %s: %w", r.ID, err)
		}
	}
	return tx.Commit()
}

// GraphPruneUnseen deletes nodes and edges with last_seen_at strictly
// older than cutoff. Pair with a cutoff = Run start time so rows
// touched in the current pass survive.
func (s *Store) GraphPruneUnseen(ctx context.Context, cutoff time.Time) (nodes, edges int64, err error) {
	ts := cutoff.UnixNano()
	r, err := s.db.ExecContext(ctx, `DELETE FROM graph_edges WHERE last_seen_at < ?`, ts)
	if err != nil {
		return 0, 0, err
	}
	edges, _ = r.RowsAffected()
	r, err = s.db.ExecContext(ctx, `DELETE FROM graph_nodes WHERE last_seen_at < ?`, ts)
	if err != nil {
		return 0, edges, err
	}
	nodes, _ = r.RowsAffected()
	return nodes, edges, nil
}

// GraphStats reports the current graph_nodes / graph_edges row counts.
func (s *Store) GraphStats(ctx context.Context) (nodes, edges int64, err error) {
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM graph_nodes`).Scan(&nodes); err != nil {
		return 0, 0, err
	}
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM graph_edges`).Scan(&edges); err != nil {
		return nodes, 0, err
	}
	return nodes, edges, nil
}

// GraphAllNodes streams every node row. Used by exporters and tests; a
// real query API arrives in a follow-up layer.
func (s *Store) GraphAllNodes(ctx context.Context) ([]GraphNodeRow, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, kind, name, qualified_name, package_path, file_path,
		        start_line, end_line, COALESCE(chunk_id, 0), metadata_json, content_hash,
		        in_degree, out_degree, cross_pkg_callers, pagerank
		   FROM graph_nodes
		  ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []GraphNodeRow
	for rows.Next() {
		var r GraphNodeRow
		var md string
		if err := rows.Scan(&r.ID, &r.Kind, &r.Name, &r.QualifiedName, &r.PackagePath, &r.FilePath,
			&r.StartLine, &r.EndLine, &r.ChunkID, &md, &r.ContentHash,
			&r.InDegree, &r.OutDegree, &r.CrossPkgCallers, &r.PageRank); err != nil {
			return nil, err
		}
		r.MetadataJSON = []byte(md)
		out = append(out, r)
	}
	return out, rows.Err()
}

// GraphSetCentrality batch-updates centrality columns on graph_nodes.
// Run after GraphUpsertNodes / GraphUpsertEdges have settled — the
// caller computes degrees + PageRank from the in-memory edge slice and
// writes the result here in a single transaction. Rows whose ID is not
// in the table are silently ignored (UPDATE is a no-op).
func (s *Store) GraphSetCentrality(ctx context.Context, rows []GraphCentralityRow) error {
	if len(rows) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx,
		`UPDATE graph_nodes
		    SET in_degree         = ?,
		        out_degree        = ?,
		        cross_pkg_callers = ?,
		        pagerank          = ?
		  WHERE id = ?`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	defer stmt.Close()
	for _, r := range rows {
		if _, err := stmt.ExecContext(ctx, r.InDegree, r.OutDegree, r.CrossPkgCallers, r.PageRank, r.ID); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("set centrality %s: %w", r.ID, err)
		}
	}
	return tx.Commit()
}

// GraphAllEdges streams every edge row.
func (s *Store) GraphAllEdges(ctx context.Context) ([]GraphEdgeRow, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, kind, src_id, dst_id, file_path, start_line, end_line, metadata_json, content_hash
		   FROM graph_edges
		  ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []GraphEdgeRow
	for rows.Next() {
		var r GraphEdgeRow
		var md string
		if err := rows.Scan(&r.ID, &r.Kind, &r.SrcID, &r.DstID, &r.FilePath,
			&r.StartLine, &r.EndLine, &md, &r.ContentHash); err != nil {
			return nil, err
		}
		r.MetadataJSON = []byte(md)
		out = append(out, r)
	}
	return out, rows.Err()
}

// ChunksByPaths returns (id, start_line, end_line) for every chunk
// under any of paths. Used by the graph extractor's chunk-linkage
// pass; batched the same way as ExistingSHAsBatch to stay within
// SQLite's parameter limit.
func (s *Store) ChunksByPaths(ctx context.Context, paths []string) (map[string][]ChunkLocation, error) {
	out := make(map[string][]ChunkLocation, len(paths))
	if len(paths) == 0 {
		return out, nil
	}
	const batchSize = 500
	for i := 0; i < len(paths); i += batchSize {
		end := min(i+batchSize, len(paths))
		slice := paths[i:end]
		args := make([]any, len(slice))
		for j, p := range slice {
			args[j] = p
		}
		rows, err := s.db.QueryContext(ctx,
			`SELECT id, path, start_line, end_line FROM chunks WHERE path IN (`+inPlaceholders(len(slice))+`)`,
			args...)
		if err != nil {
			return nil, err
		}
		if err := scanChunkLocations(rows, out); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func scanChunkLocations(rows *sql.Rows, out map[string][]ChunkLocation) error {
	defer rows.Close()
	for rows.Next() {
		var c ChunkLocation
		if err := rows.Scan(&c.ID, &c.Path, &c.StartLine, &c.EndLine); err != nil {
			return err
		}
		out[c.Path] = append(out[c.Path], c)
	}
	return rows.Err()
}
