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

// GraphSymbol carries the columns the guide renderer needs to list
// exported declarations and top-centrality entry points. QualifiedName
// is preferred over Name for display because methods carry their
// receiver type there (e.g. "Store.Search" vs bare "Search").
type GraphSymbol struct {
	Name          string
	QualifiedName string
	Kind          string // function|method|struct|interface|type
	FilePath      string
	StartLine     int
	EndLine       int
	PageRank      float64
	InDegree      int
}

// dirLikePattern returns the SQL LIKE pattern that matches files
// directly under relDir (one level deep, no nested subdirs). For
// relDir="internal/store" the pattern is "internal/store/%" and the
// caller adds an additional `NOT LIKE 'internal/store/%/%'` clause.
func dirLikePattern(relDir string) string { return relDir + "/%" }
func nestedExclude(relDir string) string  { return relDir + "/%/%" }

// ExportedSymbolsByDir returns Go-exported declarations whose source
// file lives directly under relDir (not nested subdirectories).
// Filters to declarable kinds (function, method, struct, interface,
// type) and to capitalized names. Ordered by name. Used by the guide
// renderer to ground "Exported API" sections in real symbols instead
// of LLM-invented ones.
func (s *Store) ExportedSymbolsByDir(ctx context.Context, relDir string) ([]GraphSymbol, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT name, qualified_name, kind, file_path, start_line, end_line, pagerank, in_degree
		FROM graph_nodes
		WHERE file_path LIKE ?
		  AND file_path NOT LIKE ?
		  AND kind IN ('function','method','struct','interface','type')
		  AND substr(name,1,1) BETWEEN 'A' AND 'Z'
		ORDER BY name`,
		dirLikePattern(relDir), nestedExclude(relDir))
	if err != nil {
		return nil, err
	}
	return scanSymbols(rows)
}

// TopCentralByDir returns the top-k functions and methods under relDir
// sorted by PageRank then by in_degree. Used by the renderer to surface
// the "key entry points" of a module — the ground-truth answer to
// "what should I read first?".
func (s *Store) TopCentralByDir(ctx context.Context, relDir string, k int) ([]GraphSymbol, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT name, qualified_name, kind, file_path, start_line, end_line, pagerank, in_degree
		FROM graph_nodes
		WHERE file_path LIKE ?
		  AND file_path NOT LIKE ?
		  AND kind IN ('function','method')
		ORDER BY pagerank DESC, in_degree DESC, name ASC
		LIMIT ?`,
		dirLikePattern(relDir), nestedExclude(relDir), k)
	if err != nil {
		return nil, err
	}
	return scanSymbols(rows)
}

func scanSymbols(rows *sql.Rows) ([]GraphSymbol, error) {
	defer func() { _ = rows.Close() }()
	var out []GraphSymbol
	for rows.Next() {
		var g GraphSymbol
		if err := rows.Scan(&g.Name, &g.QualifiedName, &g.Kind, &g.FilePath, &g.StartLine, &g.EndLine, &g.PageRank, &g.InDegree); err != nil {
			return nil, err
		}
		out = append(out, g)
	}
	return out, rows.Err()
}

// ImportsForDir returns the unique import targets of the Go packages
// whose source files live under relDir. import nodes carry their
// owning package in `package_path` (their `file_path` is empty —
// imports are a package-level fact, not per-file), so we resolve via
// the package set rather than filtering imports by file_path.
func (s *Store) ImportsForDir(ctx context.Context, relDir string) ([]string, error) {
	rows, err := s.db.QueryContext(ctx, `
		WITH our_pkgs AS (
		  SELECT DISTINCT package_path FROM graph_nodes
		  WHERE file_path LIKE ?
		    AND file_path NOT LIKE ?
		    AND package_path != ''
		)
		SELECT DISTINCT name FROM graph_nodes
		WHERE kind = 'import'
		  AND package_path IN (SELECT package_path FROM our_pkgs)
		ORDER BY name`,
		dirLikePattern(relDir), nestedExclude(relDir))
	if err != nil {
		return nil, err
	}
	return scanStringColumn(rows)
}

// UsedByPackages returns the Go package paths that import a package
// whose source files live under relDir. import nodes carry their
// importer in the package_path column (file_path is empty for them,
// since imports are a package-level fact), so the renderer-side
// caller strips the module prefix to display directories.
//
// Result excludes our_pkgs themselves so a package isn't shown as
// using itself.
func (s *Store) UsedByPackages(ctx context.Context, relDir string) ([]string, error) {
	rows, err := s.db.QueryContext(ctx, `
		WITH our_pkgs AS (
		  SELECT DISTINCT package_path FROM graph_nodes
		  WHERE file_path LIKE ?
		    AND file_path NOT LIKE ?
		    AND package_path != ''
		)
		SELECT DISTINCT package_path FROM graph_nodes
		WHERE kind = 'import'
		  AND name IN (SELECT package_path FROM our_pkgs)
		  AND package_path NOT IN (SELECT package_path FROM our_pkgs)
		  AND package_path != ''
		ORDER BY package_path`,
		dirLikePattern(relDir), nestedExclude(relDir))
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	return scanStringColumn(rows)
}

func scanStringColumn(rows *sql.Rows) ([]string, error) {
	defer func() { _ = rows.Close() }()
	var out []string
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, rows.Err()
}
