package graph

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/alehatsman/dex/internal/ignore"
)

// yamlPkg is the synthetic "package" used to namespace YAML-graph IDs
// away from the Go-graph rows in the same store. Go file nodes carry
// the real Go package path; YAML files have no package, so we group
// them under a fixed sentinel.
const yamlPkg = "_yaml"

// mooncakeRefKeys are the YAML keys whose scalar value points at another
// YAML file. Picked specifically for the mooncake config dialect — these
// are the only cross-file refs that appear in mooncake-managed dotfiles
// trees today. Generalising to Helm/k8s/GHA is a follow-up.
var mooncakeRefKeys = map[string]struct{}{
	"import":    {},
	"vars.load": {},
	"use":       {},
}

// yamlRefLine matches a single list-item line of the form
//
//   - import: ./path.yml
//   - vars.load: "./other.yml"
//     import: ../sibling.yml   # without leading dash
//
// Group 1 = key, group 2 = the rest of the line (path + optional
// trailing comment). The leading dash is optional so block-mapping
// children also match.
var yamlRefLine = regexp.MustCompile(`^\s*-?\s*(import|vars\.load|use)\s*:\s*(.+?)\s*$`)

// ExtractYAML walks projectRoot for .yml/.yaml files (honoring
// .gitignore + .dex-ignore via internal/ignore) and emits file
// nodes + `imports` edges for mooncake-style `import:` / `vars.load:`
// references. Returns an empty ExtractResult on a tree with no YAML.
func ExtractYAML(ctx context.Context, projectRoot string) (*ExtractResult, error) {
	matcher, err := ignore.New(projectRoot)
	if err != nil {
		return nil, fmt.Errorf("ignore.New: %w", err)
	}

	nodeSet := newNodeSet()
	edgeSet := newEdgeSet()
	res := &ExtractResult{}

	type pendingRef struct {
		srcFile string // relpath
		target  string // resolved relpath
		line    int
	}
	var refs []pendingRef
	knownFiles := make(map[string]struct{})

	walkErr := filepath.WalkDir(projectRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		rel, relErr := filepath.Rel(projectRoot, path)
		if relErr != nil {
			return relErr
		}
		if rel == "." {
			return nil
		}
		if matcher.Match(rel, d.IsDir()) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(rel))
		if ext != ".yml" && ext != ".yaml" {
			return nil
		}

		relSlash := filepath.ToSlash(rel)
		knownFiles[relSlash] = struct{}{}
		nodeSet.add(yamlFileNode(relSlash))

		fileRefs, scanErr := scanYAMLRefs(path)
		if scanErr != nil {
			res.Warnings = append(res.Warnings, fmt.Sprintf("%s: %v", relSlash, scanErr))
			return nil
		}
		dir := filepath.Dir(rel)
		for _, r := range fileRefs {
			resolved, ok := resolveYAMLRef(projectRoot, dir, r.target)
			if !ok {
				continue
			}
			refs = append(refs, pendingRef{srcFile: relSlash, target: resolved, line: r.line})
		}
		return nil
	})
	if walkErr != nil {
		return nil, walkErr
	}

	for _, r := range refs {
		// Skip refs whose target never appeared on disk — keeps the
		// graph free of dangling nodes (mooncake typos, refs into
		// ignored dirs).
		if _, ok := knownFiles[r.target]; !ok {
			continue
		}
		src := NodeID("", yamlPkg, NodeFile, r.srcFile)
		dst := NodeID("", yamlPkg, NodeFile, r.target)
		edgeSet.add(Edge{
			ID:        EdgeID(src, EdgeImports, dst, r.srcFile, r.line),
			Kind:      EdgeImports,
			SrcID:     src,
			DstID:     dst,
			FilePath:  r.srcFile,
			StartLine: r.line,
			EndLine:   r.line,
		})
	}

	res.Nodes = nodeSet.flatten()
	res.Edges = edgeSet.flatten()
	return res, nil
}

func yamlFileNode(relSlash string) Node {
	return Node{
		ID:            NodeID("", yamlPkg, NodeFile, relSlash),
		Kind:          NodeFile,
		Name:          filepath.Base(relSlash),
		QualifiedName: relSlash,
		PackagePath:   yamlPkg,
		FilePath:      relSlash,
		Metadata:      map[string]any{"language": "yaml"},
	}
}

type yamlRef struct {
	target string
	line   int
}

// scanYAMLRefs returns every mooncake-style ref found in path. Cheap
// line scanner — no YAML parser dependency, since mooncake refs are
// always single-key scalars on a line.
func scanYAMLRefs(path string) ([]yamlRef, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	var out []yamlRef
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 64*1024), 1024*1024)
	lineno := 0
	for s.Scan() {
		lineno++
		line := s.Text()
		if !strings.ContainsAny(line, ":") {
			continue
		}
		m := yamlRefLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		key := m[1]
		if _, ok := mooncakeRefKeys[key]; !ok {
			continue
		}
		val := stripYAMLValue(m[2])
		if val == "" {
			continue
		}
		out = append(out, yamlRef{target: val, line: lineno})
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// stripYAMLValue removes a trailing `# comment`, surrounding quotes,
// and whitespace. Returns "" for values that look templated (contain
// `{{`) since those can't be resolved statically.
func stripYAMLValue(v string) string {
	v = strings.TrimSpace(v)
	if i := strings.Index(v, " #"); i >= 0 {
		v = strings.TrimSpace(v[:i])
	}
	if len(v) >= 2 {
		first, last := v[0], v[len(v)-1]
		if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
			v = v[1 : len(v)-1]
		}
	}
	if strings.Contains(v, "{{") {
		return ""
	}
	return v
}

// resolveYAMLRef joins target against the source file's dir and confirms
// the result stays inside projectRoot. Returns the slash-form relpath
// and true on success.
func resolveYAMLRef(projectRoot, srcDir, target string) (string, bool) {
	if target == "" || filepath.IsAbs(target) {
		return "", false
	}
	joined := filepath.Clean(filepath.Join(srcDir, target))
	if strings.HasPrefix(joined, "..") {
		return "", false
	}
	// Re-anchor to projectRoot to catch ../ escapes.
	abs := filepath.Clean(filepath.Join(projectRoot, joined))
	rel, err := filepath.Rel(projectRoot, abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", false
	}
	return filepath.ToSlash(rel), true
}
