package graph

import (
	"context"
	"testing"
)

func TestExtractYAMLMooncake(t *testing.T) {
	root := copyFixture(t, "mooncake")
	res, err := ExtractYAML(context.Background(), root)
	if err != nil {
		t.Fatalf("ExtractYAML: %v", err)
	}

	wantFiles := []string{
		"main.yml",
		"shared/variables.yml",
		"machines/main/vars.yml",
		"machines/main/index.yml",
	}
	for _, qn := range wantFiles {
		if findNode(res.Nodes, NodeFile, qn) == nil {
			t.Errorf("missing file node %s; files=%v", qn, nodesOfKind(res.Nodes, NodeFile))
		}
	}

	main := NodeID("", yamlPkg, NodeFile, "main.yml")
	wantEdges := []string{
		"shared/variables.yml",
		"machines/main/vars.yml",
		"machines/main/index.yml",
	}
	for _, dstQN := range wantEdges {
		dst := NodeID("", yamlPkg, NodeFile, dstQN)
		if findEdge(res.Edges, EdgeImports, main, dst) == nil {
			t.Errorf("missing imports edge main.yml → %s", dstQN)
		}
	}

	// Dangling ref shared/missing.yml is silently skipped.
	dangling := NodeID("", yamlPkg, NodeFile, "shared/missing.yml")
	if findEdge(res.Edges, EdgeImports, main, dangling) != nil {
		t.Errorf("dangling ref shared/missing.yml emitted an edge; should be skipped")
	}
	if findNode(res.Nodes, NodeFile, "shared/missing.yml") != nil {
		t.Errorf("dangling ref shared/missing.yml emitted a node; should be skipped")
	}

	// Relative-up resolution: machines/main/index.yml → shared/variables.yml.
	innerSrc := NodeID("", yamlPkg, NodeFile, "machines/main/index.yml")
	innerDst := NodeID("", yamlPkg, NodeFile, "shared/variables.yml")
	if findEdge(res.Edges, EdgeImports, innerSrc, innerDst) == nil {
		t.Errorf("missing imports edge machines/main/index.yml → shared/variables.yml")
	}
}

func TestExtractGoNoModule(t *testing.T) {
	// Reuse mooncake fixture — no go.mod present.
	root := copyFixture(t, "mooncake")
	res, err := ExtractGo(context.Background(), root)
	if err != nil {
		t.Fatalf("ExtractGo: %v", err)
	}
	if len(res.Nodes) != 0 || len(res.Edges) != 0 || len(res.Warnings) != 0 || res.Packages != 0 {
		t.Errorf("expected empty ExtractResult on non-Go tree; got packages=%d nodes=%d edges=%d warnings=%v",
			res.Packages, len(res.Nodes), len(res.Edges), res.Warnings)
	}
}
