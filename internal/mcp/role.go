package mcp

import (
	"fmt"
	"unicode"
)

// formatRole composes the compact role tag attached to symbol-shaped
// results (SearchHit.Role, CallSite.Role). The tag is meant to give an
// agent a 1-glance read on how the symbol sits in the call graph —
// without it having to call graph_callers or graph_callees first.
//
// Rules, in priority order:
//
//   "central:<in>/<pkg>pkg"  — in_degree ≥ 5 OR cross_pkg_callers ≥ 2.
//                              The headline tier: lots of callers, or
//                              callers spanning real package boundaries.
//                              The pkg suffix is omitted when 0.
//   "exported-unused"        — name begins with an upper-case rune
//                              (Go's exportedness rule) AND in_degree
//                              == 0. Useful for spotting dead public
//                              API and APIs only consumed externally.
//   "leaf"                   — out_degree == 0 AND in_degree > 0.
//                              The symbol is called but calls nothing
//                              indexed itself — typically a base-case
//                              helper or an io/syscall wrapper.
//   ""                       — unremarkable middle (the common case).
//                              Also the result when no graph node
//                              exists (all-zero centrality).
//
// Thresholds chosen empirically against this repo: in_degree=5 cleanly
// separates utility helpers from real domain symbols, and pkg=2 catches
// genuine cross-package APIs without flagging every type that happens
// to be referenced from one neighbour.
func formatRole(name string, inDegree, outDegree, crossPkg int) string {
	allZero := inDegree == 0 && outDegree == 0 && crossPkg == 0
	if allZero {
		return ""
	}
	if inDegree >= 5 || crossPkg >= 2 {
		if crossPkg > 0 {
			return fmt.Sprintf("central:%d/%dpkg", inDegree, crossPkg)
		}
		return fmt.Sprintf("central:%d", inDegree)
	}
	if inDegree == 0 && isExported(name) {
		return "exported-unused"
	}
	if outDegree == 0 && inDegree > 0 {
		return "leaf"
	}
	return ""
}

// isExported is Go's exportedness check — first rune is upper-case.
// Used as a heuristic across languages here; for non-Go projects the
// convention often doesn't match, so "exported-unused" simply won't
// fire (centrality stays zero anyway when no graph is indexed).
func isExported(name string) bool {
	for _, r := range name {
		return unicode.IsUpper(r)
	}
	return false
}
