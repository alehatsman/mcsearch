package mcp

import "testing"

func TestFormatRole(t *testing.T) {
	cases := []struct {
		name       string
		nameStr    string
		in, out, x int
		want       string
	}{
		{"all-zero stays empty", "Foo", 0, 0, 0, ""},
		{"high in_degree → central", "Indexer", 12, 4, 0, "central:12"},
		{"cross_pkg ≥ 2 → central with pkg suffix", "Builder", 3, 1, 4, "central:3/4pkg"},
		{"in_degree threshold + pkg suffix", "Hub", 5, 1, 2, "central:5/2pkg"},
		{"exported with no callers", "Public", 0, 2, 0, "exported-unused"},
		{"lower-case with no callers stays empty", "private", 0, 2, 0, ""},
		{"leaf: in>0, out=0", "helper", 2, 0, 0, "leaf"},
		{"middle of the road stays empty", "mid", 2, 3, 0, ""},
		{"cross_pkg=1 alone is not enough", "thin", 1, 1, 1, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := formatRole(tc.nameStr, tc.in, tc.out, tc.x)
			if got != tc.want {
				t.Errorf("formatRole(%q, in=%d, out=%d, pkg=%d) = %q, want %q",
					tc.nameStr, tc.in, tc.out, tc.x, got, tc.want)
			}
		})
	}
}
