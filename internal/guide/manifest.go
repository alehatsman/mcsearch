package guide

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

// Manifest tracks when the guide was last rendered. last_summary_seen_at
// is the max(last_seen_at) of every summary chunk that fed the render.
// If a fresh index pass touches any summary, that max ticks forward —
// which is how `dex guide` (incremental) decides whether to re-render.
type Manifest struct {
	Version            int   `json:"version"`
	LastSummarySeenAt  int64 `json:"last_summary_seen_at"`
	RenderedAt         int64 `json:"rendered_at"`
}

const manifestVersion = 1

func manifestPath(root string) string {
	return filepath.Join(root, ".dex", "llm_guide.manifest.json")
}

// ReadManifest returns the persisted manifest. A missing file yields a
// zero-value Manifest with no error — callers treat that as "first run".
func ReadManifest(root string) (Manifest, error) {
	var m Manifest
	data, err := os.ReadFile(manifestPath(root))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return m, nil
		}
		return m, err
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return m, err
	}
	return m, nil
}

// WriteManifest persists the manifest under .dex/. The directory is
// created if absent.
func WriteManifest(root string, m Manifest) error {
	m.Version = manifestVersion
	dir := filepath.Join(root, ".dex")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(manifestPath(root), append(data, '\n'), 0o644)
}
