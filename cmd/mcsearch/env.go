// `mcsearch env` — print effective configuration with sources.
//
// The CLI accepts ~24 MCSEARCH_* env vars; remembering which are set,
// which fell back to defaults, and which optional features are
// currently disabled is a chore. This subcommand answers that.
//
// The table below is the single source of truth for env-var docs;
// README.md and docs/tuning.md should reference it instead of
// duplicating the list. If you add a knob anywhere in the codebase,
// add the corresponding entry here so `mcsearch env` stays honest.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

// envVar declares one MCSEARCH_* knob the CLI honours.
//
//   - Default is the value the binary uses when the env var is unset.
//     Empty string + Disable=true means the feature is OFF until set.
//   - Group steers display: "core"/"chat"/"rerank"/"compress"/"draft"
//     show by default; "tuning" hides behind `--all`.
type envVar struct {
	Name    string
	Default string
	Doc     string
	Group   string
	Disable bool // empty value + this flag = the feature is disabled
}

var allEnvVars = []envVar{
	// core — every install touches these.
	{"MCSEARCH_EMBED_URL", "http://127.0.0.1:8082", "OpenAI-compatible /v1/embeddings base URL.", "core", false},
	{"MCSEARCH_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B", "Model name forwarded as `model`.", "core", false},
	{"MCSEARCH_INDEX_DIR", "~/.cache/mcsearch", "Where per-project index files live.", "core", false},

	// chat — required for generate / summarize_path / ask_codebase.
	{"MCSEARCH_CHAT_URL", "http://127.0.0.1:8081", "OpenAI-compatible /v1/chat/completions base URL.", "chat", false},
	{"MCSEARCH_CHAT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct", "Model for the chat leg.", "chat", false},

	// rerank — optional, off by default.
	{"MCSEARCH_RERANK_URL", "", "Rerank server base URL.", "rerank", true},
	{"MCSEARCH_RERANK_STYLE", "cohere", "Backend shape: cohere | chat.", "rerank", false},
	{"MCSEARCH_RERANK_MODEL", "BAAI/bge-reranker-v2-m3", "Model for the rerank leg.", "rerank", false},

	// compress — optional context-compression server.
	{"MCSEARCH_COMPRESS_URL", "", "Context-compression /v1/chat/completions server.", "compress", true},
	{"MCSEARCH_COMPRESS_MODEL", "<MCSEARCH_CHAT_MODEL>", "Model for the compress leg.", "compress", false},

	// draft — optional speculative-draft server for generate_code.
	{"MCSEARCH_DRAFT_URL", "", "Speculative-draft /v1/chat/completions server.", "draft", true},
	{"MCSEARCH_DRAFT_MODEL", "<MCSEARCH_CHAT_MODEL>", "Model for the draft leg.", "draft", false},

	// summary — optional override for the chat leg used during indexing
	// (file / chunk / package / repo summaries). Defaults to MCSEARCH_CHAT_*.
	{"MCSEARCH_SUMMARY_URL", "", "Chat server for index-time summaries (falls back to MCSEARCH_CHAT_URL).", "summary", true},
	{"MCSEARCH_SUMMARY_MODEL", "<MCSEARCH_CHAT_MODEL>", "Model for index-time summaries. Smaller is fine — outputs are 1–4 sentences.", "summary", false},

	// tuning — hidden unless --all. Most installs leave these alone.
	{"MCSEARCH_EMBED_BATCH", "32", "Max chunks per /v1/embeddings call.", "tuning", false},
	{"MCSEARCH_EMBED_TIMEOUT", "60s", "HTTP timeout per embed call.", "tuning", false},
	{"MCSEARCH_CHAT_TIMEOUT", "120s", "HTTP timeout per chat call.", "tuning", false},
	{"MCSEARCH_COMPRESS_TIMEOUT", "30s", "HTTP timeout per compress call.", "tuning", false},
	{"MCSEARCH_DRAFT_TIMEOUT", "120s", "HTTP timeout per draft call.", "tuning", false},
	{"MCSEARCH_RERANK_TIMEOUT", "5s", "HTTP timeout per rerank call.", "tuning", false},
	{"MCSEARCH_RERANK_POOL", "40", "Candidates fed to the reranker. Clamped to [1, 100].", "tuning", false},
	{"MCSEARCH_RERANK_CONCURRENCY", "4", "Parallel rerank goroutines (chat style only).", "tuning", false},
	{"MCSEARCH_SUMMARY_TIMEOUT", "120s", "HTTP timeout per index-time summary call.", "tuning", false},
	{"MCSEARCH_SUMMARY_CONCURRENCY", "4", "Parallel chunk-summary chat calls per file during indexing.", "tuning", false},
	{"MCSEARCH_CHUNK_SUMMARY_MIN_LINES", "30", "Minimum chunk size (lines) eligible for a per-chunk summary. Raise to cut summary volume.", "tuning", false},
	{"MCSEARCH_DISABLE_RERANK", "", "Set 1 to short-circuit rerank even when URL is set.", "tuning", false},
	{"MCSEARCH_DISABLE_VEC_CACHE", "", "Set 1 to skip the in-RAM vector cache.", "tuning", false},
	{"MCSEARCH_DISABLE_BM25", "", "Set 1 to disable the BM25 leg.", "tuning", false},
	{"MCSEARCH_MAX_HITS_PER_FILE", "", "Cap hits per file in search results (0 = no cap).", "tuning", false},
	{"MCSEARCH_ALLOW_PATHS", "", "Colon-separated path prefixes accepted outside git work trees.", "tuning", false},
}

// effVar is one resolved row for output: name, current value, where
// that value came from, and the documentation snippet.
type effVar struct {
	Name   string `json:"name"`
	Value  string `json:"value"`
	Source string `json:"source"` // env | default | unset | disabled
	Group  string `json:"group"`
	Doc    string `json:"doc"`
}

func resolveEnv(vars []envVar) []effVar {
	out := make([]effVar, 0, len(vars))
	for _, v := range vars {
		raw := os.Getenv(v.Name)
		var val, src string
		switch {
		case raw != "":
			val, src = raw, "env"
		case v.Default != "":
			val, src = v.Default, "default"
		case v.Disable:
			val, src = "", "disabled"
		default:
			val, src = "", "unset"
		}
		out = append(out, effVar{
			Name:   v.Name,
			Value:  val,
			Source: src,
			Group:  v.Group,
			Doc:    v.Doc,
		})
	}
	return out
}

func cmdEnv(_ context.Context, args []string) error {
	fs := flag.NewFlagSet("env", flag.ContinueOnError)
	setHelp(fs,
		"Print effective MCSEARCH_* configuration with sources (env|default|disabled|unset).",
		"mcsearch env [--all] [--doc] [--format=text|json]")
	format := fs.String("format", "text", "output format: text | json")
	showAll := fs.Bool("all", false, "include tuning knobs (default: core/chat/rerank/compress/draft only)")
	doc := fs.Bool("doc", false, "include doc strings in text output")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}

	resolved := resolveEnv(allEnvVars)
	if !*showAll {
		filtered := resolved[:0]
		for _, v := range resolved {
			if v.Group != "tuning" {
				filtered = append(filtered, v)
			}
		}
		resolved = filtered
	}

	switch *format {
	case "json":
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(resolved)
	case "", "text":
		printEnvText(resolved, *doc)
		return nil
	default:
		return fmt.Errorf("unknown format %q (want text|json)", *format)
	}
}

func printEnvText(vars []effVar, withDoc bool) {
	groupOrder := []string{"core", "chat", "rerank", "compress", "draft", "tuning"}
	byGroup := map[string][]effVar{}
	nameW, valW := 0, 0
	for _, v := range vars {
		byGroup[v.Group] = append(byGroup[v.Group], v)
		if len(v.Name) > nameW {
			nameW = len(v.Name)
		}
		display := v.Value
		if display == "" {
			display = "—"
		}
		if len(display) > valW {
			valW = len(display)
		}
	}
	first := true
	for _, g := range groupOrder {
		items := byGroup[g]
		if len(items) == 0 {
			continue
		}
		if !first {
			fmt.Println()
		}
		first = false
		fmt.Println(g)
		for _, v := range items {
			display := v.Value
			if display == "" {
				display = "—"
			}
			fmt.Printf("  %-*s  %-*s  (%s)", nameW, v.Name, valW, display, v.Source)
			if withDoc && v.Doc != "" {
				fmt.Printf("\n  %-*s  %s", nameW, "", v.Doc)
			}
			fmt.Println()
		}
	}
}
