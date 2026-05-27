package store

import (
	"container/list"
	"crypto/sha1"
	"encoding/hex"
	"sort"
	"strconv"
	"sync"
)

// RerankCache memoizes rerank results across calls. The key is opaque
// (see rerankCacheKey); the value is the post-rerank scored list and a
// per-id score map so the cached call assembles identical Hit metadata
// to the uncached one.
//
// Implementations must be safe for concurrent Get/Put.
type RerankCache interface {
	Get(key string) (rerankCached, bool)
	Put(key string, val rerankCached)
}

// rerankCached holds the data store.rerank produces. Both fields are
// shared by reference — callers must not mutate the slices/maps they
// get back from Get.
type rerankCached struct {
	scored      []scored
	rerankScore map[int64]float32
}

// rerankLRU is a fixed-capacity LRU. 256 entries is plenty for an
// interactive session — at ~200 bytes per entry that's well under
// 100 KiB resident.
type rerankLRU struct {
	mu    sync.Mutex
	cap   int
	ll    *list.List               // back = LRU, front = MRU
	index map[string]*list.Element // key → element holding lruEntry
}

type lruEntry struct {
	key string
	val rerankCached
}

// newRerankLRU returns a fresh LRU with the given capacity.
func newRerankLRU(capacity int) *rerankLRU {
	if capacity <= 0 {
		capacity = 256
	}
	return &rerankLRU{
		cap:   capacity,
		ll:    list.New(),
		index: make(map[string]*list.Element, capacity),
	}
}

func (c *rerankLRU) Get(key string) (rerankCached, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.index[key]
	if !ok {
		return rerankCached{}, false
	}
	c.ll.MoveToFront(el)
	return el.Value.(*lruEntry).val, true
}

func (c *rerankLRU) Put(key string, val rerankCached) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.index[key]; ok {
		el.Value.(*lruEntry).val = val
		c.ll.MoveToFront(el)
		return
	}
	el := c.ll.PushFront(&lruEntry{key: key, val: val})
	c.index[key] = el
	if c.ll.Len() > c.cap {
		oldest := c.ll.Back()
		if oldest != nil {
			c.ll.Remove(oldest)
			delete(c.index, oldest.Value.(*lruEntry).key)
		}
	}
}

// rerankCacheKey builds a stable key from (query, fused ids). The fused
// pool comes in score order — we sort by id so cosmetically different
// orderings of the same set hit the cache.
func rerankCacheKey(query string, ids []int64) string {
	sorted := make([]int64, len(ids))
	copy(sorted, ids)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	h := sha1.New()
	h.Write([]byte(query))
	h.Write([]byte{0})
	for _, id := range sorted {
		h.Write([]byte(strconv.FormatInt(id, 10)))
		h.Write([]byte{','})
	}
	return hex.EncodeToString(h.Sum(nil))
}
