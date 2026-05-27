package store

import (
	"container/list"
	"encoding/binary"
	"hash/fnv"
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

// entryOf unwraps the list element's Value. The two-value form keeps
// errcheck happy; the assertion is infallible by construction since
// every PushFront writes a *lruEntry, but yielding ok=false on a stray
// type means a misbehaving caller gets a miss rather than a panic.
func entryOf(el *list.Element) (*lruEntry, bool) {
	e, ok := el.Value.(*lruEntry)
	return e, ok
}

func (c *rerankLRU) Get(key string) (rerankCached, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.index[key]
	if !ok {
		return rerankCached{}, false
	}
	c.ll.MoveToFront(el)
	if e, ok := entryOf(el); ok {
		return e.val, true
	}
	return rerankCached{}, false
}

func (c *rerankLRU) Put(key string, val rerankCached) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.index[key]; ok {
		if e, ok := entryOf(el); ok {
			e.val = val
		}
		c.ll.MoveToFront(el)
		return
	}
	el := c.ll.PushFront(&lruEntry{key: key, val: val})
	c.index[key] = el
	if c.ll.Len() > c.cap {
		oldest := c.ll.Back()
		if oldest != nil {
			c.ll.Remove(oldest)
			if e, ok := entryOf(oldest); ok {
				delete(c.index, e.key)
			}
		}
	}
}

// rerankCacheKey builds a stable key from (query, fused ids). The fused
// pool comes in score order — we sort by id so cosmetically different
// orderings of the same set hit the cache. FNV-64a is a non-cryptographic
// hash; the only failure mode is a key collision, which costs a wasted
// cache miss not a security breach.
func rerankCacheKey(query string, ids []int64) string {
	sorted := make([]int64, len(ids))
	copy(sorted, ids)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	h := fnv.New64a()
	_, _ = h.Write([]byte(query))
	_, _ = h.Write([]byte{0})
	var buf [8]byte
	for _, id := range sorted {
		binary.LittleEndian.PutUint64(buf[:], uint64(id))
		_, _ = h.Write(buf[:])
	}
	return strconv.FormatUint(h.Sum64(), 16)
}
