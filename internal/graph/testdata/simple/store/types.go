package store

// DB embeds Inner to exercise the embeds edge.
type DB struct {
	Inner
}

// Inner holds the actual map. Two named fields exercise has_field;
// the unexported field exercises that visibility doesn't filter
// extraction (the graph cares about structure, not export status).
type Inner struct {
	Name string
	data map[string]string
}

func (i *Inner) lookup(key string) string {
	return i.data[key]
}

func (i *Inner) write(key, value string) {
	if i.data == nil {
		i.data = map[string]string{}
	}
	i.data[key] = value
}

// Reader is an interface with two methods plus an embedded interface,
// so the test can assert has_method + embeds on interfaces.
type Reader interface {
	Get(key string) string
	NameOf() string
}

// ReadCloser embeds Reader and adds a method.
type ReadCloser interface {
	Reader
	Close() error
}

// Impl is a concrete type whose pointer receiver implements Reader,
// used to assert the implements edge in graph tests.
type Impl struct {
	name string
	data map[string]string
}

func (im *Impl) Get(key string) string { return im.data[key] }
func (im *Impl) NameOf() string        { return im.name }
