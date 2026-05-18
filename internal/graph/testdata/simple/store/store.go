package store

// Store holds key/value state. Used by the graph extractor tests
// as a stand-in for a real type with methods + fields.
type Store struct {
	db *DB
}

// NewStore returns a Store with an empty DB.
func NewStore() *Store {
	return &Store{db: &DB{}}
}

// Get returns the value at key, or "" if absent.
func (s *Store) Get(key string) string {
	return s.db.Inner.lookup(key)
}

// Set writes value at key.
func (s *Store) Set(key, value string) {
	if s.db == nil {
		s.db = &DB{}
	}
	s.db.Inner.write(key, value)
}
