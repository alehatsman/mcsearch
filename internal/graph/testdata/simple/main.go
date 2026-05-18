package main

import (
	"fmt"

	"example.com/simple/store"
)

func main() {
	s := store.NewStore()
	fmt.Println(s.Get("k"))
}
